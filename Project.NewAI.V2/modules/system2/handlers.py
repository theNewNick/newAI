import os
import io
import uuid
import logging
import boto3
import tempfile
import pdfplumber
import openai
import re
import nltk
from flask import request, jsonify, render_template, send_file, Blueprint
from werkzeug.utils import secure_filename
from datetime import datetime
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from logging.handlers import RotatingFileHandler
import config  # We still rely on config for AWS details, but no longer use round-robin from config.

# Import Celery tasks for chunking & embedding in the background
from modules.system2.tasks import process_pdf_chunks_task

# Import the "model_selector" helper
from model_selector import choose_model_for_task

# Import the “smart” load-balancer calls for chat & embedding
from smart_load_balancer import call_openai_smart, call_openai_embedding_smart

# NEW import for Pinecone 5.x
from pinecone import Pinecone, ServerlessSpec

# Define the blueprint
system2_bp = Blueprint('system2_bp', __name__, template_folder='templates')

# Configure logging to modules/system2/system2.log
LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'system2.log')
handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=100000, backupCount=1)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Clear existing handlers (avoid duplication) and add the new one
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(handler)

# Load environment variables (region, bucket, etc.)
load_dotenv()

AWS_REGION = config.AWS_REGION
S3_BUCKET_NAME = config.S3_BUCKET_NAME

# We no longer do openai.api_key = config.OPENAI_API_KEY
# Instead, calls go through our new "smart_load_balancer."

PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_ENVIRONMENT = config.PINECONE_ENVIRONMENT
PINECONE_INDEX_NAME = config.PINECONE_INDEX_NAME

s3 = boto3.client('s3', region_name=AWS_REGION)

# Initialize Pinecone using the 5.x client
pc = Pinecone(api_key=PINECONE_API_KEY)

# (Optional) check if index exists; if not, create it
existing_indexes = pc.list_indexes().names()
if PINECONE_INDEX_NAME not in existing_indexes:
    logger.info(f"Index '{PINECONE_INDEX_NAME}' not found. Creating it now...")
    # Provide the dimension that matches your embeddings (e.g. 1536 for 'text-embedding-ada-002')
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
else:
    logger.info(f"Index '{PINECONE_INDEX_NAME}' already exists.")

# Get a reference to the existing (or newly created) index
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

NLTK_DATA_PATH = os.path.join(os.path.expanduser('~'), 'nltk_data')
nltk.data.path.append(NLTK_DATA_PATH)

try:
    nltk.data.find('tokenizers/punkt')
    logger.debug("NLTK 'punkt' tokenizer found")
except LookupError:
    logger.debug("NLTK 'punkt' tokenizer not found, downloading...")
    nltk.download('punkt', download_dir=NLTK_DATA_PATH)
    logger.debug("NLTK 'punkt' tokenizer downloaded")


##############################################################################
# Helper functions for PDF handling, text preprocessing, etc.
# (Note: chunking & embeddings now typically occur in tasks.py)
##############################################################################

def download_pdf_from_s3(bucket_name, object_key, download_path):
    logger.debug(f"Entering download_pdf_from_s3 with object_key: {object_key}")
    try:
        s3.download_file(bucket_name, object_key, download_path)
        logger.debug(f"Downloaded {object_key} to {download_path}")
    except Exception as e:
        logger.error(f"Error downloading file from S3: {e}", exc_info=True)
        raise


def extract_text_from_pdf(pdf_path):
    logger.debug(f"Entering extract_text_from_pdf with pdf_path: {pdf_path}")
    text = ''
    try:
        with pdfplumber.open(pdf_path) as pdf:
            logger.debug(f"Opened PDF {pdf_path} with {len(pdf.pages)} pages")
            for page_num, page in enumerate(pdf.pages, start=1):
                logger.debug(f"Extracting text from page {page_num}")
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
                else:
                    logger.warning(f"No text found on page {page_num}")
        logger.debug("Completed text extraction from PDF")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}", exc_info=True)
        raise


def preprocess_text(text):
    """
    Basic text cleanup, removing non-ASCII and normalizing whitespace.
    """
    logger.debug("Entering preprocess_text")
    try:
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        text = re.sub(r'\s+', ' ', text).strip()
        logger.debug("Completed text preprocessing")
        return text
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}", exc_info=True)
        raise


# NOTE: We used to do chunking here with `split_text_into_chunks`. 
# However, we now do chunking + embedding in tasks.py (process_pdf_chunks_task).
# If you want to keep a local chunk function for smaller usage, you can, 
# but it won't be used in the new embedding pipeline.
def split_text_into_chunks(text, max_tokens=500):
    """
    Legacy approach: Splits text by sentences until ~500 words are reached.
    Kept here for reference, but the new pipeline in tasks.py uses 
    tiktoken-based token chunking and runs in Celery.
    """
    logger.debug("Entering split_text_into_chunks (legacy sentence-based approach)")
    try:
        sentences = sent_tokenize(text)
        logger.debug(f"Tokenized text into {len(sentences)} sentences")
        chunks = []
        chunk = ''
        token_count = 0

        for sentence in sentences:
            sentence_tokens = sentence.split()
            sentence_token_count = len(sentence_tokens)
            if token_count + sentence_token_count <= max_tokens:
                chunk += ' ' + sentence
                token_count += sentence_token_count
            else:
                if chunk.strip():
                    chunks.append(chunk.strip())
                chunk = sentence
                token_count = sentence_token_count

        if chunk:
            chunks.append(chunk.strip())

        logger.debug(f"Split text into {len(chunks)} chunks (sentence-based, ~{max_tokens} words each)")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {e}", exc_info=True)
        raise


##############################################################################
# Routes
##############################################################################

@system2_bp.route('/upload', methods=['POST'])
def upload_files():
    """
    1) Accept PDF uploads
    2) Store them in S3
    3) Queue Celery task (process_pdf_chunks_task) to parse, chunk, embed
    """
    logger.debug("Accessed upload_files route")
    logger.info("Entering upload_files() route in system2.handlers")
    try:
        if 'files' not in request.files:
            logger.error("No files part in the request")
            return jsonify({'error': 'No files part in the request'}), 400

        logger.info("upload_files(): request.files keys => %s", list(request.files.keys()))

        files = request.files.getlist('files')
        logger.debug(f"Received {len(files)} files")

        if not files:
            logger.error("No files selected for uploading")
            return jsonify({'error': 'No files selected for uploading'}), 400

        logger.info("Preparing to upload %d PDF(s) to S3 and enqueue tasks...", len(files))
        uploaded_files = []
        task_ids = []

        for file in files:
            logger.debug(f"Processing file: {file.filename}")
            if file.filename == '':
                logger.error("One of the files has no filename")
                return jsonify({'error': 'One of the files has no filename'}), 400

            if not file.filename.lower().endswith('.pdf'):
                logger.error(f"File {file.filename} is not a PDF")
                return jsonify({'error': 'Only PDF files are allowed'}), 400

            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            logger.debug(f"Unique filename generated: {unique_filename}")

            # Extra log before uploading
            logger.info("About to upload file '%s' to bucket '%s'", unique_filename, S3_BUCKET_NAME)

            # Step 1: Upload to S3
            try:
                logger.debug(f"Uploading {unique_filename} to S3")
                s3.upload_fileobj(
                    file,
                    S3_BUCKET_NAME,
                    unique_filename,
                    ExtraArgs={
                        'Metadata': {
                            'original_filename': filename,
                            'upload_time': datetime.utcnow().isoformat()
                        }
                    }
                )
                logger.info("File '%s' successfully uploaded to S3 bucket '%s'",
                            unique_filename, S3_BUCKET_NAME)
                logger.debug(f"Uploaded {unique_filename} to S3")
            except Exception as e:
                logger.error(f"Error uploading {filename} to S3: {e}", exc_info=True)
                return jsonify({'error': f'File upload failed for {filename}'}), 500

            # Queue the Celery task for background processing
            logger.info("Queueing Celery task for PDF: %s", unique_filename)

            task_result = process_pdf_chunks_task.delay(
                bucket_name=S3_BUCKET_NAME,
                object_key=unique_filename,
                pinecone_index_name=PINECONE_INDEX_NAME
            )
            # Extra log after .delay() to confirm the Task ID:
            logger.info("Celery task queued (Task ID=%s) for PDF: %s",
                        task_result.id, unique_filename)
            task_ids.append(task_result.id)

            uploaded_files.append({
                'original_filename': filename,
                'stored_filename': unique_filename
            })

        logger.debug("All files enqueued for background processing")
        return jsonify({
            'message': 'Files uploaded successfully. Processing in background...',
            'files': uploaded_files,
            'task_ids': task_ids
        }), 200

    except Exception as e:
        logger.error(f"Unhandled exception in upload_files: {e}", exc_info=True)
        return jsonify({'error': 'An internal error occurred'}), 500


@system2_bp.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint to handle chat requests. It:
      1) Accepts user_message & document_id
      2) Creates an embedding for user_message
      3) Queries Pinecone for the relevant context
      4) Calls GPT for a final answer
    """
    logger.debug("Accessed chat route in /chat")
    logger.debug("Accessed chat route")
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        document_id = data.get('document_id', '').strip()

        if not user_message:
            logger.error("Empty message received")
            return jsonify({'error': 'Empty message received'}), 400

        if not document_id:
            logger.error("No document selected")
            return jsonify({'error': 'No document selected'}), 400

        logger.debug(f"User message: {user_message}")
        logger.debug(f"Document ID: {document_id}")

        # 1) Generate a query embedding
        query_embedding = generate_query_embedding(user_message)

        # 2) Query Pinecone for relevant contexts
        results = query_pinecone(
            pinecone_index,
            query_embedding,
            top_k=5,
            document_id=document_id
        )
        context_texts = [match['metadata']['text'] for match in results['matches']]

        # 3) Get final answer from GPT
        assistant_response = get_response_from_openai(user_message, context_texts)

        logger.debug(f"Assistant response: {assistant_response}")
        return jsonify({'answer': assistant_response}), 200

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred while processing your request.'}), 500


def generate_query_embedding(query):
    """
    Replaces old round-robin embedding calls with the new smart LB approach,
    using choose_model_for_task("embedding").
    """
    logger.debug("Entering generate_query_embedding")
    try:
        embedding_model = choose_model_for_task("embedding")
        response = call_openai_embedding_smart(
            input_list=[query],
            model=embedding_model,
            max_retries=5
        )
        query_embedding = response['data'][0]['embedding']
        logger.debug("Generated query embedding via smart LB")
        return query_embedding
    except openai.error.OpenAIError as e:
        logger.error(f"Error generating query embedding: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"General exception in generate_query_embedding: {e}", exc_info=True)
        raise


def query_pinecone(pinecone_index, query_embedding, top_k, document_id):
    """
    Performs a vector similarity search in Pinecone, filtering by document_id.
    """
    logger.debug("Entering query_pinecone")
    try:
        query_filter = {'document_id': {'$eq': document_id}}
        response = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            filter=query_filter
        )
        logger.debug(f"Received {len(response['matches'])} matches from Pinecone")
        return response
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}", exc_info=True)
        raise


def get_response_from_openai(query, context_texts):
    """
    Calls GPT for a short Q&A summary of the retrieved context, 
    defaulting to gpt-3.5-turbo (short_summarization).
    """
    logger.debug("Entering get_response_from_openai")
    try:
        # Combine context chunks
        context = "\n\n".join(context_texts)

        # Provide strong instructions for brevity
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant. "
                    "Provide concise answers. Limit your response to 100 words or fewer. "
                    "Use bullet points if necessary, but keep it short."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{query}"
                )
            }
        ]

        # We choose "short_summarization" => gpt-3.5-turbo in model_selector
        chosen_model = choose_model_for_task("short_summarization")

        # Cap max_tokens to keep responses short
        response = call_openai_smart(
            messages=messages,
            model=chosen_model,
            temperature=0.7,
            max_tokens=300,
            max_retries=3
        )

        answer = response['choices'][0]['message']['content'].strip()
        logger.debug("Received response from GPT via smart LB")
        return answer

    except openai.error.OpenAIError as e:
        logger.error(f"Error getting response from OpenAI: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"General exception in get_response_from_openai: {e}", exc_info=True)
        raise


@system2_bp.route('/view_document/<document_id>')
def view_document(document_id):
    """
    Serves a PDF from S3 so users can view it in-browser.
    """
    logger.debug(f"Accessed view_document route with document_id: {document_id}")
    try:
        file_obj = io.BytesIO()
        s3.download_fileobj(S3_BUCKET_NAME, document_id, file_obj)
        file_obj.seek(0)
        logger.debug(f"Serving document {document_id}")
        return send_file(file_obj, mimetype='application/pdf')
    except Exception as e:
        logger.error(f"Error serving document {document_id}: {e}", exc_info=True)
        return 'Error retrieving document', 500


@system2_bp.route('/test_nltk')
def test_nltk():
    """
    Quick route to test if NLTK is working.
    """
    try:
        text = "This is a sentence. This is another sentence."
        sentences = sent_tokenize(text)
        logger.debug(f"Tokenized sentences: {sentences}")
        return jsonify({'sentences': sentences})
    except Exception as e:
        logger.error(f"Error testing NLTK: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@system2_bp.route('/list_documents', methods=['GET'])
def list_documents():
    """
    Returns a JSON list of PDF documents in S3, for use in a dropdown 
    of available docs to pass as document_id, etc.
    """
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME)
        if 'Contents' not in response:
            return jsonify([])

        documents = []
        for obj in response['Contents']:
            key = obj['Key']
            if key.lower().endswith('.pdf'):
                documents.append(key)

        return jsonify(documents)
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        return jsonify({'error': 'Failed to list documents.'}), 500
