# modules/system2/handlers.py

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
import pinecone
from logging.handlers import RotatingFileHandler
import config  # <-- We now import config for multi-account approach

# If you need session for cross-referencing user session or IDs:
# from flask import session

# Import Celery tasks (the newly created tasks.py) so we can queue them
# This import statement assumes your Celery tasks are defined in modules/system2/tasks.py
# or a similar location. Adjust the path to match your actual structure.
from modules.system2.tasks import process_pdf_chunks_task

# Define the blueprint here
system2_bp = Blueprint('system2_bp', __name__, template_folder='templates')

# Configure logging
LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'system2.log')
handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=100000, backupCount=1)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(handler)

# Load environment variables via config
AWS_REGION = config.AWS_REGION
S3_BUCKET_NAME = config.S3_BUCKET_NAME

# Instead of openai.api_key = config.OPENAI_API_KEY, calls to openai.Embedding.create() or
# openai.ChatCompletion.create() will go through our load-balancer in config.py

PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_ENVIRONMENT = config.PINECONE_ENVIRONMENT
PINECONE_INDEX_NAME = config.PINECONE_INDEX_NAME

s3 = boto3.client('s3', region_name=AWS_REGION)

# Initialize Pinecone using the new client method
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)
pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)

NLTK_DATA_PATH = os.path.join(os.path.expanduser('~'), 'nltk_data')
nltk.data.path.append(NLTK_DATA_PATH)

try:
    nltk.data.find('tokenizers/punkt')
    logger.debug("NLTK 'punkt' tokenizer found")
except LookupError:
    logger.debug("NLTK 'punkt' tokenizer not found, downloading...")
    nltk.download('punkt', download_dir=NLTK_DATA_PATH)
    logger.debug("NLTK 'punkt' tokenizer downloaded")


##################################################################################
# If you created a wrapper for embeddings in config.py, e.g. call_openai_embedding_with_loadbalancer,
# you would import it here. Example:
##################################################################################
from config import call_openai_embedding_with_loadbalancer
# If you also have a ChatCompletion wrapper, e.g. call_gpt_4_with_loadbalancer,
# you could import that if you do any chat or summarization in system2.


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
    logger.debug("Entering preprocess_text")
    try:
        # Remove non-ASCII
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        logger.debug("Completed text preprocessing")
        return text
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}", exc_info=True)
        raise


def split_text_into_chunks(text, max_tokens=500):
    """
    Pre-chunk large documents: break them into smaller pieces of ~500 tokens each.
    """
    logger.debug("Entering split_text_into_chunks")
    try:
        sentences = sent_tokenize(text)
        logger.debug(f"Tokenized text into {len(sentences)} sentences")
        chunks = []
        chunk = ''
        token_count = 0

        for sentence in sentences:
            sentence_tokens = sentence.split()
            sentence_token_count = len(sentence_tokens)
            # If adding this sentence won't exceed max_tokens, add it to the current chunk.
            if token_count + sentence_token_count <= max_tokens:
                chunk += ' ' + sentence
                token_count += sentence_token_count
            else:
                # Start a new chunk
                chunks.append(chunk.strip())
                chunk = sentence
                token_count = sentence_token_count

        # Add the last chunk if non-empty
        if chunk:
            chunks.append(chunk.strip())

        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {e}", exc_info=True)
        raise


@system2_bp.route('/upload', methods=['POST'])
def upload_files():
    """
    This route now:
      1) Accepts PDF uploads
      2) Stores them in S3
      3) Triggers a Celery task to parse, pre-chunk, and embed the PDF
         in the background, preventing timeouts for large files.
    """
    logger.debug("Accessed upload_files route")
    try:
        if 'files' not in request.files:
            logger.error("No files part in the request")
            return jsonify({'error': 'No files part in the request'}), 400

        files = request.files.getlist('files')
        logger.debug(f"Received {len(files)} files")

        if len(files) == 0:
            logger.error("No files selected for uploading")
            return jsonify({'error': 'No files selected for uploading'}), 400

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
                logger.debug(f"Uploaded {unique_filename} to S3")
            except Exception as e:
                logger.error(f"Error uploading {filename} to S3: {e}", exc_info=True)
                return jsonify({'error': f'File upload failed for {filename}'}), 500

            # Instead of parsing/embedding here, we now queue Celery task:
            # The Celery task will:
            #   1) Download PDF from S3
            #   2) Extract + preprocess + chunk text
            #   3) Generate embeddings for each chunk
            #   4) Upsert to Pinecone
            task_result = process_pdf_chunks_task.delay(
                bucket_name=S3_BUCKET_NAME,
                object_key=unique_filename,
                pinecone_index_name=PINECONE_INDEX_NAME
            )
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

        # We'll generate a query embedding and query Pinecone to get relevant contexts
        query_embedding = generate_query_embedding(user_message)
        results = query_pinecone(pinecone_index, query_embedding, top_k=5, document_id=document_id)
        context_texts = [match['metadata']['text'] for match in results['matches']]

        assistant_response = get_response_from_openai(user_message, context_texts)

        logger.debug(f"Assistant response: {assistant_response}")
        return jsonify({'answer': assistant_response}), 200

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred while processing your request.'}), 500


def generate_query_embedding(query):
    logger.debug("Entering generate_query_embedding")
    try:
        response = call_openai_embedding_with_loadbalancer(
            input_list=[query],
            model='text-embedding-ada-002'
        )
        query_embedding = response['data'][0]['embedding']
        logger.debug("Generated query embedding")
        return query_embedding
    except openai.error.OpenAIError as e:
        logger.error(f"Error generating query embedding: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"General exception in generate_query_embedding: {e}", exc_info=True)
        raise


def query_pinecone(pinecone_index, query_embedding, top_k, document_id):
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
    logger.debug("Entering get_response_from_openai")
    try:
        context = "\n\n".join(context_texts)
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that provides helpful answers based on the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ]
        # If you want to load-balance chat calls, do:
        #   response = call_gpt_4_with_loadbalancer(messages)
        # Instead of direct openai.ChatCompletion.create
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=messages,
            max_tokens=400,
            temperature=0.7,
            n=1,
        )
        answer = response['choices'][0]['message']['content'].strip()
        logger.debug("Received response from OpenAI")
        return answer
    except openai.error.OpenAIError as e:
        logger.error(f"Error getting response from OpenAI: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"General exception in get_response_from_openai: {e}", exc_info=True)
        raise


@system2_bp.route('/view_document/<document_id>')
def view_document(document_id):
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
    Returns a JSON list of documents stored in S3 (limited to PDF files).
    You could display these in a dropdown on the UI
    so the user can pick a 'document_id' for the chatbot.
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


# -------------------------------------------------------------------------
# END OF FILE
# The heavy-lifting tasks are now offloaded to Celery (in tasks.py),
# so we only store the PDF in S3 and queue the job here.
# -------------------------------------------------------------------------
