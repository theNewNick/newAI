import os
import io
import uuid
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
import config

# If you need session for cross-referencing user session or IDs:
# from flask import session

# Define the blueprint here
system2_bp = Blueprint('system2_bp', __name__, template_folder='templates')

# Configure logging
import logging
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
OPENAI_API_KEY = config.OPENAI_API_KEY
PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_ENVIRONMENT = config.PINECONE_ENVIRONMENT
PINECONE_INDEX_NAME = config.PINECONE_INDEX_NAME

openai.api_key = OPENAI_API_KEY

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
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        logger.debug("Completed text preprocessing")
        return text
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}", exc_info=True)
        raise


def split_text_into_chunks(text, max_tokens=500):
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
            if token_count + sentence_token_count <= max_tokens:
                chunk += ' ' + sentence
                token_count += sentence_token_count
            else:
                chunks.append(chunk.strip())
                chunk = sentence
                token_count = sentence_token_count

        if chunk:
            chunks.append(chunk.strip())

        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {e}", exc_info=True)
        raise


def generate_embeddings(text_chunks, document_id):
    logger.debug(f"Entering generate_embeddings for document_id: {document_id}")
    data = []
    batch_size = 100
    for i in range(0, len(text_chunks), batch_size):
        batch_chunks = text_chunks[i:i + batch_size]
        try:
            logger.debug(f"Generating embeddings for batch {i // batch_size + 1}")
            response = openai.Embedding.create(
                input=batch_chunks,
                model='text-embedding-ada-002'
            )
            logger.debug(f"OpenAI response: {response}")

            if 'data' not in response or not response['data']:
                logger.error("OpenAI response missing 'data' field or it's empty.")
                raise ValueError("Invalid response from OpenAI API: 'data' field is missing or empty.")

            for j, embedding_info in enumerate(response['data']):
                embedding = embedding_info['embedding']
                chunk = batch_chunks[j]
                chunk_index = i + j
                # vector_id => You could also store user_id or analysis_id if you want
                vector_id = f"{document_id}_chunk_{chunk_index}"
                metadata = {
                    'document_id': document_id,
                    'chunk_index': chunk_index,
                    'text': chunk
                }
                data.append((vector_id, embedding, metadata))
                logger.debug(f"Generated embedding for chunk {chunk_index}")
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"General exception in generate_embeddings: {e}", exc_info=True)
            raise
    logger.debug(f"Generated embeddings for {len(data)} chunks")
    return data


def upsert_embeddings(pinecone_index, data):
    logger.debug("Entering upsert_embeddings")
    try:
        batch_size = 100
        for i in range(0, len(data), batch_size):
            to_upsert = data[i:i + batch_size]
            vectors = [
                {
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                }
                for vector_id, embedding, metadata in to_upsert
            ]
            pinecone_index.upsert(vectors=vectors)
            logger.debug(f"Upserted batch {i // batch_size + 1} to Pinecone")
        logger.debug(f"Upserted total of {len(data)} embeddings")
    except Exception as e:
        logger.error(f"Error upserting embeddings to Pinecone: {e}", exc_info=True)
        raise


@system2_bp.route('/upload', methods=['POST'])
def upload_files():
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

            # Step 2: Download from S3 locally, process text, embeddings, upsert
            try:
                download_path = os.path.join(tempfile.gettempdir(), unique_filename)
                logger.debug(f"Downloading {unique_filename} from S3 to {download_path}")
                download_pdf_from_s3(S3_BUCKET_NAME, unique_filename, download_path)

                logger.debug(f"Extracting text from {download_path}")
                raw_text = extract_text_from_pdf(download_path)

                logger.debug("Preprocessing extracted text")
                clean_text = preprocess_text(raw_text)

                logger.debug("Splitting text into chunks")
                text_chunks = split_text_into_chunks(clean_text, max_tokens=500)

                logger.debug("Generating embeddings for text chunks")
                document_id = unique_filename
                embedding_data = generate_embeddings(text_chunks, document_id)

                logger.debug("Upserting embeddings to Pinecone")
                upsert_embeddings(pinecone_index, embedding_data)

                logger.debug("Cleaning up temporary files")
                if os.path.exists(download_path):
                    os.remove(download_path)
                    logger.debug(f"Deleted {download_path}")

                uploaded_files.append({
                    'original_filename': filename,
                    'stored_filename': unique_filename
                })
            except Exception as e:
                logger.error(f"Error processing PDF {filename}: {e}", exc_info=True)
                return jsonify({'error': f'File processing failed for {filename}'}), 500

        logger.debug("All files uploaded and processed successfully")
        return jsonify({
            'message': 'Files uploaded and processed successfully',
            'files': uploaded_files
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
        response = openai.Embedding.create(
            input=[query],
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
        # By default, we filter on 'document_id'—but you could filter by user_id or analysis_id if needed
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
            {"role": "system", "content": "You are an AI assistant that provides helpful answers based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]
        response = openai.ChatCompletion.create(
            model='gpt-4',  # Updated from gpt-3.5-turbo
            messages=messages,
            max_tokens=400,  # Optionally increased from 200
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
