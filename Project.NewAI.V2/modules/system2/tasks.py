import os
import re
import logging
import pdfplumber
import nltk
import boto3
import tiktoken

# Import the *same* Celery app we forcibly set to Redis
from celery_app import celery

from pinecone import Pinecone, ServerlessSpec
from smart_load_balancer import call_openai_embedding_smart

logger = logging.getLogger(__name__)

CHOSEN_EMBEDDING_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 2000

@celery.task
def process_pdf_chunks_task(bucket_name, object_key, pinecone_index_name):
    """
    This Celery task performs the following steps:
      1) Download a PDF from S3.
      2) Extract + preprocess text.
      3) Chunk the text based on token count (using tiktoken).
      4) Generate embeddings for each chunk using a single embedding model.
      5) Upsert the resulting vectors into Pinecone.
    """
    try:
        s3 = boto3.client('s3')
        local_pdf_path = f"/tmp/{object_key}"
        s3.download_file(bucket_name, object_key, local_pdf_path)

        text = ""
        with pdfplumber.open(local_pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    logger.warning(f"No text found on page {page_num}")

        # Basic cleanup
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Add NLTK data path
        nltk.data.path.append(os.path.join(os.path.expanduser('~'), 'nltk_data'))
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', download_dir=os.path.join(os.path.expanduser('~'), 'nltk_data'))

        # Tokenize with tiktoken
        encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        all_tokens = encoder.encode(text)

        chunks = []
        start_index = 0
        while start_index < len(all_tokens):
            end_index = min(start_index + CHUNK_SIZE, len(all_tokens))
            chunk_tokens = all_tokens[start_index:end_index]
            chunk_text = encoder.decode(chunk_tokens)
            chunks.append(chunk_text.strip())
            start_index += CHUNK_SIZE

        # Pinecone initialization
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY', ''))
        existing_indexes = pc.list_indexes().names()
        if pinecone_index_name not in existing_indexes:
            logger.info(f"Index '{pinecone_index_name}' not found. Creating...")
            pc.create_index(
                name=pinecone_index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        else:
            logger.info(f"Index '{pinecone_index_name}' already exists.")

        index = pc.Index(pinecone_index_name)

        vectors = []
        for i, c_text in enumerate(chunks):
            resp = call_openai_embedding_smart(
                input_list=[c_text],
                model=CHOSEN_EMBEDDING_MODEL
            )
            embedding = resp['data'][0]['embedding']
            vector_id = f"{object_key}_chunk_{i}"
            metadata = {
                "document_id": object_key,
                "chunk_index": i,
                "text": c_text
            }
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })

        batch_size = 50
        for start in range(0, len(vectors), batch_size):
            subset = vectors[start:start+batch_size]
            index.upsert(vectors=subset)

        logger.info(
            f"Successfully processed {len(chunks)} token-based chunks for {object_key}"
        )
        return {"status": "success", "chunks_processed": len(chunks)}

    except Exception as e:
        logger.exception(f"Error in process_pdf_chunks_task for {object_key}: {e}")
        return {"status": "error", "error": str(e)}
