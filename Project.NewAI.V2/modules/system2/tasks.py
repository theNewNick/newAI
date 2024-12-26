# modules/system2/tasks.py

import os
import re
import logging
import pdfplumber
import nltk
import pinecone
import boto3
from celery import shared_task
from smart_load_balancer import call_openai_embedding_smart

logger = logging.getLogger(__name__)

@shared_task
def process_pdf_chunks_task(bucket_name, object_key, pinecone_index_name):
    """
    This is the Celery task that was referenced in handlers.py:
      from modules.system2.tasks import process_pdf_chunks_task

    1) Download PDF from S3
    2) Extract + preprocess + chunk text
    3) Generate embeddings for each chunk
    4) Upsert to Pinecone
    """
    try:
        # 1) Download from S3
        s3 = boto3.client('s3')
        local_pdf_path = f"/tmp/{object_key}"
        s3.download_file(bucket_name, object_key, local_pdf_path)

        # 2) Extract text with pdfplumber
        text = ""
        with pdfplumber.open(local_pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    logger.warning(f"No text on page {page_num}")

        # 3) Preprocess + chunk text
        nltk.data.path.append(os.path.join(os.path.expanduser('~'), 'nltk_data'))
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', download_dir=os.path.join(os.path.expanduser('~'), 'nltk_data'))

        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)

        max_tokens = 500
        chunks = []
        chunk = ""
        token_count = 0

        for sentence in sentences:
            words = sentence.split()
            if token_count + len(words) <= max_tokens:
                chunk += " " + sentence
                token_count += len(words)
            else:
                if chunk.strip():
                    chunks.append(chunk.strip())
                chunk = sentence
                token_count = len(words)
        if chunk.strip():
            chunks.append(chunk.strip())

        # 4) Initialize Pinecone + embed + upsert
        pinecone.init(api_key=os.getenv('PINECONE_API_KEY',''), environment=os.getenv('PINECONE_ENVIRONMENT',''))
        index = pinecone.Index(pinecone_index_name)

        vectors = []
        for i, c_text in enumerate(chunks):
            resp = call_openai_embedding_smart(
                input_list=[c_text],
                model='text-embedding-ada-002'
            )
            embedding = resp['data'][0]['embedding']
            vector_id = f"{object_key}_chunk_{i}"
            metadata = {"document_id": object_key, "chunk_index": i, "text": c_text}
            vectors.append({"id": vector_id, "values": embedding, "metadata": metadata})

        batch_size = 50
        for start in range(0, len(vectors), batch_size):
            subset = vectors[start:start+batch_size]
            index.upsert(vectors=subset)

        logger.info(f"Successfully processed {len(chunks)} chunks for {object_key}")
        return {"status": "success", "chunks_processed": len(chunks)}

    except Exception as e:
        logger.exception(f"Error in process_pdf_chunks_task for {object_key}: {e}")
        return {"status": "error", "error": str(e)}
