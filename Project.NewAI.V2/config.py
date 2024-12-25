import os
import sys
import logging
import time

from dotenv import load_dotenv
import openai

# Load environment variables from .env
load_dotenv()

###############################################
# OPENAI MULTI-ACCOUNT KEYS (for load balancing)
###############################################
# Since you're not testing locally or using org IDs,
# we simply read multiple API keys. Adjust as needed.
OPENAI_API_KEY_1 = os.getenv('OPENAI_API_KEY_1', '')
OPENAI_API_KEY_2 = os.getenv('OPENAI_API_KEY_2', '')
OPENAI_API_KEY_3 = os.getenv('OPENAI_API_KEY_3', '')
OPENAI_API_KEY_4 = os.getenv('OPENAI_API_KEY_4', '')
OPENAI_API_KEY_5 = os.getenv('OPENAI_API_KEY_5', '')

OPENAI_ACCOUNTS = [
    {'api_key': OPENAI_API_KEY_1},
    {'api_key': OPENAI_API_KEY_2},
    {'api_key': OPENAI_API_KEY_3},
    {'api_key': OPENAI_API_KEY_4},
    {'api_key': OPENAI_API_KEY_5},
]

###############################################
# OTHER ENVIRONMENT VARIABLES
###############################################
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'YOUR_FLASK_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'your-bucket-name')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'your-pinecone-api-key')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east1-gcp')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'your-index-name')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', 'your-newsapi-key')

###############################################
# SQLALCHEMY CONFIGURATION
###############################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'feedback.db')}"
SECRET_KEY = FLASK_SECRET_KEY
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

###############################################
# LOGGING
###############################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

###############################################
# MULTI-ACCOUNT OPENAI API SETUP
###############################################
current_account_index = 0

def call_gpt_4_with_loadbalancer(
    messages,
    temperature=0.5,
    max_tokens=750,
    max_retries=5
):
    """
    Wrapper function to route GPT-4 requests among multiple accounts,
    reducing the chance of rate-limit errors on a single account.
    Round-robin approach: if we get a RateLimitError, move to the next account.
    """
    global current_account_index

    for attempt in range(max_retries):
        # Retrieve the current account's API key
        account = OPENAI_ACCOUNTS[current_account_index]
        openai.api_key = account['api_key']

        logger.debug(f"[GPT-4 LB] Attempt {attempt+1}/{max_retries} using account index={current_account_index}")

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response
        except openai.error.RateLimitError as e:
            logger.warning(f"[GPT-4 LB] Rate limit on account {current_account_index}. Switching accounts.")
            current_account_index = (current_account_index + 1) % len(OPENAI_ACCOUNTS)
            time.sleep(5)  # short backoff
        except openai.error.OpenAIError as e:
            logger.error(f"[GPT-4 LB] OpenAIError on account {current_account_index}: {str(e)}")
            current_account_index = (current_account_index + 1) % len(OPENAI_ACCOUNTS)
            time.sleep(5)

    raise Exception("[GPT-4 LB] All accounts or retries exhausted.")


def call_openai_embedding_with_loadbalancer(
    input_list,
    model='text-embedding-ada-002',
    max_retries=5
):
    """
    Round-robin approach for openai.Embedding.create() calls,
    cycling among your 3 OpenAI API keys (OPENAI_ACCOUNTS).
    If a RateLimitError or other OpenAIError occurs, we switch
    to the next account, up to max_retries attempts.
    """
    global current_account_index

    for attempt in range(max_retries):
        account = OPENAI_ACCOUNTS[current_account_index]
        openai.api_key = account['api_key']

        logger.debug(f"[Embedding LB] Attempt {attempt+1}/{max_retries} using account index={current_account_index}")

        try:
            response = openai.Embedding.create(
                input=input_list,
                model=model
            )
            return response
        except openai.error.RateLimitError as e:
            logger.warning(f"[Embedding LB] Rate limit on account {current_account_index}. Switching accounts.")
            current_account_index = (current_account_index + 1) % len(OPENAI_ACCOUNTS)
            time.sleep(5)
        except openai.error.OpenAIError as e:
            logger.error(f"[Embedding LB] OpenAIError on account {current_account_index}: {str(e)}")
            current_account_index = (current_account_index + 1) % len(OPENAI_ACCOUNTS)
            time.sleep(5)

    raise Exception("[Embedding LB] All accounts or retries exhausted for embeddings.")
