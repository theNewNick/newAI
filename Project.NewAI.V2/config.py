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
# we simply read multiple API keys from .env.
OPENAI_API_KEY_1 = os.getenv('OPENAI_API_KEY_1', '')
OPENAI_API_KEY_2 = os.getenv('OPENAI_API_KEY_2', '')
OPENAI_API_KEY_3 = os.getenv('OPENAI_API_KEY_3', '')
OPENAI_API_KEY_4 = os.getenv('OPENAI_API_KEY_4', '')
OPENAI_API_KEY_5 = os.getenv('OPENAI_API_KEY_5', '')

# You can still keep a list if needed for any references,
# or simply rely on your new smart_load_balancer.py
OPENAI_ACCOUNTS = [
    {'api_key': OPENAI_API_KEY_1},
#    {'api_key': OPENAI_API_KEY_2},
#    {'api_key': OPENAI_API_KEY_3},
#    {'api_key': OPENAI_API_KEY_4},
#    {'api_key': OPENAI_API_KEY_5},
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
# (Old Round-Robin Code Removed / Disabled)
###############################################

# If you previously used round-robin, you can remove or comment out
# the following items:

# current_account_index = 0

# def call_gpt_4_with_loadbalancer(
#     messages,
#     temperature=0.5,
#     max_tokens=750,
#     max_retries=5
# ):
#     """
#     OLD ROUND-ROBIN function removed. 
#     Now replaced by your 'smart_load_balancer.py' approach.
#     """
#     raise NotImplementedError("Use call_openai_smart from smart_load_balancer.py instead.")

# def call_openai_embedding_with_loadbalancer(
#     input_list,
#     model='text-embedding-ada-002',
#     max_retries=5
# ):
#     """
#     OLD ROUND-ROBIN function removed.
#     Now replaced by your 'smart_load_balancer.py' approach.
#     """
#     raise NotImplementedError("Use a new function in smart_load_balancer.py instead.")
