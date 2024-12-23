import os

# Load environment variables
# Make sure you have these environment variables set or defaults will be used.
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY')
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'YOUR_FLASK_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'your-bucket-name')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'your-pinecone-api-key')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east1-gcp')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'your-index-name')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', 'your-newsapi-key')

# Application configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'feedback.db')}"
SECRET_KEY = FLASK_SECRET_KEY
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')