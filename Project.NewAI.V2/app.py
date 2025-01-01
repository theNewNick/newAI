import os
import sys
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import config
from modules.system2.handlers import test_bp

# NEW IMPORTS
import logging
import boto3
from modules.system2.tasks import process_pdf_chunks_task

print("Current Working Directory:", os.getcwd())
print("Python Path:", sys.path)

# Ensure project root is in the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the db object from extensions (no circular imports)
from modules.extensions import db

# Blueprint imports
from modules.system1 import system1_bp
from modules.system2 import system2_bp
from modules.system3 import system3_bp

# NEW IMPORT: bring in your call_local_llama function
from llama_client import call_local_llama

# NEW FUNCTION: Eager embedding of all PDFs in S3
def eager_embed_all_pdfs_on_startup():
    """
    On server startup, list all PDFs in S3 and enqueue Celery tasks
    to embed them. This runs once each time we start the server.
    """
    s3 = boto3.client('s3', region_name=config.AWS_REGION)
    bucket_name = config.S3_BUCKET_NAME

    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' not in response:
            print("No files in the bucket. Skipping eager embedding.")
            return

        for obj in response['Contents']:
            key = obj['Key']
            if key.lower().endswith('.pdf'):
                print(f"Eager embedding PDF: {key}")
                task_result = process_pdf_chunks_task.delay(
                    bucket_name=bucket_name,
                    object_key=key,
                    pinecone_index_name=config.PINECONE_INDEX_NAME
                )
                print(f"Enqueued Celery task: {task_result.id} for {key}")

    except Exception as e:
        print(f"Error during eager embedding: {e}")

# Flask app setup
app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = config.SQLALCHEMY_DATABASE_URI
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Database setup
db.init_app(app)
migrate = Migrate(app, db)

# Blueprint registration
app.register_blueprint(system1_bp, url_prefix='/system1')
app.register_blueprint(system2_bp, url_prefix='/system2')
app.register_blueprint(system3_bp, url_prefix='/system3')

# Routes
@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# NEW ROUTE: ask_llama
@app.route('/ask_llama', methods=['POST'])
def ask_llama():
    user_prompt = request.form.get('prompt', '')
    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    llama_answer = call_local_llama(user_prompt, max_tokens=300)
    return jsonify({"llama_answer": llama_answer})

# Eagerly embed all existing PDFs on startup
eager_embed_all_pdfs_on_startup()

# The test blueprint (just your example):
app.register_blueprint(test_bp, url_prefix="/test")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
