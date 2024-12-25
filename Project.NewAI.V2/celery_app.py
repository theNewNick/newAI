# celery_app.py

import os
import sys
import logging
from dotenv import load_dotenv
from celery import Celery

# 1) Load environment variables (like CELERY_BROKER_URL, CELERY_RESULT_BACKEND)
load_dotenv()

# 2) Ensure the directory containing this file (Project.NewAI.V2) is in sys.path
#    so "modules/" can be imported as a top-level package.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# 3) Configure Celery broker & backend (defaults to Redis if not set)
BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# 4) Create the Celery application
celery = Celery(
    'my_celery_app',
    broker=BROKER_URL,
    backend=RESULT_BACKEND
)

# 5) Basic Celery Configuration
#    Note the 'imports' key that explicitly tells Celery to load your tasks file.
celery.conf.update({
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'UTC',
    'enable_utc': True,
    # Manually import tasks from modules/system2/tasks.py to avoid "No module named 'modules'"
    'imports': ('modules.system2.tasks',),
})

# 6) Optional: Additional logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Celery app initialized with broker=%s, backend=%s", BROKER_URL, RESULT_BACKEND)


"""
--------------------------------------------------------------------------------
GUIDE:

- All your heavy-lifting code (S3 download, PDF chunking, embedding, Pinecone upsert)
  is in modules/system2/tasks.py, in the @shared_task named 'process_pdf_chunks_task'.

- This celery_app.py only configures Celery. It manually imports tasks via:
    'imports': ('modules.system2.tasks',),
  so Celery won't attempt autodiscovery (which triggers 'No module named modules').

- Steps to run Celery:
    1) cd /home/ec2-user/newAI/Project.NewAI.V2    (the directory containing this file)
    2) celery -A celery_app.celery worker --loglevel=INFO

- Celery will now successfully import 'modules.system2.tasks' because:
    - We appended CURRENT_DIR to sys.path (line above).
    - We explicitly declared 'imports': ('modules.system2.tasks',).

No more "ModuleNotFoundError: No module named 'modules'" errors.
--------------------------------------------------------------------------------
"""

