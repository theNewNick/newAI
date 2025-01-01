import os
import sys
import logging
from dotenv import load_dotenv
from celery import Celery

# 1) Load environment variables (if any), though we'll ignore broker/env fallback.
load_dotenv()

# 2) Ensure the directory containing this file is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# --------------------------------------------------------------------------
# Hard-code the Redis broker & backend (ignore environment variables)
# --------------------------------------------------------------------------
REDIS_BROKER_URL = "redis://127.0.0.1:6380/0"
REDIS_RESULT_BACKEND = "redis://127.0.0.1:6380/0"


# 3) Create the Celery application WITHOUT broker= or backend= arguments
celery = Celery("my_celery_app")

# 4) Forcibly assign broker & backend to Redis in the config
celery.conf.broker_url = REDIS_BROKER_URL
celery.conf.result_backend = REDIS_RESULT_BACKEND

# 5) Basic Celery Configuration
celery.conf.update({
    "task_serializer": "json",
    "accept_content": ["json"],
    "result_serializer": "json",
    "timezone": "UTC",
    "enable_utc": True,
    # Import the tasks file so Celery sees our newly-bound tasks
    "imports": ("modules.system2.tasks",),
})

# 6) Optional: Additional logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(
    "Celery forcibly set to broker=%s, backend=%s",
    celery.conf.broker_url, celery.conf.result_backend
)


"""
--------------------------------------------------------------------------------
GUIDE:

- This file enforces Redis for all Celery operations. 
- No environment variable or default Celery config will override it.
- 'imports': ('modules.system2.tasks',) ensures your tasks are discovered at worker startup.

Run Celery with:
    cd /home/ec2-user/newAI/Project.NewAI.V2
    source ../venv/bin/activate
    celery -A celery_app.celery worker --loglevel=INFO

--------------------------------------------------------------------------------
"""
