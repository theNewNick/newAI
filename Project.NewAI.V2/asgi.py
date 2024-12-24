from asgiref.wsgi import WsgiToAsgi
from app import app  # Import your existing Flask app from app.py

# Wrap the Flask WSGI app to make it ASGI-compatible
asgi_app = WsgiToAsgi(app)

# OPTIONAL: If you want to run uvicorn directly (not via Gunicorn), you can uncomment below:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(asgi_app, host="0.0.0.0", port=8000)
