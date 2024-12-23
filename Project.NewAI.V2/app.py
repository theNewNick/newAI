import os
import sys
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import config

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

# Flask app setup
app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = config.SQLALCHEMY_DATABASE_URI
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
