import os
from flask import Flask
from routes.auth import auth_bp
from routes.main import main_bp
from routes.api import api_bp
from ml.classifier import Classifier

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models_weights')
MODEL_PATH = os.path.join(MODEL_DIR, 'trash_classifier.pth')

classifier = None

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.secret_key = app.config.get('SECRET_KEY', 'dev-secret-key')

    # ensure folders
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')

    # load classifier
    global classifier
    classifier = Classifier(model_path=MODEL_PATH, device=app.config.get('DEVICE','cpu'))
    app.classifier = classifier

    # initialize database
    from database import init_db
    init_db(app.config['DATABASE_PATH'])

    return app
