import os

# Basic configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'app.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODELS_DIR = os.path.join(BASE_DIR, 'models_weights')
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}
SECRET_KEY = os.environ.get('SECRET_KEY','dev-secret-key')
DEVICE = 'cpu'  # change to 'cuda' if you want and have torch with cuda

CLASS_NAMES = ['glass','paper','cardboard','plastic','metal','trash']
CLASS_CN = {
    'glass':'玻璃', 'paper':'纸张', 'cardboard':'纸板', 'plastic':'塑料', 'metal':'金属', 'trash':'其他垃圾'
}
