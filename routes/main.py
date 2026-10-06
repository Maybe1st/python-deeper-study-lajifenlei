from flask import Blueprint, render_template
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('dashboard.html')

@main_bp.route('/classify')
def classify_page():
    return render_template('classify.html')

@main_bp.route('/history')
def history_page():
    return render_template('history.html')

@main_bp.route('/users')
def users_page():
    return render_template('users.html')

@main_bp.route('/dataset')
def dataset_page():
    return render_template('dataset.html')

@main_bp.route('/train')
def train_page():
    return render_template('train.html')
