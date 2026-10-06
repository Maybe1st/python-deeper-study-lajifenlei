from flask import Blueprint, request, session, redirect, url_for, render_template, current_app
import werkzeug.security as ws
from database import get_db

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    username = request.form['username']
    password = request.form['password']
    conn = get_db(current_app.config['DATABASE_PATH'])
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=?', (username,))
    row = c.fetchone()
    if row and ws.check_password_hash(row['password_hash'], password):
        session['user_id'] = row['id']
        session['username'] = row['username']
        session['is_admin'] = bool(row['is_admin'])
        return redirect(url_for('main.index'))
    return render_template('login.html', error='用户名或密码错误')

@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login'))

@auth_bp.route('/register', methods=['GET','POST'])
def register():
    if request.method=='GET':
        return render_template('register.html')
    username = request.form['username']
    password = request.form['password']
    pw_hash = ws.generate_password_hash(password)
    conn = get_db(current_app.config['DATABASE_PATH'])
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username,password_hash) VALUES (?,?)',(username,pw_hash))
        conn.commit()
    except Exception as e:
        return render_template('register.html', error='用户名已存在')
    return redirect(url_for('auth.login'))
