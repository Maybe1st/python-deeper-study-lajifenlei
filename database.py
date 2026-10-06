import sqlite3
import os
from datetime import datetime

def init_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # users
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0
    )
    ''')
    # records
    c.execute('''
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT,
        predicted_class TEXT,
        confidence REAL,
        created_at TEXT
    )
    ''')
    # training records
    c.execute('''
    CREATE TABLE IF NOT EXISTS training_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TEXT,
        finished_at TEXT,
        status TEXT,
        accuracy REAL
    )
    ''')
    conn.commit()
    # create default admin if missing
    c.execute("SELECT COUNT(*) FROM users WHERE username='admin'")
    if c.fetchone()[0]==0:
        import werkzeug.security as ws
        pw = ws.generate_password_hash('admin123')
        c.execute("INSERT INTO users (username,password_hash,is_admin) VALUES (?,?,1)",('admin',pw))
        conn.commit()
    conn.close()


def get_db(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__),'data','app.db')
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn
