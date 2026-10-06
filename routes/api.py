from flask import Blueprint, request, jsonify, current_app, session
from PIL import Image
from datetime import datetime, timedelta
import os, uuid, threading, subprocess, sys, json
from database import get_db
from config import CLASS_CN, CLASS_NAMES

api_bp = Blueprint('api', __name__)

def _is_logged_in():
    return 'user_id' in session

def _is_admin():
    return bool(session.get('is_admin'))

@api_bp.route('/classify', methods=['POST'])
def classify():
    if not _is_logged_in():
        return jsonify({'code':1,'message':'login required'}),401
    if 'file' not in request.files:
        return jsonify({'code':1,'message':'no file uploaded'}),400
    f = request.files['file']
    if f.filename=='':
        return jsonify({'code':1,'message':'empty filename'}),400
    filename = f"{uuid.uuid4().hex}_{f.filename}"
    save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    f.save(save_path)
    img = Image.open(save_path)
    results = current_app.classifier.predict(img, topk=3)
    top = results[0]
    for r in results:
        r['class_cn'] = CLASS_CN.get(r['class_name'],'')
        r['category'] = '可回收物' if r['class_name']!='trash' else '干垃圾'
    # save record
    try:
        conn = get_db(current_app.config['DATABASE_PATH'])
        c = conn.cursor()
        user_id = session.get('user_id')
        c.execute('INSERT INTO records (user_id,filename,predicted_class,confidence,created_at) VALUES (?,?,?,?,?)',
                  (user_id, filename, top['class_name'], float(top['confidence']), datetime.utcnow().isoformat()))
        conn.commit()
    except Exception as e:
        current_app.logger.error('failed to save record: %s', e)
    return jsonify({'code':0,'data':{
        'predicted_class': top['class_name'],
        'predicted_cn': CLASS_CN.get(top['class_name'],''),
        'category': '可回收物' if top['class_name']!='trash' else '干垃圾',
        'confidence': float(top['confidence']),
        'top3': results
    }})

@api_bp.route('/records', methods=['GET'])
def list_records():
    if not _is_logged_in():
        return jsonify({'code':1,'message':'login required'}),401
    page = int(request.args.get('page',1))
    per_page = int(request.args.get('per_page',10))
    offset = (page-1)*per_page
    conn = get_db(current_app.config['DATABASE_PATH'])
    c = conn.cursor()
    if _is_admin():
        c.execute('SELECT COUNT(*) FROM records')
        total = c.fetchone()[0]
        c.execute('SELECT * FROM records ORDER BY id DESC LIMIT ? OFFSET ?', (per_page, offset))
    else:
        user_id = session.get('user_id')
        c.execute('SELECT COUNT(*) FROM records WHERE user_id=?',(user_id,))
        total = c.fetchone()[0]
        c.execute('SELECT * FROM records WHERE user_id=? ORDER BY id DESC LIMIT ? OFFSET ?', (user_id, per_page, offset))
    rows = c.fetchall()
    items = []
    for r in rows:
        items.append({
            'id': r['id'],
            'user_id': r['user_id'],
            'filename': r['filename'],
            'predicted_class': r['predicted_class'],
            'confidence': r['confidence'],
            'created_at': r['created_at']
        })
    return jsonify({'code':0,'data':{'total': total, 'items': items}})

@api_bp.route('/records/<int:rid>', methods=['DELETE'])
def delete_record(rid):
    if not _is_logged_in():
        return jsonify({'code':1,'message':'login required'}),401
    conn = get_db(current_app.config['DATABASE_PATH'])
    c = conn.cursor()
    c.execute('SELECT * FROM records WHERE id=?',(rid,))
    row = c.fetchone()
    if not row:
        return jsonify({'code':1,'message':'not found'}),404
    if not _is_admin() and row['user_id'] != session.get('user_id'):
        return jsonify({'code':1,'message':'forbidden'}),403
    c.execute('DELETE FROM records WHERE id=?',(rid,))
    conn.commit()
    # remove file if exists
    try:
        fp = os.path.join(current_app.config['UPLOAD_FOLDER'], row['filename'])
        if os.path.exists(fp):
            os.remove(fp)
    except Exception:
        pass
    return jsonify({'code':0,'data':{'deleted': rid}})

@api_bp.route('/dashboard', methods=['GET'])
def dashboard():
    if not _is_logged_in():
        return jsonify({'code':1,'message':'login required'}),401
    conn = get_db(current_app.config['DATABASE_PATH'])
    c = conn.cursor()
    # class distribution
    c.execute('SELECT predicted_class, COUNT(*) as cnt FROM records GROUP BY predicted_class')
    rows = c.fetchall()
    class_distribution = {r['predicted_class']: r['cnt'] for r in rows}
    # category distribution
    total = sum(class_distribution.values())
    category_distribution = {'可回收物': 0, '干垃圾':0}
    for k,v in class_distribution.items():
        if k == 'trash':
            category_distribution['干垃圾'] += v
        else:
            category_distribution['可回收物'] += v
    # confidence bins
    c.execute('SELECT confidence FROM records')
    confidences = [row['confidence'] for row in c.fetchall()]
    bins = {'0-50%':0,'50-70%':0,'70-90%':0,'90-100%':0}
    for cf in confidences:
        p = cf*100
        if p <50: bins['0-50%']+=1
        elif p<70: bins['50-70%']+=1
        elif p<90: bins['70-90%']+=1
        else: bins['90-100%']+=1
    # class average confidence
    class_avg_conf = {}
    for name in CLASS_NAMES:
        c.execute('SELECT AVG(confidence) as avgc, COUNT(*) as cnt FROM records WHERE predicted_class=?',(name,))
        r = c.fetchone()
        class_avg_conf[name] = {'avg_confidence': r['avgc'] or 0.0, 'count': r['cnt']}
    # dataset distribution - attempt to read TrashNet folder
    ds_path = os.path.join('trashnet-master','data','dataset-resized')
    dataset_distribution = {}
    if os.path.exists(ds_path):
        for cname in CLASS_NAMES:
            p = os.path.join(ds_path,cname)
            if os.path.exists(p):
                dataset_distribution[cname] = len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p,f))])
    # daily trend last 14 days
    today = datetime.utcnow().date()
    daily_trend = []
    for i in range(13,-1,-1):
        d = today - timedelta(days=i)
        start = datetime.combine(d, datetime.min.time()).isoformat()
        end = datetime.combine(d, datetime.max.time()).isoformat()
        c.execute('SELECT COUNT(*) FROM records WHERE created_at BETWEEN ? AND ?',(start,end))
        cnt = c.fetchone()[0]
        daily_trend.append({'date': d.isoformat(), 'count': cnt})
    # confusion matrix - not available, return empty
    confusion_matrix = None
    return jsonify({'code':0,'data':{
        'class_distribution': class_distribution,
        'category_distribution': category_distribution,
        'confidence_bins': bins,
        'class_avg_confidence': class_avg_conf,
        'dataset_distribution': dataset_distribution,
        'daily_trend': daily_trend,
        'confusion_matrix': confusion_matrix
    }})

@api_bp.route('/model/status', methods=['GET'])
def model_status():
    exists = os.path.exists(current_app.classifier.model_path) if current_app.classifier.model_path else False
    return jsonify({'code':0,'data':{'exists':exists}})

@api_bp.route('/training/start', methods=['POST'])
def training_start():
    if not _is_logged_in() or not _is_admin():
        return jsonify({'code':1,'message':'admin required'}),403
    req = request.get_json() or {}
    epochs = str(req.get('epochs',10))
    batch_size = str(req.get('batch_size',32))
    lr = str(req.get('learning_rate',0.001))
    out = req.get('out','models_weights/trash_classifier.pth')
    started_at = datetime.utcnow().isoformat()
    conn = get_db(current_app.config['DATABASE_PATH'])
    c = conn.cursor()
    c.execute('INSERT INTO training_records (started_at, status) VALUES (?,?)',(started_at,'running'))
    conn.commit()
    tr_id = c.lastrowid

    def run_and_update():
        cmd = [sys.executable, 'train.py', '--epochs', epochs, '--batch-size', batch_size, '--lr', lr, '--out', out]
        proc = subprocess.Popen(cmd)
        proc.wait()
        finished_at = datetime.utcnow().isoformat()
        status = 'finished' if proc.returncode==0 else 'failed'
        accuracy = None
        meta_path = os.path.splitext(out)[0]+'.json'
        if os.path.exists(meta_path):
            try:
                with open(meta_path,'r',encoding='utf-8') as f:
                    meta = json.load(f)
                    accuracy = float(meta.get('accuracy',0))
            except Exception:
                pass
        c2 = conn.cursor()
        c2.execute('UPDATE training_records SET finished_at=?, status=?, accuracy=? WHERE id=?', (finished_at, status, accuracy, tr_id))
        conn.commit()

    t = threading.Thread(target=run_and_update, daemon=True)
    t.start()
    return jsonify({'code':0,'data':{'status':'started','training_id': tr_id}})

@api_bp.route('/training/records', methods=['GET'])
def training_records():
    if not _is_logged_in() or not _is_admin():
        return jsonify({'code':1,'message':'admin required'}),403
    conn = get_db(current_app.config['DATABASE_PATH'])
    c = conn.cursor()
    c.execute('SELECT * FROM training_records ORDER BY id DESC')
    rows = c.fetchall()
    items = []
    for r in rows:
        items.append({
            'id': r['id'],
            'started_at': r['started_at'],
            'finished_at': r['finished_at'],
            'status': r['status'],
            'accuracy': r['accuracy']
        })
    return jsonify({'code':0,'data': items})

@api_bp.route('/users', methods=['GET'])
def list_users():
    if not _is_logged_in() or not _is_admin():
        return jsonify({'code':1,'message':'admin required'}),403
    conn = get_db(current_app.config['DATABASE_PATH'])
    c = conn.cursor()
    c.execute('SELECT id, username, is_admin FROM users')
    rows = c.fetchall()
    items = [{'id':r['id'],'username':r['username'],'is_admin':bool(r['is_admin'])} for r in rows]
    return jsonify({'code':0,'data':items})

@api_bp.route('/users/<int:uid>', methods=['DELETE'])
def delete_user(uid):
    if not _is_logged_in() or not _is_admin():
        return jsonify({'code':1,'message':'admin required'}),403
    conn = get_db(current_app.config['DATABASE_PATH'])
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE id=?',(uid,))
    row = c.fetchone()
    if not row:
        return jsonify({'code':1,'message':'not found'}),404
    # prevent deleting the last admin
    if row['is_admin']:
        # check if more than one admin exists
        c.execute('SELECT COUNT(*) FROM users WHERE is_admin=1')
        if c.fetchone()[0] <= 1:
            return jsonify({'code':1,'message':'cannot delete last admin'}),400
    c.execute('DELETE FROM users WHERE id=?',(uid,))
    conn.commit()
    return jsonify({'code':0,'data':{'deleted': uid}})
