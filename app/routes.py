from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify, Response
import os
from werkzeug.utils import secure_filename
from . import get_db_connection
from .utils import train_model_task, TrainingProgress
import time
import threading
import json

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        flash('No file part')
        return redirect(url_for('main.index'))
        
    file = request.files['audio_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('main.index'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        prediction_result = current_app.model_handler.predict(upload_path)
        
        verse_id = prediction_result['verse_id']
        db_data = get_verse_info(78, verse_id)
        
        return render_template('result.html', 
                               result=prediction_result, 
                               db_data=db_data,
                               filename=filename)
    else:
        flash('Invalid file type')
        return redirect(url_for('main.index'))

@main_bp.route('/training')
def training():
    return render_template('training.html')

@main_bp.route('/api/train', methods=['POST'])
def start_training():
    app_config = {
        'DATASET_DIR': current_app.config['DATASET_DIR'],
        'MODEL_PATH': current_app.config['MODEL_PATH'],
        'ENCODER_PATH': current_app.config['ENCODER_PATH'],
        'METADATA_PATH': current_app.config['METADATA_PATH']
    }
    dataset_path = app_config['DATASET_DIR']
    
    def run_training(config):
        train_model_task(dataset_path, config)

    thread = threading.Thread(target=run_training, args=(app_config,))
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Training process initiated in background using ' + dataset_path})

@main_bp.route('/api/train-progress')
def train_progress():
    def generate():
        while True:
            data = TrainingProgress.get_update()
            yield f"data: {json.dumps(data)}\n\n"
            if not data['is_training'] and data['percent'] == 100:
                break
            time.sleep(1)
            
    from flask import Response
    return Response(generate(), mimetype='text/event-stream')

@main_bp.route('/evaluate')
def evaluate():
    metrics_file = os.path.join(current_app.config['MODEL_DIR'], 'metrics.json')
    
    default_metrics = {
        'accuracy': '0.0%',
        'loss': '0.000',
        'precision': '0.00',
        'recall': '0.00',
        'f1': '0.00',
        'last_updated': 'Belum pernah dilatih'
    }
    
    metrics = default_metrics
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"Error reading metrics: {e}")
            
    return render_template('evaluate.html', metrics=metrics)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def get_verse_info(sura_id, verse_id):
    default_data = {
        'ayahText': 'Data not found',
        'indoText': '-',
        'readText': '-',
        'suraId': sura_id,
        'verseID': verse_id
    }
    
    if verse_id is None:
        return default_data

    conn = get_db_connection()
    if not conn:
        return default_data
        
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM quran_id WHERE suraId = %s AND verseID = %s"
        cursor.execute(query, (sura_id, verse_id))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result if result else default_data
    except Exception as e:
        print(f"DB Error: {e}")
        return default_data
