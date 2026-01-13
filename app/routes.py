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
    # Provide dataset summary and kamus to the front-end for dynamic UI
    from .utils import load_kamus, summarize_dataset
    kamus = load_kamus(current_app.config.get('KAMUS_PATH'))
    summary = summarize_dataset(current_app.config['DATASET_DIR'], kamus)
    return render_template('index.html', kamus=kamus, dataset_summary=summary)

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
        
        # Prefer sura/verse from prediction result (multi-sura support)
        sura = prediction_result.get('sura')
        verse = prediction_result.get('verse')
        if sura is not None and verse is not None:
            db_data = get_verse_info(int(sura), verse)
        else:
            # Fallback to legacy behavior: assume sura 78 if possible
            fallback_verse = prediction_result.get('verse_id')
            db_data = get_verse_info(78, fallback_verse)

        # Attach surah name from kamus.json if available
        from .utils import load_kamus, get_sura_name
        kamus = load_kamus(current_app.config.get('KAMUS_PATH'))
        try:
            sura_key = str(int(sura)).zfill(3) if sura is not None else str(int(db_data.get('suraId', 78))).zfill(3)
        except Exception:
            sura_key = str(db_data.get('suraId', 78))
        db_data['suraName'] = get_sura_name(kamus, sura_key) if kamus else None

        return render_template('result.html', 
                               result=prediction_result, 
                               db_data=db_data,
                               filename=filename,
                               kamus=kamus)
    else:
        flash('Invalid file type')
        return redirect(url_for('main.index'))

@main_bp.route('/training')
def training():
    from .utils import load_kamus, summarize_dataset
    kamus = load_kamus(current_app.config.get('KAMUS_PATH'))
    summary = summarize_dataset(current_app.config['DATASET_DIR'], kamus)
    return render_template('training.html', kamus=kamus, dataset_summary=summary)

@main_bp.route('/api/train', methods=['POST'])
def start_training():
    from .utils import TrainingProgress

    app_config = {
        'DATASET_DIR': current_app.config['DATASET_DIR'],
        'MODEL_PATH': current_app.config['MODEL_PATH'],
        'ENCODER_PATH': current_app.config['ENCODER_PATH'],
        'METADATA_PATH': current_app.config['METADATA_PATH']
    }
    dataset_path = app_config['DATASET_DIR']

    # Prevent concurrent trainings
    if getattr(TrainingProgress, '_is_training', False):
        return jsonify({'status': 'already_running', 'message': 'Training already in progress.'})
    
    def run_training(config, app_obj):
        # pass the app object to allow training task to reload the model into running app
        train_model_task(dataset_path, config, app_obj)

    thread = threading.Thread(target=run_training, args=(app_config, current_app._get_current_object()))
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Training process initiated in background using ' + dataset_path})


@main_bp.route('/api/stop-train', methods=['POST'])
def stop_training():
    from .utils import TrainingProgress
    if not getattr(TrainingProgress, '_is_training', False):
        return jsonify({'status': 'not_running', 'message': 'No active training to stop.'})
    TrainingProgress.request_stop()
    return jsonify({'status': 'stopping', 'message': 'Stop request submitted.'})

@main_bp.route('/api/reload-model', methods=['POST'])
def reload_model():
    try:
        current_app.model_handler.load_model()
        return jsonify({'status': 'reloaded', 'message': 'Model reloaded successfully.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@main_bp.route('/api/clear-model-cache', methods=['POST'])
def clear_model_cache():
    try:
        current_app.model_handler.clear_cache()
        return jsonify({'status': 'cleared', 'message': 'Model cache cleared.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@main_bp.route('/api/train-progress')
def train_progress():
    def generate():
        try:
            while True:
                data = TrainingProgress.get_update()
                try:
                    yield f"data: {json.dumps(data)}\n\n"
                except GeneratorExit:
                    # client disconnected
                    break

                # break if training has finished or stopped for any reason
                if not data['is_training']:
                    break

                time.sleep(1)
        except Exception as e:
            print(f"SSE generator error: {e}")
            TrainingProgress.push_log(f"SSE generator error: {e}")

    from flask import Response
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(generate(), mimetype='text/event-stream', headers=headers)

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


@main_bp.route('/tajwid')
def tajwid():
    """Static page: Panduan Tajwid"""
    from .utils import load_kamus
    kamus = load_kamus(current_app.config.get('KAMUS_PATH'))
    return render_template('tajwid.html', kamus=kamus)


@main_bp.route('/quran')
def quran_list():
    """Page: List of all Surahs"""
    from .utils import load_kamus
    kamus = load_kamus(current_app.config.get('KAMUS_PATH'))
    return render_template('quran_list.html', kamus=kamus)


@main_bp.route('/surah/<int:sura_id>')
def surah_detail(sura_id):
    """Page: Detail of a Surah with all its verses"""
    from .utils import load_kamus
    kamus = load_kamus(current_app.config.get('KAMUS_PATH'))
    
    # Get surah info from kamus
    surah_info = None
    if kamus:
        sura_key = str(sura_id).zfill(3)
        surah_info = kamus.get(sura_key)
    
    conn = get_db_connection()
    verses = []
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            query = "SELECT ayahText, indoText, readText, verseID FROM quran_id WHERE suraId = %s ORDER BY verseID ASC"
            cursor.execute(query, (sura_id,))
            verses = cursor.fetchall()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"DB Error: {e}")
            
    return render_template('surah_detail.html', 
                           surah_id=sura_id, 
                           surah_info=surah_info, 
                           verses=verses,
                           kamus=kamus)


@main_bp.route('/api/dataset-summary')
def api_dataset_summary():
    from .utils import load_kamus, summarize_dataset
    kamus = load_kamus(current_app.config.get('KAMUS_PATH'))
    summary = summarize_dataset(current_app.config['DATASET_DIR'], kamus)
    return jsonify(summary)

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

    # If verse_id not provided, return default structure
    if verse_id is None:
        return default_data

    # Normalize sura_id type where possible
    try:
        sura_query = int(sura_id)
    except Exception:
        sura_query = sura_id

    conn = get_db_connection()
    if not conn:
        return default_data

    try:
        cursor = conn.cursor(dictionary=True)
        # Select only the fields we care about to ensure consistent keys
        query = "SELECT ayahText, indoText, readText, suraId, verseID FROM quran_id WHERE suraId = %s AND verseID = %s"
        cursor.execute(query, (sura_query, verse_id))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            return default_data

        # Normalize and ensure types
        ayahText = row.get('ayahText') if row.get('ayahText') is not None else row.get('ayah_text') if row.get('ayah_text') is not None else 'Data not found'
        indoText = row.get('indoText') if row.get('indoText') is not None else row.get('indo_text') if row.get('indo_text') is not None else '-'
        readText = row.get('readText') if row.get('readText') is not None else row.get('read_text') if row.get('read_text') is not None else '-'

        try:
            sura_ret = int(row.get('suraId')) if row.get('suraId') is not None else int(sura_query)
        except Exception:
            sura_ret = sura_query

        try:
            verse_ret = int(row.get('verseID')) if row.get('verseID') is not None else int(verse_id)
        except Exception:
            verse_ret = verse_id

        normalized = {
            'ayahText': ayahText,
            'indoText': indoText,
            'readText': readText,
            'suraId': sura_ret,
            'verseID': verse_ret
        }
        return normalized
    except Exception as e:
        print(f"DB Error: {e}")
        return default_data
