import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev_key_very_secret'
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac'}

    DB_HOST = '127.0.0.1'
    DB_USER = 'root'
    DB_PASS = ''
    DB_NAME = 'quran_db'    
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'audio_model.pkl')
    ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    METADATA_PATH = os.path.join(MODEL_DIR, 'metadata.json')
    
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(Config.DATA_DIR, exist_ok=True)
