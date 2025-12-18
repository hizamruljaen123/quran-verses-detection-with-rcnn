import os
from flask import Flask
import mysql.connector
import threading
import time

model_handler = None

def get_db_connection():
    from flask import current_app
    try:
        conn = mysql.connector.connect(
            host=current_app.config['DB_HOST'],
            user=current_app.config['DB_USER'],
            password=current_app.config['DB_PASS'],
            database=current_app.config['DB_NAME']
        )
        return conn
    except Exception as e:
        print(f"Database Connection Error: {e}")
        return None

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')
    from config import Config
    Config.init_app(app)

    # Initialize Model Handler and attach to app
    from .utils import ModelHandler
    model_config = {
        'MODEL_PATH': app.config['MODEL_PATH'],
        'MODEL_DIR': app.config.get('MODEL_DIR', os.path.join(app.root_path, '..', 'models')),
        'ENCODER_PATH': app.config['ENCODER_PATH'],
        'METADATA_PATH': app.config['METADATA_PATH']
    }
    app.model_handler = ModelHandler(model_config)

    # Register Blueprint
    from .routes import main_bp
    app.register_blueprint(main_bp)

    # Auto-train if model missing
    if not os.path.exists(app.config['MODEL_PATH']):
        print("⚠️ No model found. Auto-starting training sequence...")
        dataset_dir = app.config['DATASET_DIR']
        
        def auto_train():
            from .utils import train_model_task
            time.sleep(3) # Let server boot
            train_model_task(dataset_dir, model_config)
            
        t = threading.Thread(target=auto_train)
        t.daemon = True
        t.start()

    return app
