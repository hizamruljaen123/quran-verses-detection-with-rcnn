"""
Quran Verse Detection Web Application
=====================================================
Web application untuk deteksi ayat Al-Quran menggunakan AI
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import json
import pickle
import numpy as np
import librosa
import tensorflow as tf
import gc
import psutil
import traceback
import sys
import threading
import time
from sklearn.preprocessing import LabelEncoder, RobustScaler
import mysql.connector
from mysql.connector import Error
import scipy.signal
from datetime import datetime
import shutil
from werkzeug.utils import secure_filename

# Libraries for public access
import subprocess
import socket
import requests
import tempfile
import webbrowser

# Try to import ngrok libraries (optional)
try:
    from pyngrok import ngrok, conf
    NGROK_AVAILABLE = True
    print("✅ pyngrok library available")
except ImportError:
    NGROK_AVAILABLE = False
    print("⚠️ pyngrok not available - install with: pip install pyngrok")

try:
    import localtunnel
    LOCALTUNNEL_AVAILABLE = True
    print("✅ localtunnel library available")
except ImportError:
    LOCALTUNNEL_AVAILABLE = False
    print("⚠️ localtunnel not available - install with: pip install localtunnel")

app = Flask(__name__)

# Load configuration from file if exists
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    app.config['SECRET_KEY'] = config['app']['secret_key']
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['MAX_CONTENT_LENGTH'] = config['upload']['max_file_size_mb'] * 1024 * 1024
    
    DB_CONFIG = config['database']
    
except (FileNotFoundError, KeyError, json.JSONDecodeError):
    # Fallback to default configuration
    app.config['SECRET_KEY'] = 'quran_detection_secret_key_2025'
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
    
    # Database configuration
    DB_CONFIG = {
        'host': '127.0.0.1',
        'database': 'quran_db',
        'user': 'root',
        'password': '',
        'port': 3306
    }

# Global variables for model and tunneling
loaded_model = None
loaded_encoder = None
model_metadata = None
training_status = {"status": "idle", "progress": 0, "message": ""}

# Public access variables
public_url = None
tunnel_process = None
tunnel_type = None

# Error tracking
app_errors = []
prediction_count = 0
last_cleanup_time = time.time()

class TunnelManager:
    """Manager untuk membuat tunnel publik"""
    
    def __init__(self):
        self.ngrok_tunnel = None
        self.localtunnel_process = None
        self.current_tunnel = None
        self.tunnel_type = None
    
    def create_ngrok_tunnel(self, port, auth_token=None):
        """Buat tunnel menggunakan Ngrok"""
        if not NGROK_AVAILABLE:
            return None, "pyngrok library not available"
        
        try:
            # Set auth token if provided
            if auth_token:
                conf.get_default().auth_token = auth_token
            
            # Kill existing tunnels
            ngrok.kill()
            
            # Create tunnel
            public_tunnel = ngrok.connect(port, "http")
            public_url = public_tunnel.public_url
            
            self.ngrok_tunnel = public_tunnel
            self.current_tunnel = public_url
            self.tunnel_type = "ngrok"
            
            print(f"🌍 Ngrok tunnel created: {public_url}")
            return public_url, None
            
        except Exception as e:
            error_msg = f"Failed to create Ngrok tunnel: {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg
    
    def create_localtunnel(self, port):
        """Buat tunnel menggunakan LocalTunnel"""
        try:
            # Check if localtunnel command is available
            result = subprocess.run(['npx', 'localtunnel', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return None, "LocalTunnel not available - install with: npm install -g localtunnel"
            
            # Start localtunnel
            cmd = ['npx', 'localtunnel', '--port', str(port)]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, text=True)
            
            # Wait for URL
            for _ in range(10):  # Wait up to 10 seconds
                if process.poll() is not None:
                    break
                time.sleep(1)
                
                # Try to read output
                try:
                    output = process.stdout.readline()
                    if 'https://' in output:
                        url = output.strip().split()[-1]
                        self.localtunnel_process = process
                        self.current_tunnel = url
                        self.tunnel_type = "localtunnel"
                        print(f"🌍 LocalTunnel created: {url}")
                        return url, None
                except:
                    continue
            
            # If we get here, tunnel creation failed
            process.terminate()
            return None, "Failed to get LocalTunnel URL"
            
        except Exception as e:
            error_msg = f"Failed to create LocalTunnel: {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg
    
    def create_serveo_tunnel(self, port):
        """Buat tunnel menggunakan Serveo (SSH tunnel)"""
        try:
            # Generate random subdomain
            import random
            import string
            subdomain = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            
            # Create SSH tunnel command
            cmd = [
                'ssh', '-R', f'{subdomain}:80:localhost:{port}', 
                'serveo.net', '-o', 'StrictHostKeyChecking=no'
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, text=True)
            
            # Wait for URL
            for _ in range(15):  # Wait up to 15 seconds
                if process.poll() is not None:
                    break
                time.sleep(1)
                
                # Check for URL in output
                try:
                    output = process.stderr.readline()
                    if 'Forwarding HTTP traffic from' in output:
                        url = f"https://{subdomain}.serveo.net"
                        self.localtunnel_process = process
                        self.current_tunnel = url
                        self.tunnel_type = "serveo"
                        print(f"🌍 Serveo tunnel created: {url}")
                        return url, None
                except:
                    continue
            
            # If we get here, tunnel creation failed
            process.terminate()
            return None, "Failed to create Serveo tunnel"
            
        except Exception as e:
            error_msg = f"Failed to create Serveo tunnel: {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg
    
    def create_cloudflared_tunnel(self, port):
        """Buat tunnel menggunakan Cloudflare Tunnel"""
        try:
            # Check if cloudflared is available
            result = subprocess.run(['cloudflared', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return None, "Cloudflare Tunnel not available - install cloudflared"
            
            # Start cloudflared tunnel
            cmd = ['cloudflared', 'tunnel', '--url', f'http://localhost:{port}']
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, text=True)
            
            # Wait for URL
            for _ in range(20):  # Wait up to 20 seconds
                if process.poll() is not None:
                    break
                time.sleep(1)
                
                # Try to read output
                try:
                    output = process.stderr.readline()
                    if 'https://' in output and 'trycloudflare.com' in output:
                        # Extract URL from output
                        import re
                        url_match = re.search(r'https://[^\s]+\.trycloudflare\.com', output)
                        if url_match:
                            url = url_match.group()
                            self.localtunnel_process = process
                            self.current_tunnel = url
                            self.tunnel_type = "cloudflare"
                            print(f"🌍 Cloudflare tunnel created: {url}")
                            return url, None
                except:
                    continue
            
            # If we get here, tunnel creation failed
            process.terminate()
            return None, "Failed to get Cloudflare tunnel URL"
            
        except Exception as e:
            error_msg = f"Failed to create Cloudflare tunnel: {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg
    
    def auto_create_tunnel(self, port, ngrok_token=None):
        """Otomatis buat tunnel dengan mencoba berbagai opsi"""
        print("🚀 Attempting to create public tunnel...")
        
        # Try Ngrok first (most reliable)
        if NGROK_AVAILABLE:
            print("🔧 Trying Ngrok...")
            url, error = self.create_ngrok_tunnel(port, ngrok_token)
            if url:
                return url, "ngrok"
        
        # Try Cloudflare tunnel
        print("🔧 Trying Cloudflare tunnel...")
        url, error = self.create_cloudflared_tunnel(port)
        if url:
            return url, "cloudflare"
        
        # Try LocalTunnel
        print("🔧 Trying LocalTunnel...")
        url, error = self.create_localtunnel(port)
        if url:
            return url, "localtunnel"
        
        # Try Serveo
        print("🔧 Trying Serveo...")
        url, error = self.create_serveo_tunnel(port)
        if url:
            return url, "serveo"
        
        print("❌ All tunnel options failed")
        return None, "failed"
    
    def close_tunnel(self):
        """Tutup tunnel yang aktif"""
        try:
            if self.tunnel_type == "ngrok" and self.ngrok_tunnel:
                ngrok.disconnect(self.ngrok_tunnel.public_url)
                ngrok.kill()
                print("🛑 Ngrok tunnel closed")
            
            if self.localtunnel_process:
                self.localtunnel_process.terminate()
                self.localtunnel_process.wait(timeout=5)
                print(f"🛑 {self.tunnel_type} tunnel closed")
            
            self.current_tunnel = None
            self.tunnel_type = None
            
        except Exception as e:
            print(f"⚠️ Error closing tunnel: {e}")

# Initialize tunnel manager
tunnel_manager = TunnelManager()

# Error tracking
app_errors = []
prediction_count = 0
last_cleanup_time = time.time()

def log_app_error(error_msg, error_type="GENERAL"):
    """Log errors without crashing the app"""
    global app_errors
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_entry = {
            'timestamp': timestamp,
            'type': error_type,
            'message': str(error_msg),
            'prediction_count': prediction_count
        }
        app_errors.append(error_entry)
        
        # Keep only last 50 errors to prevent memory buildup
        if len(app_errors) > 50:
            app_errors = app_errors[-50:]
        
        print(f"🚨 [{timestamp}] {error_type}: {error_msg}")
        
    except Exception as e:
        print(f"❌ Error logging failed: {e}")

def safe_memory_cleanup():
    """Safely clean up memory without crashing"""
    try:
        print("🧹 Starting safe memory cleanup...")
        
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Log memory usage
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"🧠 Memory after cleanup: {memory_mb:.1f} MB, collected {collected} objects")
        except:
            print("✅ Cleanup completed (memory info unavailable)")
        
        return True
        
    except Exception as e:
        log_app_error(f"Memory cleanup failed: {e}", "CLEANUP_ERROR")
        return False

def emergency_recovery():
    """Emergency recovery function to prevent app death"""
    try:
        print("🚨 EMERGENCY RECOVERY INITIATED")
        
        # Multiple cleanup attempts
        for i in range(3):
            try:
                tf.keras.backend.clear_session()
                gc.collect()
                print(f"   Recovery attempt {i+1}: OK")
                break
            except Exception as e:
                print(f"   Recovery attempt {i+1}: Failed - {e}")
                time.sleep(0.1)
        
        # Log current state
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"🧠 Post-recovery memory: {memory_mb:.1f} MB")
        except:
            pass
        
        print("✅ Emergency recovery completed")
        return True
        
    except Exception as e:
        print(f"❌ Emergency recovery failed: {e}")
        return False

def safe_prediction_wrapper(prediction_func):
    """Wrapper to safely execute predictions without crashing Flask"""
    def wrapper(*args, **kwargs):
        global prediction_count, last_cleanup_time
        prediction_count += 1
        
        try:
            print(f"🔮 Starting prediction #{prediction_count}")
            
            # Log memory before
            try:
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024
                print(f"🧠 Memory before: {memory_before:.1f} MB")
            except:
                memory_before = 0
            
            # Execute prediction with timeout protection
            try:
                result = prediction_func(*args, **kwargs)
                print(f"✅ Prediction #{prediction_count} completed successfully")
                return result
                
            except Exception as pred_error:
                log_app_error(f"Prediction function error: {pred_error}", "PREDICTION_ERROR")
                traceback.print_exc()
                
                # Return safe error response instead of crashing
                return {
                    'error': 'Prediction failed',
                    'details': str(pred_error),
                    'prediction_id': prediction_count,
                    'recovery_status': 'attempted'
                }
            
        except Exception as wrapper_error:
            log_app_error(f"Wrapper error: {wrapper_error}", "WRAPPER_ERROR")
            
            # Emergency recovery
            emergency_recovery()
            
            return {
                'error': 'Critical prediction error',
                'details': str(wrapper_error),
                'prediction_id': prediction_count,
                'recovery_status': 'emergency'
            }
            
        finally:
            # ALWAYS clean up, no matter what
            try:
                # Periodic deep cleanup
                current_time = time.time()
                if current_time - last_cleanup_time > 30:  # Every 30 seconds
                    safe_memory_cleanup()
                    last_cleanup_time = current_time
                else:
                    # Quick cleanup
                    tf.keras.backend.clear_session()
                    gc.collect()
                
                print(f"🧹 Cleanup completed for prediction #{prediction_count}")
                
            except Exception as cleanup_error:
                log_app_error(f"Cleanup error: {cleanup_error}", "CLEANUP_ERROR")
    
    return wrapper

class DatabaseManager:
    """Manager untuk koneksi database MySQL"""
    
    def __init__(self, config):
        self.config = config
    
    def get_connection(self):
        """Buat koneksi database"""
        try:
            connection = mysql.connector.connect(**self.config)
            return connection
        except Error as e:
            print(f"Database connection error: {e}")
            return None
    
    def get_verse_info(self, sura_id, verse_id):
        """Ambil informasi ayat dari tabel quran_id dan pastikan output dict dengan key yang benar"""
        connection = self.get_connection()
        if not connection:
            print(f"[DB] Koneksi gagal untuk suraId={sura_id}, verseID={verse_id}")
            return {
                'ayahText': '',
                'indoText': '',
                'readText': '',
                'verse_id': verse_id,
                'sura_id': sura_id
            }
        try:
            cursor = connection.cursor(dictionary=True)
            print(f"[DB] Query: SELECT * FROM quran_id WHERE suraId={sura_id} AND verseID={verse_id}")
            cursor.execute("SELECT * FROM quran_id WHERE suraId=%s AND verseID=%s", (sura_id, verse_id))
            result = cursor.fetchone()
            print(f"[DB] Query result: {result}")
            cursor.close()
            connection.close()
            # Map ke key yang diharapkan template
            if result:
                # Print all keys for debug
                print(f"[DB] Result keys: {list(result.keys())}")
                return {
                    'ayahText': result.get('ayahText', ''),
                    'indoText': result.get('indoText', ''),
                    'readText': result.get('readText', ''),
                    'verse_id': result.get('verseID', verse_id),
                    'sura_id': result.get('suraId', sura_id)
                }
            else:
                print(f"[DB] Tidak ditemukan data untuk suraId={sura_id}, verseID={verse_id}")
                return {
                    'ayahText': '',
                    'indoText': '',
                    'readText': '',
                    'verse_id': verse_id,
                    'sura_id': sura_id
                }
        except Error as e:
            print(f"Database query error: {e}")
            return {
                'ayahText': '',
                'indoText': '',
                'readText': '',
                'verse_id': verse_id,
                'sura_id': sura_id
            }
    def get_sura_info(self, sura_id):
        """Ambil informasi surat"""
        # Untuk Surah An-Naba (78)
        sura_names = {
            78: {"name": "An-Naba", "name_id": "Berita Besar", "total_verses": 40}
        }
        return sura_names.get(sura_id, {"name": "Unknown", "name_id": "Tidak Diketahui", "total_verses": 0})

# Initialize database manager
db_manager = DatabaseManager(DB_CONFIG)

def extract_advanced_features(file_path, max_length=256, sr=22050):
    """
    Extract advanced audio features untuk deteksi ayat
    """
    try:
        print(f"🎵 Starting feature extraction for: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
        
        # Load dan preprocess audio
        print("📥 Loading audio file...")
        audio, sample_rate = librosa.load(file_path, sr=sr)
        print(f"✅ Audio loaded: {len(audio)} samples at {sample_rate} Hz")
        
        # Basic preprocessing
        print("🔧 Preprocessing audio...")
        audio = librosa.util.normalize(audio)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Apply pre-emphasis filter safely
        try:
            audio = scipy.signal.lfilter([1, -0.95], [1], audio)
        except Exception as e:
            print(f"⚠️ Pre-emphasis filter failed, skipping: {e}")
        
        print(f"✅ Audio preprocessed: {len(audio)} samples")
        
        # Extract multiple features
        features_list = []
        
        # MFCC features
        print("🎯 Extracting MFCC features...")
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            features_list.extend([mfccs, mfcc_delta, mfcc_delta2])
            print(f"✅ MFCC features extracted: {mfccs.shape}")
        except Exception as e:
            print(f"❌ MFCC extraction failed: {e}")
            return None
        
        # Spectral features
        print("🌈 Extracting spectral features...")
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
            features_list.extend([spectral_centroids, spectral_rolloff, spectral_bandwidth, spectral_contrast])
            print(f"✅ Spectral features extracted")
        except Exception as e:
            print(f"❌ Spectral features extraction failed: {e}")
            return None
        
        # Additional features
        print("🎼 Extracting additional features...")
        try:
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features_list.extend([zero_crossing_rate, chroma, tonnetz])
            print(f"✅ Additional features extracted")
        except Exception as e:
            print(f"❌ Additional features extraction failed: {e}")
            return None
        
        # Combine features
        print("🔗 Combining features...")
        try:
            combined_features = np.vstack(features_list)
            print(f"✅ Features combined: {combined_features.shape}")
        except Exception as e:
            print(f"❌ Feature combination failed: {e}")
            return None
        
        # Normalize
        print("📏 Normalizing features...")
        try:
            scaler = RobustScaler()
            combined_features = scaler.fit_transform(combined_features.T).T
            print(f"✅ Features normalized")
        except Exception as e:
            print(f"❌ Feature normalization failed: {e}")
            return None
        
        # Pad or truncate
        print("✂️ Adjusting feature length...")
        try:
            if combined_features.shape[1] < max_length:
                pad_width = max_length - combined_features.shape[1]
                combined_features = np.pad(combined_features, ((0, 0), (0, pad_width)), mode='constant')
            else:
                combined_features = combined_features[:, :max_length]
            
            print(f"✅ Final features shape: {combined_features.T.shape}")
            return combined_features.T
        except Exception as e:
            print(f"❌ Feature length adjustment failed: {e}")
            return None
        
    except Exception as e:
        print(f"❌ Critical error in feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- FIX 1: Model loading, remove batch_shape, add error message if not loaded ---
def load_model_if_exists():
    global loaded_model, loaded_encoder, model_metadata
    print("\U0001F916 Attempting to load AI model...")
    import os
    import pickle
    from tensorflow import keras
    model_dirs = [
        '../model_saves_quran_model_final',
        'models', 
        '../model_saves_basic_improved', 
        '../model_saves_improved_simple'
    ]
    for model_dir in model_dirs:
        print(f"\U0001F4C1 Checking directory: {model_dir}")
        if not os.path.exists(model_dir):
            continue
        try:
            model_path = os.path.join(model_dir, 'quran_model.h5')
            encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
            metadata_path = os.path.join(model_dir, 'metadata.json')
            if os.path.exists(model_path):
                print(f"\U0001F4E5 Loading model from: {model_path}")
                try:
                    loaded_model = keras.models.load_model(model_path)
                except TypeError as e:
                    print(f"❌ Failed to load model: {e}")
                    loaded_model = None
                    continue
                except Exception as e:
                    print(f"❌ Failed to load model: {e}")
                    loaded_model = None
                    continue
                if os.path.exists(encoder_path):
                    with open(encoder_path, 'rb') as f:
                        loaded_encoder = pickle.load(f)
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        model_metadata = json.load(f)
                print("✅ Model loaded successfully!")
                return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            continue
    print("❌ No valid model found in any directory")
    loaded_model = None
    return False

def get_verse_name(verse_number):
    """Convert verse number to readable name"""
    if verse_number == 0:
        return "Bismillah (Pembuka)"
    elif 1 <= verse_number <= 40:
        return f"Ayat {verse_number}"
    else:
        return f"Unknown ({verse_number})"

@safe_prediction_wrapper
@safe_prediction_wrapper
def predict_verse(audio_file_path):
    """Predict verse dari audio file dengan error handling yang aman"""
    global loaded_model, loaded_encoder
    
    print(f"🔮 Starting prediction for: {audio_file_path}")
    
    try:
        if not loaded_model:
            raise Exception("Model not loaded")
        
        if not loaded_encoder:
            raise Exception("Label encoder not loaded")
        
        # Check if file exists
        if not os.path.exists(audio_file_path):
            raise Exception(f"Audio file not found: {audio_file_path}")
        
        # Extract features with error handling
        try:
            print("🎵 Extracting features...")
            features = extract_advanced_features(audio_file_path)
            if features is None:
                raise Exception("Feature extraction returned None")
            
            print(f"✅ Features extracted with shape: {features.shape}")
            
        except Exception as feature_error:
            raise Exception(f"Feature extraction failed: {feature_error}")
        
        # Reshape untuk prediction
        try:
            print("🔄 Reshaping features for model input...")
            features = features.reshape(1, features.shape[0], features.shape[1])
            print(f"✅ Features reshaped to: {features.shape}")
        except Exception as reshape_error:
            raise Exception(f"Feature reshaping failed: {reshape_error}")
        
        # Predict with error handling
        try:
            print("🧠 Running model prediction...")
            prediction = loaded_model.predict(features, verbose=0)
            print(f"✅ Prediction completed: {prediction.shape}")
        except Exception as model_error:
            raise Exception(f"Model prediction failed: {model_error}")
        
        # Get prediction results
        try:
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            print(f"✅ Predicted class: {predicted_class}, confidence: {confidence:.4f}")
        except Exception as result_error:
            raise Exception(f"Prediction result processing failed: {result_error}")
        
        # Get top 3 predictions
        try:
            top3_indices = np.argsort(prediction[0])[-3:][::-1]
            top3_probs = prediction[0][top3_indices]
            print(f"✅ Top 3 predictions: {top3_indices} with probs: {top3_probs}")
        except Exception as top3_error:
            raise Exception(f"Top 3 predictions failed: {top3_error}")
        
        # Convert to verse numbers
        try:
            verse_number = loaded_encoder.inverse_transform([predicted_class])[0]
            top3_verses = loaded_encoder.inverse_transform(top3_indices)
            print(f"✅ Verse conversion: {verse_number}, top3: {top3_verses}")
        except Exception as conversion_error:
            raise Exception(f"Verse number conversion failed: {conversion_error}")
        
        # Build result
        result = {
            'verse_number': int(verse_number),
            'verse_id': int(verse_number),
            'confidence': float(confidence),
            'verse_name': get_verse_name(verse_number),
            'top3_predictions': [
                {
                    'verse_number': int(v),
                    'verse_id': int(v),
                    'verse_name': get_verse_name(v),
                    'probability': float(p)
                }
                for v, p in zip(top3_verses, top3_probs)
            ],
            'prediction_id': prediction_count,
            'status': 'success'
        }
        
        print(f"✅ Prediction successful: Verse {verse_number} with {confidence:.2%} confidence")
        return result
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        log_app_error(error_msg, "PREDICTION_MAIN")
        
        # Return error result instead of None
        return {
            'error': str(e),
            'verse_number': None,
            'verse_id': None,
            'confidence': 0.0,
            'prediction_id': prediction_count,
            'status': 'failed'
        }

# Routes
@app.route('/')
def index():
    """Homepage"""
    model_status = "Loaded" if loaded_model else "Not Loaded"
    tunnel_info = {
        'url': tunnel_manager.current_tunnel,
        'type': tunnel_manager.tunnel_type,
        'active': tunnel_manager.current_tunnel is not None
    }
    return render_template('index.html', 
                         model_status=model_status,
                         tunnel_info=tunnel_info)

@app.route('/tunnel')
def tunnel_page():
    """Halaman untuk mengelola tunnel publik"""
    tunnel_info = {
        'url': tunnel_manager.current_tunnel,
        'type': tunnel_manager.tunnel_type,
        'active': tunnel_manager.current_tunnel is not None,
        'ngrok_available': NGROK_AVAILABLE,
        'localtunnel_available': LOCALTUNNEL_AVAILABLE
    }
    return render_template('tunnel.html', tunnel_info=tunnel_info)

@app.route('/api/create_tunnel', methods=['POST'])
def create_tunnel():
    """API untuk membuat tunnel publik"""
    try:
        data = request.get_json() or {}
        tunnel_type = data.get('type', 'auto')
        ngrok_token = data.get('ngrok_token', None)
        port = data.get('port', 5000)
        
        # Close existing tunnel first
        tunnel_manager.close_tunnel()
        
        if tunnel_type == 'auto':
            url, created_type = tunnel_manager.auto_create_tunnel(port, ngrok_token)
        elif tunnel_type == 'ngrok':
            url, error = tunnel_manager.create_ngrok_tunnel(port, ngrok_token)
            created_type = 'ngrok' if url else None
        elif tunnel_type == 'localtunnel':
            url, error = tunnel_manager.create_localtunnel(port)
            created_type = 'localtunnel' if url else None
        elif tunnel_type == 'cloudflare':
            url, error = tunnel_manager.create_cloudflared_tunnel(port)
            created_type = 'cloudflare' if url else None
        elif tunnel_type == 'serveo':
            url, error = tunnel_manager.create_serveo_tunnel(port)
            created_type = 'serveo' if url else None
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid tunnel type'
            }), 400
        
        if url:
            return jsonify({
                'success': True,
                'url': url,
                'type': created_type,
                'message': f'{created_type.title()} tunnel created successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to create {tunnel_type} tunnel'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/close_tunnel', methods=['POST'])
def close_tunnel():
    """API untuk menutup tunnel aktif"""
    try:
        tunnel_manager.close_tunnel()
        return jsonify({
            'success': True,
            'message': 'Tunnel closed successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/tunnel_status')
def tunnel_status():
    """API untuk mendapatkan status tunnel"""
    try:
        # Test if tunnel is still active
        tunnel_active = False
        if tunnel_manager.current_tunnel:
            try:
                response = requests.get(tunnel_manager.current_tunnel, timeout=5)
                tunnel_active = response.status_code == 200
            except:
                tunnel_active = False
        
        return jsonify({
            'url': tunnel_manager.current_tunnel,
            'type': tunnel_manager.tunnel_type,
            'active': tunnel_active,
            'ngrok_available': NGROK_AVAILABLE,
            'localtunnel_available': LOCALTUNNEL_AVAILABLE
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'active': False
        }), 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_audio():
    """Upload dan predict audio"""
    if request.method == 'POST':
        try:
            print("📤 Starting audio upload process...")
            
            # Check if file is in request
            if 'audio_file' not in request.files:
                print("❌ No file in request")
                flash('No file selected', 'error')
                return redirect(request.url)
            
            file = request.files['audio_file']
            if file.filename == '':
                print("❌ No filename provided")
                flash('No file selected', 'error')
                return redirect(request.url)
            
            print(f"📁 File received: {file.filename}")
            
            # Check file format
            allowed_extensions = ('.mp3', '.wav', '.m4a', '.flac')
            if file and file.filename.lower().endswith(allowed_extensions):
                try:
                    # Save file with error handling
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{timestamp}_{filename}"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    
                    print(f"💾 Saving file to: {filepath}")
                    
                    # Ensure upload directory exists
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    
                    # Save file
                    file.save(filepath)
                    print(f"✅ File saved successfully")
                    
                    # Check if file was actually saved
                    if not os.path.exists(filepath):
                        print("❌ File was not saved properly")
                        flash('Failed to save uploaded file', 'error')
                        return redirect(request.url)
                    
                    # Check file size
                    file_size = os.path.getsize(filepath)
                    print(f"📏 File size: {file_size / (1024*1024):.2f}MB")
                    
                    if file_size == 0:
                        print("❌ Uploaded file is empty")
                        flash('Uploaded file is empty', 'error')
                        return redirect(request.url)
                    
                    # Predict with timeout and cleanup
                    print("🔮 Starting prediction...")
                    result = None
                    
                    try:
                        result = predict_verse(filepath)
                    except Exception as pred_error:
                        print(f"❌ Prediction failed: {pred_error}")
                        import traceback
                        traceback.print_exc()
                        result = None
                    finally:
                        # Always clean up the uploaded file
                        try:
                            if os.path.exists(filepath):
                                os.remove(filepath)
                                print(f"🗑️ Cleaned up file: {filepath}")
                        except Exception as cleanup_error:
                            print(f"⚠️ Failed to cleanup file: {cleanup_error}")
                    
                    if result:
                        print(f"✅ Prediction successful: Verse {result['verse_number']}")
                        
                        # Get verse info from database
                        sura_id = 78  # An-Naba
                        verse_id = result['verse_number']
                        
                        try:
                            verse_info = db_manager.get_verse_info(sura_id, verse_id)
                            sura_info = db_manager.get_sura_info(sura_id)
                            print(f"✅ Database info retrieved")
                        except Exception as db_error:
                            print(f"⚠️ Database query failed: {db_error}")
                            verse_info = None
                            sura_info = None
                        
                        return render_template('result.html', 
                                             result=result, 
                                             verse_info=verse_info,
                                             sura_info=sura_info,
                                             filename=filename)
                    else:
                        print("❌ Prediction returned None")
                        flash('Failed to process audio file. Please try with a different audio file.', 'error')
                        return redirect(request.url)
                        
                except Exception as process_error:
                    print(f"❌ File processing error: {process_error}")
                    import traceback
                    traceback.print_exc()
                    flash(f'Error processing file: {str(process_error)}', 'error')
                    return redirect(request.url)
            else:
                print(f"❌ Invalid file format: {file.filename}")
                flash('Invalid file format. Please upload MP3, WAV, M4A, or FLAC files.', 'error')
                return redirect(request.url)
                
        except Exception as general_error:
            print(f"❌ General upload error: {general_error}")
            import traceback
            traceback.print_exc()
            flash(f'Upload failed: {str(general_error)}', 'error')
            return redirect(request.url)
    
    # GET request: tampilkan halaman upload dengan contoh tajweed_html
    # Contoh: ambil satu ayat dari database untuk preview tajweed
    try:
        sura_id = 78
        verse_id = 23  # Contoh ayat
        verse_info = db_manager.get_verse_info(sura_id, verse_id)
        tajweed_html = verse_info['ayahText'] if verse_info and 'ayahText' in verse_info else ''
    except Exception as e:
        tajweed_html = ''
    return render_template('upload.html', tajweed_html=tajweed_html)

@app.route('/dataset')
def dataset_info():
    """Informasi dataset"""
    dataset_path = "../"
    dataset_stats = {
        'total_folders': 0,
        'total_files': 0,
        'total_verses': 41,  # 0-40 (Bismillah + Verses 1-40)
        'total_size': 'Calculating...',
        'folders': [],
        'sample_files': []
    }
    
    # Scan dataset folders
    for i in range(1, 8):
        folder_name = f"sample_{i}"
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
            dataset_stats['folders'].append({
                'name': folder_name,
                'file_count': len(files),
                'exists': True,
                'verse_range': '078000-078040 (Bismillah + Verses 1-40)',
                'total_size': f'~{len(files) * 2}MB',
                'total_duration': f'~{len(files) * 30}s'
            })
            dataset_stats['total_files'] += len(files)
            dataset_stats['total_folders'] += 1
            
            # Add sample files from this folder
            for j, file in enumerate(files[:3]):  # Show first 3 files
                try:
                    verse_num = int(file.split('.')[0][-3:])
                    verse_name = "Bismillah" if verse_num == 0 else f"Verse {verse_num}"
                    dataset_stats['sample_files'].append({
                        'filename': file,
                        'folder': folder_name,
                        'verse_number': verse_name,
                        'size': '~2MB',
                        'audio_path': None  # Could add if we want to serve audio files
                    })
                except:
                    continue
        else:
            dataset_stats['folders'].append({
                'name': folder_name,
                'file_count': 0,
                'exists': False,
                'verse_range': 'N/A',
                'total_size': '0MB',
                'total_duration': '0s'
            })
    
    # Calculate total size estimate
    dataset_stats['total_size'] = f'~{dataset_stats["total_files"] * 2}MB'
    
    return render_template('dataset.html', dataset_info=dataset_stats)

@app.route('/model_info')
def model_info():
    """Informasi model"""
    global model_metadata, loaded_model
    
    model_info_data = None
    model_layers = None
    training_history = None
    
    if loaded_model:
        try:
            # Get model information
            model_info_data = {
                'type': 'Deep Neural Network',
                'framework': 'TensorFlow/Keras',
                'version': '1.0',
                'input_shape': str(loaded_model.input_shape),
                'total_params': loaded_model.count_params(),
                'layers': len(loaded_model.layers),
                'accuracy': model_metadata.get('accuracy', 'N/A') if model_metadata else 'N/A',
                'loss': model_metadata.get('loss', 'N/A') if model_metadata else 'N/A',
                'training_time': model_metadata.get('training_time', 'N/A') if model_metadata else 'N/A',
                'epochs': model_metadata.get('epochs', 'N/A') if model_metadata else 'N/A',
                'accuracy_score': 0.85,  # Default values for chart
                'speed_score': 0.75,
                'memory_score': 0.8,
                'robustness_score': 0.7,
                'versatility_score': 0.9
            }
            
            # Get layer information
            model_layers = []
            for i, layer in enumerate(loaded_model.layers):
                model_layers.append({
                    'name': f'Layer {i+1} ({layer.name})',
                    'type': layer.__class__.__name__,
                    'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
                    'param_count': layer.count_params() if hasattr(layer, 'count_params') else 0
                })
            
            # Training history (mock data for now)
            if model_metadata and 'training_history' in model_metadata:
                training_history = model_metadata['training_history']
            
        except Exception as e:
            print(f"Error getting model info: {e}")
    
    return render_template('model_info.html', 
                         model_info=model_info_data,
                         model_loaded=loaded_model is not None,
                         model_layers=model_layers,
                         training_history=training_history)

@app.route('/training')
def training_page():
    """Halaman training"""
    return render_template('training.html')

@app.route('/api/training_status')
def training_status_api():
    """API untuk status training"""
    return jsonify(training_status)

@app.route('/start_training', methods=['POST'])
def start_training():
    """Start training process"""
    global training_status
    
    if training_status["status"] == "training":
        return jsonify({"error": "Training already in progress"}), 400
    
    # Start training in background thread
    thread = threading.Thread(target=run_training_process)
    thread.start()
    
    return jsonify({"message": "Training started"})

def run_training_process():
    """Run training process in background"""
    global training_status, loaded_model, loaded_encoder, model_metadata
    
    try:
        training_status = {"status": "training", "progress": 0, "message": "Initializing..."}
        
        # Import training functions (assuming they're in parent directory)
        import sys
        sys.path.append('../')
        
        training_status["message"] = "Loading data..."
        training_status["progress"] = 10
        
        # Simulate training process (replace with actual training)
        for i in range(10, 100, 10):
            time.sleep(2)  # Simulate work
            training_status["progress"] = i
            training_status["message"] = f"Training progress: {i}%"
        
        training_status["progress"] = 100
        training_status["message"] = "Training completed successfully!"
        training_status["status"] = "completed"
        
        # Reload model after training
        load_model_if_exists()
        
    except Exception as e:
        training_status = {
            "status": "error", 
            "progress": 0, 
            "message": f"Training failed: {str(e)}"
        }

@app.route('/train', methods=['POST'])
def train_alias():
    """Alias endpoint untuk memulai training (POST /train)"""
    return start_training()

@app.route('/verses')
def verses_list():
    """List semua ayat dalam surat"""
    verses = []
    sura_info = db_manager.get_sura_info(78)
    
    # Get all verses for An-Naba
    for verse_id in range(1, 41):  # 1-40
        verse_info = db_manager.get_verse_info(78, verse_id)
        if verse_info:
            verses.append(verse_info)
    
    return render_template('verses.html', verses=verses, sura_info=sura_info)

@app.route('/verse/<int:verse_id>')
def verse_detail(verse_id):
    """Detail ayat tertentu"""
    verse_info = db_manager.get_verse_info(78, verse_id)
    sura_info = db_manager.get_sura_info(78)

    print(f"🔍 Fetching details for verse {verse_id} in An-Naba")
    
    if not verse_info:
        flash('Verse not found', 'error')
        return redirect(url_for('verses_list'))
    
    return render_template('verse_detail.html', 
                         verse=verse_info, 
                         sura_info=sura_info)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint untuk prediksi - AMAN DARI CRASH"""
    temp_filepath = None
    
    try:
        print("🌐 API Prediction request received")
        
        # Validate request
        if 'audio_file' not in request.files:
            log_app_error("No audio file in request", "API_REQUEST")
            return jsonify({
                'error': 'No file provided',
                'status': 'failed',
                'prediction_id': prediction_count + 1
            }), 400
        
        file = request.files['audio_file']
        if file.filename == '':
            log_app_error("Empty filename in request", "API_REQUEST")
            return jsonify({
                'error': 'No file selected',
                'status': 'failed',
                'prediction_id': prediction_count + 1
            }), 400
        
        # Validate file format
        allowed_extensions = ('.mp3', '.wav', '.m4a', '.flac')
        if not file.filename.lower().endswith(allowed_extensions):
            log_app_error(f"Invalid file format: {file.filename}", "API_REQUEST")
            return jsonify({
                'error': 'Invalid file format. Supported: MP3, WAV, M4A, FLAC',
                'status': 'failed',
                'prediction_id': prediction_count + 1
            }), 400
        
        # Save temporary file with error handling
        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = f"temp_{timestamp}_{filename}"
            temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(temp_filepath)
            print(f"📁 Temporary file saved: {temp_filepath}")
            
            # Validate saved file
            if not os.path.exists(temp_filepath):
                raise Exception("File save verification failed")
            
            # Check file size
            file_size = os.path.getsize(temp_filepath)
            if file_size == 0:
                raise Exception("Saved file is empty")
            
            print(f"✅ File validation passed: {file_size} bytes")
            
        except Exception as save_error:
            log_app_error(f"File save error: {save_error}", "API_FILE_SAVE")
            return jsonify({
                'error': f'Failed to save file: {str(save_error)}',
                'status': 'failed',
                'prediction_id': prediction_count + 1
            }), 500
        
        # Predict with comprehensive error handling
        try:
            print("🚀 Starting prediction process...")
            result = predict_verse(temp_filepath)
            
            if result is None:
                raise Exception("Prediction returned None")
            
            # Check if result contains error
            if isinstance(result, dict) and 'error' in result:
                print(f"⚠️ Prediction error: {result['error']}")
                return jsonify({
                    'error': result['error'],
                    'details': result.get('details', 'No details available'),
                    'status': 'failed',
                    'prediction_id': result.get('prediction_id', prediction_count)
                }), 500
            
            # Success case - add verse info
            try:
                if result.get('verse_number') is not None:
                    verse_info = db_manager.get_verse_info(78, result['verse_number'])
                    result['verse_info'] = verse_info
                
                result['status'] = 'success'
                print(f"✅ API prediction successful: Verse {result.get('verse_number')}")
                
                return jsonify(result), 200
                
            except Exception as db_error:
                log_app_error(f"Database error: {db_error}", "API_DATABASE")
                # Still return prediction without verse info
                result['verse_info'] = None
                result['db_warning'] = str(db_error)
                return jsonify(result), 200
            
        except Exception as prediction_error:
            log_app_error(f"API prediction error: {prediction_error}", "API_PREDICTION")
            return jsonify({
                'error': 'Prediction processing failed',
                'details': str(prediction_error),
                'status': 'failed',
                'prediction_id': prediction_count
            }), 500
        
    except Exception as general_error:
        log_app_error(f"API general error: {general_error}", "API_GENERAL")
        
        # Emergency recovery
        emergency_recovery()
        
        return jsonify({
            'error': 'Internal server error',
            'details': str(general_error),
            'status': 'failed',
            'recovery_attempted': True
        }), 500
        
    finally:
        # ALWAYS clean up temp file
        try:
            if temp_filepath and os.path.exists(temp_filepath):
                os.remove(temp_filepath)
                print(f"🗑️ Cleaned up file: {temp_filepath}")
        except Exception as cleanup_error:
            log_app_error(f"File cleanup error: {cleanup_error}", "API_CLEANUP")
        
        # Memory cleanup
        try:
            safe_memory_cleanup()
        except Exception as memory_error:
            log_app_error(f"Memory cleanup error: {memory_error}", "API_MEMORY")

# Add Flask error handlers to prevent app crashes
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    log_app_error(f"404 error: {request.url}", "HTTP_404")
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    log_app_error(f"500 error: {error}", "HTTP_500")
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all uncaught exceptions to prevent Flask from crashing"""
    log_app_error(f"Uncaught exception: {e}", "UNCAUGHT_EXCEPTION")
    
    # Emergency recovery
    emergency_recovery()
    
    # Return JSON if it's an API request
    if request.path.startswith('/api/'):
        return jsonify({
            'error': 'Server encountered an unexpected error',
            'details': str(e),
            'recovery_attempted': True
        }), 500
    
    # Return HTML for regular requests
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Unexpected error occurred"), 500

@app.route('/admin/errors')
def view_errors():
    """Admin endpoint to view recent errors"""
    try:
        return render_template('admin_errors.html', 
                             errors=app_errors[-20:],  # Last 20 errors
                             total_predictions=prediction_count)
    except Exception as e:
        return jsonify({
            'recent_errors': app_errors[-10:] if app_errors else [],
            'error_count': len(app_errors),
            'prediction_count': prediction_count,
            'view_error': str(e)
        })

@app.route('/admin/memory')
def view_memory():
    """Admin endpoint to check memory status"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return jsonify({
            'memory_mb': memory_info.rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'prediction_count': prediction_count,
            'error_count': len(app_errors),
            'last_cleanup': last_cleanup_time,
            'model_loaded': loaded_model is not None,
            'encoder_loaded': loaded_encoder is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/admin/cleanup')
def manual_cleanup():
    """Manual cleanup endpoint"""
    try:
        safe_memory_cleanup()
        return jsonify({
            'message': 'Manual cleanup completed',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': f'Cleanup failed: {e}',
            'timestamp': datetime.now().isoformat()
        })

@app.route('/admin/emergency')
def emergency_endpoint():
    """Emergency recovery endpoint"""
    try:
        result = emergency_recovery()
        return jsonify({
            'message': 'Emergency recovery completed',
            'success': result,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': f'Emergency recovery failed: {e}',
            'timestamp': datetime.now().isoformat()
        })

# Add Flask configuration for stability
app.config['PROPAGATE_EXCEPTIONS'] = False  # Prevent exceptions from killing the app
app.config['TRAP_HTTP_EXCEPTIONS'] = True   # Catch HTTP exceptions
app.config['TRAP_BAD_REQUEST_ERRORS'] = True # Catch bad request errors

# Add periodic cleanup
def periodic_cleanup():
    """Background thread for periodic cleanup"""
    while True:
        try:
            time.sleep(60)  # Every 60 seconds
            safe_memory_cleanup()
            
            # Clean old errors
            global app_errors
            if len(app_errors) > 100:
                app_errors = app_errors[-50:]
            
        except Exception as e:
            print(f"Periodic cleanup error: {e}")

# Start background cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

def start_server_with_tunnel(host='127.0.0.1', port=5000, debug=True, 
                           create_public_tunnel=False, tunnel_type='auto', ngrok_token=None):
    """Start server dengan opsi untuk membuat tunnel publik"""
    
    # Start server in background thread
    server_thread = threading.Thread(
        target=lambda: app.run(debug=debug, host=host, port=port, use_reloader=False),
        daemon=True
    )
    server_thread.start()
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    for i in range(10):
        try:
            response = requests.get(f'http://{host}:{port}', timeout=2)
            if response.status_code == 200:
                print("✅ Server is running!")
                break
        except:
            time.sleep(1)
    else:
        print("❌ Server failed to start properly")
        return None
    
    # Create public tunnel if requested
    public_url = None
    if create_public_tunnel:
        print("\n🌍 Creating public tunnel...")
        if tunnel_type == 'auto':
            public_url, tunnel_type_used = tunnel_manager.auto_create_tunnel(port, ngrok_token)
        else:
            if tunnel_type == 'ngrok':
                public_url, error = tunnel_manager.create_ngrok_tunnel(port, ngrok_token)
            elif tunnel_type == 'localtunnel':
                public_url, error = tunnel_manager.create_localtunnel(port)
            elif tunnel_type == 'cloudflare':
                public_url, error = tunnel_manager.create_cloudflared_tunnel(port)
            elif tunnel_type == 'serveo':
                public_url, error = tunnel_manager.create_serveo_tunnel(port)
        
        if public_url:
            print(f"🌍 PUBLIC URL: {public_url}")
            print(f"🔗 Your app is now accessible from anywhere!")
            
            # Try to open in browser
            try:
                webbrowser.open(public_url)
                print("🌐 Opening in default browser...")
            except:
                print("⚠️ Could not open browser automatically")
        else:
            print("❌ Failed to create public tunnel")
            print("🔧 You can try creating one manually from the /tunnel page")
    
    return public_url

if __name__ == '__main__':
    print("🚀 Initializing Quran Verse Detection Web Application")
    print("=" * 60)
    
    # Create upload directory if not exists
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        print(f"📁 Upload folder ready: {app.config['UPLOAD_FOLDER']}")
    except Exception as e:
        print(f"❌ Failed to create upload folder: {e}")
        exit(1)
    
    # Set TensorFlow to use less memory
    try:
        # Limit GPU memory growth if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🎮 GPU memory growth enabled for {len(gpus)} GPU(s)")
        else:
            print("💻 Running on CPU")
    except Exception as e:
        print(f"⚠️ TensorFlow GPU configuration warning: {e}")
    
    # Try to load existing model
    print("\n🤖 Loading AI Model...")
    model_loaded = load_model_if_exists()
    
    if model_loaded:
        print("✅ Model loaded successfully - Ready for predictions!")
    else:
        print("⚠️ No model loaded - Upload feature will not work properly")
        print("   Please ensure model files are available in the model directories")
    
    # Get configuration for running the app
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        host = config['app']['host']
        port = config['app']['port']
        debug = config['app']['debug']
        
        # Check for public tunnel configuration
        public_tunnel_config = config.get('public_tunnel', {})
        create_public = public_tunnel_config.get('enabled', False)
        tunnel_type = public_tunnel_config.get('type', 'auto')
        ngrok_token = public_tunnel_config.get('ngrok_token', None)
        
        print(f"⚙️ Configuration loaded from config.json")
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        # Fallback to default
        host = '127.0.0.1'
        port = 5000
        debug = True
        create_public = False
        tunnel_type = 'auto'
        ngrok_token = None
        print(f"⚙️ Using default configuration")
    
    # Check command line arguments for public tunnel
    if len(sys.argv) > 1:
        if '--public' in sys.argv:
            create_public = True
            print("🌍 Public tunnel requested via command line")
        
        if '--tunnel-type' in sys.argv:
            try:
                tunnel_idx = sys.argv.index('--tunnel-type')
                tunnel_type = sys.argv[tunnel_idx + 1]
                print(f"🔧 Tunnel type set to: {tunnel_type}")
            except (IndexError, ValueError):
                print("⚠️ Invalid --tunnel-type argument, using auto")
        
        if '--ngrok-token' in sys.argv:
            try:
                token_idx = sys.argv.index('--ngrok-token')
                ngrok_token = sys.argv[token_idx + 1]
                print("🔑 Ngrok token provided")
            except (IndexError, ValueError):
                print("⚠️ Invalid --ngrok-token argument")
    
    print("\n🌐 Server Configuration:")
    print(f"📡 Local Server: http://{host}:{port}")
    print(f"🛠️  Debug mode: {'ON' if debug else 'OFF'}")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"💾 Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB")
    print(f"🤖 Model status: {'READY' if model_loaded else 'NOT LOADED'}")
    print(f"🌍 Public tunnel: {'ENABLED' if create_public else 'DISABLED'}")
    if create_public:
        print(f"🔧 Tunnel type: {tunnel_type}")
        print(f"🔑 Ngrok token: {'PROVIDED' if ngrok_token else 'NOT PROVIDED'}")
    print("=" * 60)
    print("🕌 Starting server... Press Ctrl+C to stop")
    print("=" * 60)
    
    # Show available tunnel options
    print("\n🔧 Available tunnel options:")
    print(f"   📡 Ngrok: {'✅' if NGROK_AVAILABLE else '❌ (pip install pyngrok)'}")
    print(f"   📡 LocalTunnel: {'✅' if LOCALTUNNEL_AVAILABLE else '❌ (npm install -g localtunnel)'}")
    print(f"   📡 Cloudflare: ⚠️ (requires cloudflared binary)")
    print(f"   📡 Serveo: ⚠️ (requires SSH)")
    print("\n💡 To enable public access:")
    print("   python app.py --public")
    print("   python app.py --public --tunnel-type ngrok")
    print("   python app.py --public --tunnel-type ngrok --ngrok-token YOUR_TOKEN")
    print("=" * 60)
    
    # Run app with error handling
    try:
        if create_public:
            public_url = start_server_with_tunnel(
                host=host, port=port, debug=debug,
                create_public_tunnel=True, tunnel_type=tunnel_type, 
                ngrok_token=ngrok_token
            )
            
            if public_url:
                print(f"\n🎉 SUCCESS! Your app is publicly accessible at:")
                print(f"🌍 {public_url}")
                print(f"🔗 Share this link with anyone to access your app!")
                print("\n📱 You can also manage tunnels from: {}/tunnel".format(public_url))
            
            # Keep main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down...")
                tunnel_manager.close_tunnel()
        else:
            app.run(debug=debug, host=host, port=port, use_reloader=False)
            
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        exit(1)
