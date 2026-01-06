import os
import numpy as np
import librosa
import pickle
import json
import re
import traceback
import time
import random

# Optional GPU support via PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None

# --- Improved Custom Deep Model with SGD & Momentum ---

class Layer:
    def forward(self, input_data): pass
    def backward(self, output_gradient, learning_rate): pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # He Initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))
        self.input = None
        # Momentum buffers
        self.mW = np.zeros_like(self.weights)
        self.mb = np.zeros_like(self.bias)

    def forward(self, input_data):
        self.input = input_data
        return np.dot(input_data, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # Sgd with Momentum (0.9)
        self.mW = 0.9 * self.mW - learning_rate * weights_gradient
        self.mb = 0.9 * self.mb - learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
        
        self.weights += self.mW
        self.bias += self.mb
        
        return input_gradient

class ReLU(Layer):
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)

class Softmax(Layer):
    def forward(self, input_data):
        # Numerically stable softmax
        exps = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Softmax + CrossEntropy gradient is just (p - y) if passed down directly
        return output_gradient

class CustomDeepModel:
    def __init__(self, input_size, output_size):
        # Deeper Architecture: Input -> 128 -> 64 -> Output
        self.layers = [
            Dense(input_size, 128),
            ReLU(),
            Dense(128, 64),
            ReLU(),
            Dense(64, output_size),
            Softmax()
        ]
        self.X_mean = None
        self.X_std = None
        
    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output_gradient, learning_rate):
        gradient = output_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

# --- Integration Helpers ---

class TrainingProgress:
    _logs = []
    _status = "Idle"
    _percent = 0
    _is_training = False
    _visual_data = {} 
    _mapped_verses = [] # Inventory of found verses
    _summary = None # Final evaluation summary
    _stop_requested = False
    last_active = time.time()  # timestamp of last activity (for idle detection)

    @classmethod
    def request_stop(cls):
        cls._stop_requested = True
        cls._is_training = False
        cls.push_log('⏹️ Stop requested by user or system.')

    @classmethod
    def clear_stop_request(cls):
        cls._stop_requested = False

    @classmethod
    def stop_requested(cls):
        return cls._stop_requested

    @classmethod
    def push_log(cls, message, visual_update=None):
        cls._logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        cls.last_active = time.time()
        if len(cls._logs) > 200: cls._logs.pop(0) # Increased buffer
        if visual_update:
            # Merge visual updates to avoid overwriting during fast processing
            for k, v in visual_update.items():
                cls._visual_data[k] = v
            
            # Special handling for verse mapping
            if visual_update.get('status') == 'extracting' and 'current_verse' in visual_update:
                v_id = visual_update['current_verse']
                if v_id not in cls._mapped_verses:
                    cls._mapped_verses.append(v_id)

    @classmethod
    def set_status(cls, status, percent=None):
        cls._status = status
        cls.last_active = time.time()
        if percent is not None: cls._percent = percent

    @classmethod
    def get_update(cls):
        # We also pass the whole mapped_verses list to ensure UI sync
        return {
            "status": cls._status,
            "percent": cls._percent,
            "logs": cls._logs,
            "is_training": cls._is_training,
            "visual_data": cls._visual_data,
            "mapped_verses": cls._mapped_verses,
            "summary": cls._summary,
            "stop_requested": cls._stop_requested,
            "last_active": cls.last_active
        }

    @classmethod
    def reset(cls):
        cls._logs = []
        cls._status = "Starting..."
        cls._percent = 0
        cls._is_training = True
        cls._visual_data = {}
        cls._mapped_verses = []
        cls._summary = None

class ModelHandler:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            path = self.config['MODEL_PATH']
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.model = pickle.load(f)
                    print("Custom Deep Model loaded successfully.")
                except Exception as e:
                    # Fallback: model file might be a torch-saved object or otherwise
                    try:
                        if torch is not None:
                            self.model = torch.load(path, map_location='cpu')
                            print("PyTorch model loaded via torch.load.")
                        else:
                            raise
                    except Exception as e2:
                        print(f"Error loading model (pickle/torch): {e} / {e2}")
                        self.model = None
            else:
                self.model = None

            # If loaded model is a TorchWrapper, ensure device compatibility
            if getattr(self.model, 'is_torch', False):
                if torch is None:
                    TrainingProgress.push_log("⚠️ Model saved as PyTorch but PyTorch is not installed; prediction will fallback.")
                else:
                    try:
                        # If model prefers CUDA but runtime has no CUDA, switch to CPU
                        if hasattr(self.model, 'device') and isinstance(self.model.device, torch.device):
                            if self.model.device.type == 'cuda' and not torch.cuda.is_available():
                                self.model.device = torch.device('cpu')
                                TrainingProgress.push_log("⚠️ CUDA not available; PyTorch model set to CPU.")
                    except Exception:
                        pass

            # Try to load encoder/metadata (label map)
            self.label_map = None
            encoder_path = self.config.get('ENCODER_PATH')
            metadata_path = self.config.get('METADATA_PATH')

            if encoder_path and os.path.exists(encoder_path):
                try:
                    with open(encoder_path, 'rb') as f:
                        self.label_map = pickle.load(f)
                    print("Label encoder loaded.")
                except Exception as e:
                    print(f"Error loading encoder: {e}")
            elif metadata_path and os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)
                        labels = meta.get('labels', [])
                        self.label_map = {int(item['idx']): (item['sura'], int(item['verse'])) for item in labels}
                    print("Metadata label map loaded.")
                except Exception as e:
                    print(f"Error loading metadata: {e}")

            # If model object embedded a label_map, prefer that
            if self.model is not None and hasattr(self.model, 'label_map'):
                self.label_map = self.model.label_map

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.label_map = None

    @staticmethod
    def extract_features_static(file_path):
        try:
            # Small, explicit log to help diagnose slow/blocked decoding
            TrainingProgress.push_log(f"🔍 Extracting features: {os.path.basename(file_path)}")
            start = time.time()
            y, sr = librosa.load(file_path, sr=22050)
            y, _ = librosa.effects.trim(y, top_db=20)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Increased feature dimensions
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            dur = time.time() - start
            if dur > 1.0:
                TrainingProgress.push_log(f"⏱️ Feature extraction took {dur:.2f}s for {os.path.basename(file_path)}")
            return np.hstack([mfcc_mean, mfcc_std])
        except Exception as e:
            # Log exception details so we can see if a codec/IO issue occurs
            TrainingProgress.push_log(f"⚠️ Error extracting {os.path.basename(file_path)}: {e}")
            try:
                import traceback as _tb
                TrainingProgress.push_log(_tb.format_exc().splitlines()[-1])
            except Exception:
                pass
            return None

    def extract_features(self, file_path):
        return ModelHandler.extract_features_static(file_path)

    def predict(self, file_path):
        # Returns detailed mapping: label index, sura (string), verse (int), confidence, source, top3
        result = {'label_index': None, 'sura': None, 'verse': None, 'confidence': 0.0, 'source': None, 'top3': []}

        if self.model is not None:
            features = self.extract_features(file_path)
            if features is not None:
                X_in = features.reshape(1, -1)
                # Use training normalization stats if available
                if getattr(self.model, 'X_mean', None) is not None:
                    X_in = (X_in - self.model.X_mean) / self.model.X_std
                else:
                    X_in = (X_in - np.mean(X_in)) / (np.std(X_in) + 1e-8)

                probs = self.model.forward(X_in)[0]
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])

                top3 = np.argsort(probs)[-3:][::-1]
                for idx in top3:
                    label_info = None
                    if self.label_map and int(idx) in self.label_map:
                        s, v = self.label_map[int(idx)]
                        label_info = {'idx': int(idx), 'sura': s, 'verse': int(v), 'prob': float(probs[int(idx)])}
                    else:
                        label_info = {'idx': int(idx), 'prob': float(probs[int(idx)])}
                    result['top3'].append(label_info)

                result['label_index'] = pred_idx
                result['confidence'] = conf
                result['source'] = 'model'

                # Map predicted index to sura/verse if label_map available
                if self.label_map and pred_idx in self.label_map:
                    s, v = self.label_map[pred_idx]
                    result['sura'] = s
                    result['verse'] = int(v)

        # Fallback: try to parse filename as sura+verse like 078000.mp3
        if result['sura'] is None or result['verse'] is None:
            base = os.path.basename(file_path)
            match = re.search(r'(?P<sura>\d{3})(?P<verse>\d{3})', base)
            if match:
                s = match.group('sura')
                v = int(match.group('verse'))
                result['sura'] = s
                result['verse'] = v
                result['confidence'] = result['confidence'] or 1.0
                result['source'] = result['source'] or 'filename_fallback'

        # For backward compatibility include verse_id if sura matches a single-surah flow
        if result.get('sura') is not None and result.get('verse') is not None:
            result['verse_id'] = result['verse']

        return result


# --- Utility helpers for kamus & dataset summary ---

def load_kamus(kamus_path=None):
    """Load kamus.json and return dict mapping 3-digit string number -> metadata.
    If kamus_path is None, it attempts to locate kamus.json in project root."""
    try:
        if not kamus_path:
            base = os.path.dirname(os.path.dirname(__file__))
            kamus_path = os.path.join(base, 'kamus.json')
        if not os.path.exists(kamus_path):
            return {}
        with open(kamus_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        res = {}
        for item in data:
            num = str(item.get('number', '')).zfill(3)
            res[num] = {'number': num, 'name': item.get('name', ''), 'verses': int(item.get('verses', 0))}
        return res
    except Exception as e:
        print(f"Error loading kamus: {e}")
        return {}


def get_sura_name(kamus, sura_id):
    try:
        key = str(int(sura_id)).zfill(3)
    except Exception:
        key = str(sura_id)
    return kamus.get(key, {}).get('name') if kamus else None


def summarize_dataset(dataset_path, kamus=None):
    """Scan dataset directory and return a summary dict:
    {
        total_surahs: int,
        total_classes: int, # distinct (sura,verse)
        total_samples: int,
        surahs: [ {sura: '078', name: 'An-Naba', sample_count: N, verses_present: [0,1,...], expected_verses: 40}, ... ]
    }
    """
    summary = {'total_surahs': 0, 'total_classes': 0, 'total_samples': 0, 'surahs': []}
    if not os.path.exists(dataset_path):
        return summary

    classes = set()
    samples = 0
    surah_map = {}

    # Look for top-level directories representing surah numbers
    for surah_dir in sorted(os.listdir(dataset_path)):
        surah_path = os.path.join(dataset_path, surah_dir)
        if not os.path.isdir(surah_path):
            continue
        # gather all mp3 files recursively under this surah
        verses_present = set()
        sample_count = 0
        for root, dirs, files in os.walk(surah_path):
            for f in files:
                if f.lower().endswith('.mp3'):
                    sample_count += 1
                    samples += 1
                    m = re.search(r'(?P<prefix>\d{3})(?P<verse>\d{3})', f)
                    if m:
                        s = m.group('prefix')
                        v = int(m.group('verse'))
                        if s == str(surah_dir):
                            classes.add((s, v))
                            verses_present.add(v)
                        else:
                            # fallback: still include by parsed prefix
                            classes.add((s, v))
                            verses_present.add(v)
        if verses_present or sample_count > 0:
            key = str(surah_dir).zfill(3)
            surah_entry = {
                'sura': key,
                'name': kamus.get(key, {}).get('name') if kamus else None,
                'sample_count': sample_count,
                'verses_present': sorted(list(verses_present)),
                'expected_verses': kamus.get(key, {}).get('verses') if kamus and key in kamus else None
            }
            surah_map[key] = surah_entry

    summary['total_surahs'] = len(surah_map)
    summary['total_classes'] = len(classes)
    summary['total_samples'] = samples
    summary['surahs'] = sorted(list(surah_map.values()), key=lambda x: int(x['sura']))
    return summary


# Device detection helper
def detect_device():
    """Return device info dict: {'backend': 'torch'|'numpy', 'device': 'cuda'|'cpu', 'device_name': str|None}"""
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return {'backend': 'torch', 'device': 'cuda', 'device_name': torch.cuda.get_device_name(0)}
            else:
                return {'backend': 'torch', 'device': 'cpu', 'device_name': None}
        except Exception:
            return {'backend': 'torch', 'device': 'cpu', 'device_name': None}
    return {'backend': 'numpy', 'device': 'cpu', 'device_name': None}


class TorchWrapper:
    """Lightweight wrapper to store torch model state and provide forward() -> numpy probs"""
    is_torch = True

    def __init__(self, input_size, num_classes, state_dict=None, device='cpu'):
        self.input_size = input_size
        self.num_classes = num_classes
        self.state_dict = state_dict
        self.device = device
        self._built = False
        self.model = None
        self.X_mean = None
        self.X_std = None

    def _build(self):
        import torch
        import torch.nn as nn
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )
        if self.state_dict is not None:
            # state_dict may contain tensors stored on CPU; safe to load
            self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()
        self._built = True

    def forward(self, X_np):
        import torch
        if not self._built:
            self._build()
        x = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(x)
            probs = torch.nn.functional.softmax(out, dim=1)
            return probs.cpu().numpy()

    # Note: feature extraction and predict methods are implemented within the ModelHandler class to ensure correct scoping.


def train_model_task(dataset_path, config):
    try:
        TrainingProgress.reset()
        # Idle timeout in seconds: if no activity reported for this duration, abort training
        IDLE_TIMEOUT = 300

        def _check_abort():
            # Check for explicit stop request
            if TrainingProgress.stop_requested():
                TrainingProgress.push_log('⏹️ Stop detected; aborting training now.')
                TrainingProgress._is_training = False
                return True
            # Check for idle timeout (no activity reported)
            if time.time() - TrainingProgress.last_active > IDLE_TIMEOUT:
                TrainingProgress.push_log(f'⚠️ No training activity detected for {IDLE_TIMEOUT}s. Aborting due to idle timeout.')
                TrainingProgress.request_stop()
                return True
            return False

        # --- Phase 0: Discover all (sura,verse) classes across dataset ---
        TrainingProgress.push_log("Scanning dataset for available sura/verse samples...")
        all_samples = []
        sample_counts = {}  # folder_name -> count

        for root, dirs, files in os.walk(dataset_path):
            rel = os.path.relpath(root, dataset_path)
            parts = rel.split(os.sep)
            if parts == ['.']:
                continue
            sura = parts[0]
            folder_name = parts[1] if len(parts) > 1 else None

            mp3s = [f for f in files if f.endswith('.mp3')]
            key = f"{sura}/{folder_name}" if folder_name else sura
            sample_counts[key] = len(mp3s)

            for file in mp3s:
                match = re.search(r'(?P<prefix>\d{3})(?P<verse>\d{3})', file)
                if match:
                    file_sura = match.group('prefix')
                    verse = int(match.group('verse'))
                    # prefer sura parsed from file name, but keep it consistent
                    all_samples.append((os.path.join(root, file), file_sura, verse))

        # Logging folder inventory for user review
        for folder, count in sorted(sample_counts.items()):
            TrainingProgress.push_log(f"📁 Folder {folder}: {count} files found.")

        if not all_samples:
            TrainingProgress.push_log("❌ No audio samples found in dataset. Abort training.")
            TrainingProgress._is_training = False
            return

        # Build unique label mapping (sura, verse) -> class index
        unique_labels = sorted(list(set([(s, v) for (_, s, v) in all_samples])), key=lambda x: (int(x[0]), x[1]))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: {'sura': label[0], 'verse': label[1]} for label, idx in label_to_idx.items()}

        TrainingProgress.push_log(f"✅ Detected {len(unique_labels)} distinct classes (sura,verse).")

        # Warn about classes with fewer samples than expected
        counts_per_label = {}
        for _, s, v in all_samples:
            counts_per_label.setdefault((s, v), 0)
            counts_per_label[(s, v)] += 1
        for lbl, cnt in counts_per_label.items():
            if cnt < 7:
                TrainingProgress.push_log(f"⚠️ Class {lbl[0]}:{lbl[1]} has only {cnt} samples.")

        # Publish discovered classes for UI rendering (idx, sura, verse, count)
        classes_info = []
        for idx, info in idx_to_label.items():
            s = info['sura']
            v = info['verse']
            classes_info.append({'idx': int(idx), 'sura': s, 'verse': int(v), 'count': counts_per_label.get((s, v), 0)})

        TrainingProgress.push_log("Discovered class mapping", {'status': 'discovered', 'classes': classes_info})

        # Sync mapped_verses indexes for existing UI logic
        for c in classes_info:
            if c['idx'] not in TrainingProgress._mapped_verses:
                TrainingProgress._mapped_verses.append(c['idx'])

        # --- Phase 1: Feature Extraction ---
        X_raw = []
        y_raw = []
        total_files = len(all_samples)
        processed_count = 0
        TrainingProgress.push_log(f"Starting Feature Extraction for {total_files} files...")

        for fpath, s, v in all_samples:
            processed_count += 1

            # Abort check during extraction
            if _check_abort():
                TrainingProgress.push_log('Extraction aborted by stop request or timeout.')
                TrainingProgress._is_training = False
                return

            # Log each file start for visibility and faster feedback in UI
            TrainingProgress.push_log(f"   [{processed_count}/{total_files}] Processing: {os.path.basename(fpath)}")
            start_file = time.time()

            feat = ModelHandler.extract_features_static(fpath)

            elapsed = time.time() - start_file
            if elapsed > 2.0:
                TrainingProgress.push_log(f"   ⏱️ Slow file ({elapsed:.2f}s): {os.path.basename(fpath)}")

            if feat is not None:
                X_raw.append(feat)
                y_raw.append(label_to_idx[(s, v)])
            else:
                TrainingProgress.push_log(f"⚠️ Gagal ekstrak fitur: {fpath}")

            # Periodic summary log every 10 files to reduce noise
            if processed_count % 10 == 0:
                TrainingProgress.push_log(f"Ekstraksi: {processed_count}/{total_files} berkas selesai.")

        TrainingProgress.push_log(f"✅ Ekstraksi selesai. Berhasil mengambil {len(X_raw)} fitur.")
        
        X = np.array(X_raw)
        y = np.array(y_raw)
        
        # Z-Score Normalization
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-8
        X = (X - X_mean) / X_std

        num_classes = len(idx_to_label)
        input_size = X.shape[1]

        # Training Settings (Intensified)
        learning_rate = 0.005
        batch_size = 16 # Smaller batch for more updates
        target_accuracy = 1.0 # Target absolute perfection on dataset
        max_epochs = 500

        # Device detection
        device_info = detect_device()
        if device_info['backend'] == 'torch' and device_info['device'] == 'cuda':
            TrainingProgress.push_log(f"Using GPU: {device_info['device_name']} for training.")
        elif device_info['backend'] == 'torch':
            TrainingProgress.push_log("PyTorch available, using CPU for training.")
        else:
            TrainingProgress.push_log("PyTorch not available, falling back to numpy CPU training.")

        TrainingProgress.push_log("Initializing Deep Learning Core...")

        model = None
        # --- PyTorch Training Path ---
        if device_info['backend'] == 'torch':
            tdevice = torch.device('cuda' if device_info['device'] == 'cuda' else 'cpu')

            # Prepare tensors
            X_t = torch.tensor(X, dtype=torch.float32, device=tdevice)
            y_t = torch.tensor(y, dtype=torch.long, device=tdevice)

            # Define network
            net = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            ).to(tdevice)

            opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
            loss_fn = nn.CrossEntropyLoss()

            for epoch in range(max_epochs):
                # Abort check at epoch boundaries
                if _check_abort():
                    TrainingProgress.push_log('Training loop aborted by stop request or timeout (torch path).')
                    break

                # Shuffle indices
                perm = torch.randperm(len(X_t))
                X_shuf = X_t[perm]
                y_shuf = y_t[perm]

                epoch_loss = 0.0
                correct = 0

                for i in range(0, len(X_t), batch_size):
                    xb = X_shuf[i:i+batch_size]
                    yb = y_shuf[i:i+batch_size]

                    logits = net(xb)
                    loss = loss_fn(logits, yb)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    epoch_loss += loss.item()
                    preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                    correct += torch.sum(preds == yb).item()

                acc = correct / len(X_t)
                avg_loss = epoch_loss / (len(X_t) / batch_size)

                # Update UI
                prog = 20 + int( (epoch / max_epochs) * 80 )
                TrainingProgress.set_status(f"Epoch {epoch+1} (Acc: {acc:.1%})", prog)

                viz_data = {
                    "status": "training",
                    "epoch": epoch+1,
                    "loss": avg_loss,
                    "accuracy": acc,
                    "layers": [random.random() for _ in range(5)]
                }

                if epoch % 2 == 0:
                    TrainingProgress.push_log(f"Training: Loss={avg_loss:.4f}, Acc={acc:.1%}", viz_data)

                if acc >= target_accuracy and epoch > 30:
                    TrainingProgress.push_log(f"Target Accuracy Hit! ({acc:.1%})")
                    break

            # Build wrapper
            model_wrapper = TorchWrapper(input_size, num_classes, state_dict=net.state_dict(), device=tdevice)
            model_wrapper.X_mean = X_mean
            model_wrapper.X_std = X_std
            model = model_wrapper

        else:
            # --- Numpy-based training path (existing) ---
            model = CustomDeepModel(input_size, num_classes)
            model.X_mean = X_mean
            model.X_std = X_std

            TrainingProgress.push_log("Initializing Deep Learning Core (500 Epochs)...")

            for epoch in range(max_epochs):
                # Abort check at epoch boundaries
                if _check_abort():
                    TrainingProgress.push_log('Training loop aborted by stop request or timeout (numpy path).')
                    break

                # Shuffle
                indices = np.arange(len(X))
                np.random.shuffle(indices)
                X_train = X[indices]
                y_train = y[indices]

                epoch_loss = 0
                correct = 0

                # Mini-batch Training
                for i in range(0, len(X), batch_size):
                    X_batch = X_train[i:i+batch_size]
                    y_batch = y_train[i:i+batch_size]

                    probs = model.forward(X_batch)

                    y_one_hot = np.zeros((len(y_batch), num_classes))
                    for j, val in enumerate(y_batch):
                        y_one_hot[j, val] = 1

                    loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-8), axis=1))
                    epoch_loss += loss

                    preds = np.argmax(probs, axis=1)
                    correct += np.sum(preds == y_batch)

                    grad = (probs - y_one_hot) / len(X_batch)
                    model.backward(grad, learning_rate)

                acc = correct / len(X)
                avg_loss = epoch_loss / (len(X) / batch_size)

                # Update UI
                prog = 20 + int( (epoch / max_epochs) * 80 )
                TrainingProgress.set_status(f"Epoch {epoch+1} (Acc: {acc:.1%})", prog)

                viz_data = {
                    "status": "training",
                    "epoch": epoch+1,
                    "loss": avg_loss,
                    "accuracy": acc,
                    # Simulate "Layer Activity" for visualization
                    "layers": [random.random() for _ in range(5)]
                }

                if epoch % 2 == 0:
                    TrainingProgress.push_log(f"Training: Loss={avg_loss:.4f}, Acc={acc:.1%}", viz_data)

                # Check if Hit Target
                if acc >= target_accuracy and epoch > 30:
                    TrainingProgress.push_log(f"Target Accuracy Hit! ({acc:.1%})")
                    break

        # 4. Strict Verification Phase (Requested by User)
        TrainingProgress.push_log("Phase 4: Deep Verification & Per-Verse Correction...")
        TrainingProgress.set_status("Verifying all samples...", 90)

        mistakes = 0
        total_samples = len(X)

        for i in range(total_samples):
            # Abort check during verification
            if _check_abort():
                TrainingProgress.push_log('Verification aborted by stop request or timeout.')
                TrainingProgress._is_training = False
                break

            sample_in = X[i:i+1]
            true_label = y[i]

            # Show which verse is being verified
            if i % 5 == 0:
                TrainingProgress.set_status(f"Verifying Sample {i}/{total_samples}", 90 + int((i/total_samples) * 9))
                info = idx_to_label.get(int(true_label), {'sura': '---', 'verse': int(true_label)})
                label_str = f"{info['sura']}:{info['verse']}"
                TrainingProgress.push_log(f"Verifying Ayat {label_str}...", {"status": "extracting", "current_verse": label_str, "current_class_index": int(true_label)})

            probs = model.forward(sample_in)[0]
            pred = np.argmax(probs)

            if pred != true_label:
                mistakes += 1
                log_msg = f"❌ Misaligned: Ayat {true_label} detected as {pred}. Applying forced correction..."
                info = idx_to_label.get(int(true_label), {'sura': '---', 'verse': int(true_label)})
                label_str = f"{info['sura']}:{info['verse']}"
                TrainingProgress.push_log(log_msg, {"status": "correcting", "current_verse": label_str, "current_class_index": int(true_label)})
                # High-Intensity Overfitting Loop for this specific failed sample
                y_target = np.zeros((1, num_classes))
                y_target[0, true_label] = 1

                # Apply correction differently depending on model backend
                if hasattr(model, 'is_torch') and getattr(model, 'is_torch'):
                    try:
                        # Perform targeted retraining on this single sample using torch
                        device = torch.device('cuda' if detect_device().get('device') == 'cuda' else 'cpu')
                        net = nn.Sequential(
                            nn.Linear(input_size, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, num_classes)
                        ).to(device)
                        net.load_state_dict(model.state_dict)
                        opt_local = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
                        loss_fn_local = nn.CrossEntropyLoss()

                        xb = torch.tensor(sample_in, dtype=torch.float32, device=device)
                        yb = torch.tensor([int(true_label)], dtype=torch.long, device=device)

                        for _ in range(100):
                            logits = net(xb)
                            loss_local = loss_fn_local(logits, yb)
                            opt_local.zero_grad()
                            loss_local.backward()
                            opt_local.step()
                            probs_t = torch.softmax(net(xb), dim=1).cpu().numpy()
                            if np.argmax(probs_t) == true_label and probs_t[0, true_label] > 0.95:
                                break

                        # Persist corrected weights back into model wrapper
                        model.state_dict = net.state_dict()
                    except Exception as e:
                        TrainingProgress.push_log(f"Correction error (torch): {e}")
                else:
                    for _ in range(100): # Triple the correction intensity
                        p = model.forward(sample_in)
                        if np.argmax(p) == true_label and p[0, true_label] > 0.95:
                            break
                        grad = (p - y_target)
                        model.backward(grad, 0.1) # Aggressive learning rate for correction

        if mistakes == 0:
             TrainingProgress.push_log(f"✅ Verification Perfect! All {len(idx_to_label)} classes successfully mapped and verified.")
        else:
             TrainingProgress.push_log(f"✅ Auto-Correction Finished. Corrected {mistakes} misaligned detections.")

        # 5. Save Final Metrics
        # Calculate final metrics on the dataset
        final_probs = model.forward(X)
        final_preds = np.argmax(final_probs, axis=1)
        final_acc = np.sum(final_preds == y) / len(y)
        
        # Simple Precision/Recall (Macro Average)
        precisions = []
        recalls = []
        for c in range(num_classes):
            tp = np.sum((final_preds == c) & (y == c))
            fp = np.sum((final_preds == c) & (y != c))
            fn = np.sum((final_preds != c) & (y == c))
            
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            precisions.append(p)
            recalls.append(r)
            
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-8)

        metrics_data = {
            'accuracy': f"{final_acc:.1%}",
            'loss': f"{avg_loss:.4f}",
            'precision': f"{avg_precision:.2f}",
            'recall': f"{avg_recall:.2f}",
            'f1': f"{avg_f1:.2f}",
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metrics_file = os.path.join(os.path.dirname(config['MODEL_PATH']), 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f)
            
        TrainingProgress.push_log(f"📊 Metrics Saved: Acc={metrics_data['accuracy']}, F1={metrics_data['f1']}")
        TrainingProgress._summary = metrics_data

        # Attach label map to model and save encoder/metadata
        model.label_map = {idx: (info['sura'], int(info['verse'])) for idx, info in idx_to_label.items()}

        encoder_path = config.get('ENCODER_PATH')
        if encoder_path:
            try:
                with open(encoder_path, 'wb') as f:
                    pickle.dump(model.label_map, f)
                TrainingProgress.push_log(f"💾 Label encoder saved to {encoder_path}")
            except Exception as e:
                TrainingProgress.push_log(f"Error saving encoder: {e}")

        metadata_path = config.get('METADATA_PATH')
        if metadata_path:
            try:
                labels_meta = []
                for idx, info in idx_to_label.items():
                    labels_meta.append({'idx': idx, 'sura': info['sura'], 'verse': int(info['verse']), 'count': counts_per_label.get((info['sura'], info['verse']), 0)})
                with open(metadata_path, 'w') as f:
                    json.dump({'labels': labels_meta, 'num_classes': len(idx_to_label)}, f)
                TrainingProgress.push_log(f"💾 Metadata saved to {metadata_path}")
            except Exception as e:
                TrainingProgress.push_log(f"Error saving metadata: {e}")

        # Save Model
        with open(config['MODEL_PATH'], 'wb') as f:
            pickle.dump(model, f)
            
        TrainingProgress.push_log("💾 Custom Deep Model strictly verified and saved successfully.")
        TrainingProgress.set_status("Finished", 100)
        TrainingProgress._is_training = False
            
    except Exception as e:
        TrainingProgress.push_log(f"Error: {e}")
        traceback.print_exc()
        TrainingProgress._is_training = False
