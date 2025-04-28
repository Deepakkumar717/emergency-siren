from flask import Flask, request, jsonify, send_from_directory
import os
import torch
import torchaudio
import numpy as np
import uuid
from werkzeug.utils import secure_filename
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'flac'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class AudioClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 256)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AudioEmergencyDetector:
    def __init__(self, model_path="best_emergency_detector.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Fixed to match your pretrained model (2 classes: normal and siren)
        self.classes = ["normal", "siren"]
        self.model = self.load_model(model_path)
        self.target_sample_rate = 16000
        self.num_samples = 16000 * 5  # 5 seconds

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def load_model(self, model_path):
        model = AudioClassifier(num_classes=2).to(self.device)  # Fixed to 2 classes
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def predict_from_waveform(self, waveform):
        waveform = waveform.to(self.device)
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        mel_spec = mel_spec.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(mel_spec)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()

        return self.classes[predicted.item()], confidence

    def analyze_audio_file(self, audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if needed
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Just take first chunk
            if waveform.shape[1] >= self.num_samples:
                audio_chunk = waveform[:, :self.num_samples]
            else:
                audio_chunk = waveform  # if file is small

            class_name, confidence = self.predict_from_waveform(audio_chunk)

            return {
                "status": "success",
                "class": class_name,
                "confidence": confidence,
                "message": f"Detected: {class_name}, Confidence: {confidence:.2%}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

# Initialize detector
detector = AudioEmergencyDetector()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        result = detector.analyze_audio_file(filepath)
        os.remove(filepath)  # Clean up after processing
        
        return jsonify(result)
    
    return jsonify({"status": "error", "message": "Invalid file type"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)