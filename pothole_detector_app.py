"""
VestGuard Road Intelligence — Pothole Detection Interface
Flask app for real-time pothole detection using best YOLO model.

Usage:
    1. Place your best trained model weights at: models/best.pt
    2. Install requirements: pip install flask ultralytics pillow
    3. Run: python pothole_detector_app.py
    4. Open on phone: http://YOUR_IP:5000
"""

from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os
import cv2
import numpy as np

app = Flask(__name__)

# ── Load best model ─────────────────────────────────────────────────────────
# Replace this path with your best trained model weights
MODEL_PATH = 'models/best.pt'

model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            # Try Ultralytics YOLO first (YOLOv8/11)
            model = YOLO(MODEL_PATH)
            print(f'Model loaded (Ultralytics) from {MODEL_PATH}')
        except Exception:
            # Fall back to torch.hub for YOLOv5
            import torch
            model = torch.hub.load('ultralytics/yolov5', 'custom',
                                   path=MODEL_PATH, force_reload=False)
            model.eval()
            print(f'Model loaded (YOLOv5 hub) from {MODEL_PATH}')
    else:
        print(f'WARNING: Model not found at {MODEL_PATH}')
        print('Using default YOLOv8n for demo — replace with your trained model')
        model = YOLO('yolov8n.pt')

# ── HTML Interface ───────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>PotholeAI — Road Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0a;
    --surface: #111111;
    --border: #1f1f1f;
    --accent: #f5c842;
    --accent2: #ff4d1c;
    --text: #f0f0f0;
    --muted: #555;
    --success: #22c55e;
    --danger: #ef4444;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Noise texture overlay */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
  }

  .container {
    position: relative;
    z-index: 1;
    max-width: 480px;
    margin: 0 auto;
    padding: 24px 20px 48px;
  }

  /* Header */
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .logo-icon {
    width: 36px;
    height: 36px;
    background: var(--accent);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
  }

  .logo-text {
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.5px;
  }

  .logo-text span {
    color: var(--accent);
  }

  .status-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--success);
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.2);
    padding: 4px 10px;
    border-radius: 99px;
  }

  .status-dot {
    width: 6px;
    height: 6px;
    background: var(--success);
    border-radius: 50%;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  /* Upload zone */
  .upload-zone {
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 40px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    background: var(--surface);
    position: relative;
    overflow: hidden;
    margin-bottom: 16px;
  }

  .upload-zone::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(245, 200, 66, 0.04) 0%, transparent 70%);
    pointer-events: none;
  }

  .upload-zone:hover, .upload-zone.drag-over {
    border-color: var(--accent);
    background: rgba(245, 200, 66, 0.03);
  }

  .upload-icon {
    font-size: 48px;
    margin-bottom: 12px;
    display: block;
  }

  .upload-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 6px;
  }

  .upload-sub {
    font-size: 13px;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
  }

  #fileInput { display: none; }

  /* Camera button */
  .btn-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 20px;
  }

  .btn {
    padding: 14px;
    border-radius: 12px;
    border: none;
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }

  .btn-primary {
    background: var(--accent);
    color: #000;
  }

  .btn-primary:hover {
    background: #f0ba20;
    transform: translateY(-1px);
  }

  .btn-secondary {
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
  }

  .btn-secondary:hover {
    border-color: var(--accent);
    color: var(--accent);
  }

  .btn-detect {
    width: 100%;
    padding: 18px;
    background: var(--accent2);
    color: white;
    border-radius: 14px;
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-bottom: 24px;
    display: none;
  }

  .btn-detect:hover {
    background: #e03d10;
    transform: translateY(-1px);
  }

  .btn-detect.visible { display: flex; }

  /* Preview */
  #previewContainer {
    display: none;
    margin-bottom: 16px;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--border);
    position: relative;
  }

  #previewContainer.visible { display: block; }

  #preview {
    width: 100%;
    display: block;
    max-height: 300px;
    object-fit: contain;
    background: #000;
  }

  .preview-label {
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid rgba(245, 200, 66, 0.3);
  }

  /* Results */
  #resultContainer { display: none; }
  #resultContainer.visible { display: block; }

  #resultImage {
    width: 100%;
    border-radius: 14px;
    border: 1px solid var(--border);
    margin-bottom: 16px;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 20px;
  }

  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 10px;
    text-align: center;
  }

  .stat-value {
    font-size: 24px;
    font-weight: 800;
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    line-height: 1;
    margin-bottom: 4px;
  }

  .stat-label {
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .alert-card {
    border-radius: 12px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
  }

  .alert-danger {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
  }

  .alert-success {
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.25);
  }

  .alert-icon { font-size: 24px; }

  .alert-title {
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 2px;
  }

  .alert-sub {
    font-size: 12px;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
  }

  /* Detections list */
  .detections-title {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    margin-bottom: 10px;
  }

  .detection-item {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .detection-label {
    font-size: 14px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .detection-dot {
    width: 8px;
    height: 8px;
    background: var(--accent2);
    border-radius: 50%;
  }

  .detection-conf {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: var(--accent);
  }

  .conf-bar {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 6px;
    overflow: hidden;
  }

  .conf-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    border-radius: 2px;
    transition: width 0.5s ease;
  }

  /* Loading */
  .loading {
    display: none;
    text-align: center;
    padding: 32px;
  }

  .loading.visible { display: block; }

  .spinner {
    width: 40px;
    height: 40px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 12px;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .loading-text {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: var(--muted);
  }

  /* Footer */
  .footer {
    text-align: center;
    margin-top: 32px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--muted);
  }

  .footer strong { color: var(--accent); }

  /* Reset btn */
  .btn-reset {
    width: 100%;
    padding: 12px;
    background: transparent;
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 10px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    cursor: pointer;
    margin-top: 8px;
    transition: all 0.15s;
  }

  .btn-reset:hover { color: var(--text); border-color: var(--text); }

  /* Confidence slider */
  .conf-control {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 16px;
  }

  .conf-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }

  .conf-label {
    font-size: 12px;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
  }

  .conf-value {
    font-size: 14px;
    font-weight: 700;
    color: var(--accent);
    font-family: 'Space Mono', monospace;
  }

  input[type="range"] {
    width: 100%;
    accent-color: var(--accent);
    cursor: pointer;
  }
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <div class="logo">
      <div class="logo-icon">🕳️</div>
      <div class="logo-text">Pothole<span>AI</span></div>
    </div>
    <div class="status-badge">
      <div class="status-dot"></div>
      LIVE
    </div>
  </div>

  <!-- Upload Zone -->
  <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
    <span class="upload-icon">📸</span>
    <div class="upload-title">Tap to capture or upload</div>
    <div class="upload-sub">JPG · PNG · WEBP supported</div>
    <input type="file" id="fileInput" accept="image/*" capture="environment">
  </div>

  <!-- Button row -->
  <div class="btn-row">
    <button class="btn btn-secondary" onclick="document.getElementById('fileInput').click()">
      📁 Gallery
    </button>
    <button class="btn btn-secondary" onclick="openCamera()">
      📷 Camera
    </button>
  </div>

  <!-- Confidence control -->
  <div class="conf-control">
    <div class="conf-header">
      <span class="conf-label">CONFIDENCE THRESHOLD</span>
      <span class="conf-value" id="confValue">0.25</span>
    </div>
    <input type="range" id="confSlider" min="0.1" max="0.9" step="0.05" value="0.25"
      oninput="document.getElementById('confValue').textContent = parseFloat(this.value).toFixed(2)">
  </div>

  <!-- Preview -->
  <div id="previewContainer">
    <div class="preview-label">READY TO ANALYZE</div>
    <img id="preview" src="" alt="Preview">
  </div>

  <!-- Detect button -->
  <button class="btn btn-detect" id="detectBtn" onclick="detect()">
    🔍 DETECT POTHOLES
  </button>

  <!-- Loading -->
  <div class="loading" id="loading">
    <div class="spinner"></div>
    <div class="loading-text">Analyzing road surface...</div>
  </div>

  <!-- Results -->
  <div id="resultContainer">
    <img id="resultImage" src="" alt="Detection Result">

    <div id="alertCard" class="alert-card"></div>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-value" id="statCount">0</div>
        <div class="stat-label">Potholes</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" id="statConf">—</div>
        <div class="stat-label">Max Conf</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" id="statTime">—</div>
        <div class="stat-label">ms</div>
      </div>
    </div>

    <div id="detectionsList"></div>

    <button class="btn-reset" onclick="reset()">↩ Analyze another image</button>
  </div>

  <div class="footer">
    Built by <strong>Arpit Awasthi</strong> · GHEC Bilaspur<br>
    Road Pothole Detection Research
  </div>

</div>

<script>
  let selectedFile = null;

  // File input handler
  document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;
    selectedFile = file;
    showPreview(file);
  });

  // Drag and drop
  const uploadZone = document.getElementById('uploadZone');
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      selectedFile = file;
      showPreview(file);
    }
  });

  function openCamera() {
    const input = document.getElementById('fileInput');
    input.setAttribute('capture', 'environment');
    input.click();
  }

  function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      document.getElementById('preview').src = e.target.result;
      document.getElementById('previewContainer').classList.add('visible');
      const btn = document.getElementById('detectBtn');
      btn.style.display = 'flex';
      btn.classList.add('visible');
      document.getElementById('resultContainer').classList.remove('visible');
      document.getElementById('detectionsList').innerHTML = '';
    };
    reader.readAsDataURL(file);
  }

  async function detect() {
    if (!selectedFile) return;

    const conf = parseFloat(document.getElementById('confSlider').value);

    // Show loading
    document.getElementById('loading').classList.add('visible');
    document.getElementById('detectBtn').style.display = 'none';
    document.getElementById('previewContainer').classList.remove('visible');
    document.getElementById('resultContainer').classList.remove('visible');

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('conf', conf);

    try {
      const startTime = Date.now();
      const response = await fetch('/detect', { method: 'POST', body: formData });
      const elapsed = Date.now() - startTime;
      const data = await response.json();

      document.getElementById('loading').classList.remove('visible');

      if (data.error) {
        alert('Detection failed: ' + data.error);
        reset();
        return;
      }

      showResults(data, elapsed);

    } catch (err) {
      document.getElementById('loading').classList.remove('visible');
      alert('Error: ' + err.message);
      reset();
    }
  }

  function showResults(data, elapsed) {
    // Result image
    document.getElementById('resultImage').src = 'data:image/jpeg;base64,' + data.image;

    // Stats
    const count = data.detections.length;
    document.getElementById('statCount').textContent = count;
    document.getElementById('statTime').textContent = elapsed;

    const maxConf = count > 0 ? Math.max(...data.detections.map(d => d.confidence)) : 0;
    document.getElementById('statConf').textContent = count > 0 ? (maxConf * 100).toFixed(0) + '%' : '—';

    // Alert card
    const alertCard = document.getElementById('alertCard');
    if (count > 0) {
      alertCard.className = 'alert-card alert-danger';
      alertCard.innerHTML = `
        <div class="alert-icon">⚠️</div>
        <div>
          <div class="alert-title">${count} Pothole${count > 1 ? 's' : ''} Detected</div>
          <div class="alert-sub">Road maintenance recommended</div>
        </div>`;
    } else {
      alertCard.className = 'alert-card alert-success';
      alertCard.innerHTML = `
        <div class="alert-icon">✅</div>
        <div>
          <div class="alert-title">No Potholes Detected</div>
          <div class="alert-sub">Road surface appears clear</div>
        </div>`;
    }

    // Detections list
    const list = document.getElementById('detectionsList');
    if (count > 0) {
      list.innerHTML = '<div class="detections-title">DETECTIONS</div>' +
        data.detections.map((d, i) => `
          <div class="detection-item">
            <div>
              <div class="detection-label">
                <div class="detection-dot"></div>
                Pothole ${i + 1}
              </div>
              <div class="conf-bar">
                <div class="conf-fill" style="width: ${d.confidence * 100}%"></div>
              </div>
            </div>
            <div class="detection-conf">${(d.confidence * 100).toFixed(1)}%</div>
          </div>`).join('');
    } else {
      list.innerHTML = '';
    }

    document.getElementById('resultContainer').classList.add('visible');
  }










</script>
</body>
</html>
"""

# ── Detection endpoint ────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    conf = float(request.form.get('conf', 0.25))

    try:
        # Read image
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Could not read image'}), 400

        # Run detection — handle both YOLOv5 and YOLOv8/11
        detections = []
        annotated = img.copy()

        try:
            # YOLOv8/11 Ultralytics API
            results = model(img, conf=conf, verbose=False)
            annotated = results[0].plot()
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    detections.append({
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()
                    })
        except Exception:
            # YOLOv5 torch.hub API
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model(img_rgb, size=640)
            annotated_rgb = results.render()[0]
            annotated = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            df = results.pandas().xyxy[0]
            df = df[df['confidence'] >= conf]
            for _, row in df.iterrows():
                detections.append({
                    'confidence': float(row['confidence']),
                    'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                })

        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        # Encode result image
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'image': img_b64,
            'detections': detections,
            'count': len(detections)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_model()
    # Get local IP for phone access
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = '0.0.0.0'

    print(f'\n{"="*50}')
    print(f' PotholeAI Detection Interface')
    print(f'{"="*50}')
    print(f' Local:   http://localhost:5000')
    print(f' Network: http://{local_ip}:5000')
    print(f' Open the Network URL on your phone!')
    print(f'{"="*50}\n')

    app.run(host='0.0.0.0', port=5000, debug=False)
