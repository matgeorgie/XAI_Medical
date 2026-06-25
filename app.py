"""MedAI Vision — Flask App (EfficientNet-B0)"""

import os, sys, uuid, pickle, base64, traceback, warnings, logging, sqlite3, json
from io import BytesIO
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import numpy as np
import cv2
import torch
from flask import Flask, request, jsonify, send_file, Response
from PIL import Image

# ── Structured logging ────────────────────────────────────────────────────────
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'medai.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger('medai')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
MANIFEST_PATH = os.path.join(BASE_DIR, 'models', 'manifest.pkl')
REPORTS_DIR   = os.path.join(BASE_DIR, 'static', 'reports')
DB_PATH       = os.path.join(BASE_DIR, 'history.db')
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── SQLite history database ───────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id   TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            filename    TEXT,
            pat_name    TEXT,
            pat_age     TEXT,
            pat_gender  TEXT,
            pat_id      TEXT,
            pat_doctor  TEXT,
            modality    TEXT,
            pred_class  TEXT,
            confidence  REAL,
            severity    TEXT,
            pdf_file    TEXT
        )
    """)
    con.commit()
    con.close()

def save_history(record: dict):
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT INTO history
            (report_id,timestamp,filename,pat_name,pat_age,pat_gender,
             pat_id,pat_doctor,modality,pred_class,confidence,severity,pdf_file)
        VALUES
            (:report_id,:timestamp,:filename,:pat_name,:pat_age,:pat_gender,
             :pat_id,:pat_doctor,:modality,:pred_class,:confidence,:severity,:pdf_file)
    """, record)
    con.commit()
    con.close()

def get_history(limit=50):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]

init_db()
log.info("Database initialised at %s", DB_PATH)

# ── Load models ───────────────────────────────────────────────────────────────
MANIFEST     = None
LUNG_MODEL   = None
BONE_MODEL   = None
ROUTER_MODEL = None
LOAD_ERROR   = None

try:
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            "models/manifest.pkl not found.\n"
            "Run: python train_cnn.py")

    with open(MANIFEST_PATH, 'rb') as f:
        MANIFEST = pickle.load(f)

    from models_cnn import LungModel, BoneModel, ModalityRouter

    log.info("Loading models...")
    LUNG_MODEL   = LungModel()
    BONE_MODEL   = BoneModel()
    ROUTER_MODEL = ModalityRouter()

    LUNG_MODEL.load_state_dict(
        torch.load(MANIFEST['lung_model_path'],   map_location=DEVICE, weights_only=True))
    BONE_MODEL.load_state_dict(
        torch.load(MANIFEST['bone_model_path'],   map_location=DEVICE, weights_only=True))
    ROUTER_MODEL.load_state_dict(
        torch.load(MANIFEST['router_model_path'], map_location=DEVICE, weights_only=True))

    LUNG_MODEL.eval().to(DEVICE)
    BONE_MODEL.eval().to(DEVICE)
    ROUTER_MODEL.eval().to(DEVICE)

    log.info("Models loaded — Backbone:%s  Lung:%.1f%%  Bone:%.1f%%  Device:%s",
             MANIFEST['backbone'],
             MANIFEST['lung_accuracy']*100,
             MANIFEST['bone_accuracy']*100,
             DEVICE)

except Exception as e:
    LOAD_ERROR = str(e)
    log.error("MODEL LOAD ERROR: %s", e)


# ── Severity scoring ──────────────────────────────────────────────────────────
def get_severity(pred_cls, confidence):
    if pred_cls == 'Normal':
        return 'Clear'
    if confidence >= 0.90:
        return 'High'
    if confidence >= 0.70:
        return 'Moderate'
    return 'Mild'


# ── DICOM support ─────────────────────────────────────────────────────────────
def load_dicom(file_bytes):
    try:
        import pydicom
        from pydicom.filebase import DicomBytesIO
        ds  = pydicom.dcmread(DicomBytesIO(file_bytes))
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        return arr
    except ImportError:
        log.warning("pydicom not installed — DICOM files not supported")
        return None
    except Exception as e:
        log.warning("DICOM read error: %s", e)
        return None


# ── Image preprocessing ───────────────────────────────────────────────────────
def preprocess(file_bytes, filename=''):
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.dcm':
        arr = load_dicom(file_bytes)
        if arr is not None:
            log.info("DICOM file processed successfully")
            return arr
    img = Image.open(BytesIO(file_bytes))
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (128,128,128))
        bg.paste(img, mask=img.split()[3]); img = bg
    elif img.mode not in ('RGB','L'):
        img = img.convert('RGB')
    img.thumbnail((512,512), Image.LANCZOS)
    return np.array(img)

def to_b64(arr):
    if arr.dtype != np.uint8:
        arr = np.clip(arr,0,255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if LOAD_ERROR:
        return Response(f"""<!DOCTYPE html>
<html><head><title>MedAI Vision — Setup Required</title>
<style>
  body{{font-family:monospace;background:#0a0f1e;color:#f0f4ff;
        display:flex;align-items:center;justify-content:center;
        min-height:100vh;margin:0}}
  .box{{background:#111827;border:1px solid #1e2a45;border-radius:12px;
        padding:40px;max-width:600px;border-top:3px solid #ffa502}}
  h1{{color:#ffa502;margin-bottom:16px}}
  pre{{background:#0a0f1e;padding:20px;border-radius:8px;
       color:#00ff88;overflow-x:auto;font-size:13px;line-height:1.8}}
  .err{{color:#ff4757;background:#1a0a0a;padding:16px;
        border-radius:8px;margin-bottom:20px;font-size:13px;white-space:pre-wrap}}
</style></head><body><div class="box">
  <h1>⚠ Training Required</h1>
  <div class="err">{LOAD_ERROR}</div>
  <pre>cd C:\\Users\\ASUS TUF\\Desktop\\review
python train_cnn.py</pre>
</div></body></html>""", mimetype='text/html')

    html_path = os.path.join(BASE_DIR, 'templates', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()


@app.route('/health')
def health():
    if LOAD_ERROR:
        return jsonify({'status': 'error', 'error': LOAD_ERROR}), 503
    return jsonify({
        'status':   'ok',
        'backbone': MANIFEST.get('backbone','EfficientNet-B0'),
        'device':   DEVICE,
        'lung_acc': f"{MANIFEST['lung_accuracy']*100:.1f}%",
        'bone_acc': f"{MANIFEST['bone_accuracy']*100:.1f}%",
    })


@app.route('/history')
def history():
    return jsonify(get_history())


@app.route('/analyze', methods=['POST'])
def analyze():
    if LOAD_ERROR or LUNG_MODEL is None:
        return jsonify({'success': False,
                        'error': f'Models not loaded: {LOAD_ERROR}'}), 503

    rid = str(uuid.uuid4())[:8].upper()
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        f = request.files['file']
        if not f or f.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        log.info("[%s] Received: %s", rid, f.filename)

        patient_info = {
            'name':     request.form.get('pat_name', '').strip(),
            'age':      request.form.get('pat_age', '').strip(),
            'gender':   request.form.get('pat_gender', '').strip(),
            'pat_id':   request.form.get('pat_id', '').strip(),
            'doctor':   request.form.get('pat_doctor', '').strip(),
            'symptoms': request.form.get('pat_symptoms', '').strip(),
            'history':  request.form.get('pat_history', '').strip(),
        }

        # Preprocess (handles DICOM + regular images)
        file_bytes = f.read()
        img    = preprocess(file_bytes, f.filename)
        from models_cnn import preprocess_image
        tensor = preprocess_image(img)

        # Image sanity check
        _img_check = img if len(img.shape)==3 else np.stack([img]*3,axis=-1)
        _hsv = cv2.cvtColor(_img_check.astype(np.uint8), cv2.COLOR_RGB2HSV)
        _sat = float(_hsv[:,:,1].mean())
        if _sat > 40:
            log.warning("[%s] Rejected — colour saturation %.0f", rid, _sat)
            return jsonify({'success': False,
                'error': f'This image does not appear to be an X-ray (colour saturation: {_sat:.0f}). '
                         'Please upload a valid medical X-ray image in greyscale format.'}), 400

        # Route
        modality, route_probs = ROUTER_MODEL.route(tensor, DEVICE)
        router_conf = float(route_probs.max())
        log.info("[%s] Modality: %s (%.1f%%)", rid, modality, router_conf*100)

        if router_conf < 0.65:
            log.warning("[%s] Rejected — router confidence %.1f%%", rid, router_conf*100)
            return jsonify({'success': False,
                'error': f'Could not recognise this as a valid X-ray (routing confidence: {router_conf*100:.1f}%). '
                         'Please upload a chest or bone X-ray.'}), 400

        # Diagnose
        if modality == 'bone':
            diag_model  = BONE_MODEL
            class_names = MANIFEST['bone_classes']
        else:
            diag_model  = LUNG_MODEL
            class_names = MANIFEST['lung_classes']

        pred_cls, pred_conf, pred_probs = diag_model.predict(tensor, DEVICE)
        class_idx = class_names.index(pred_cls)
        log.info("[%s] Prediction: %s (%.1f%%)", rid, pred_cls, pred_conf*100)

        if pred_conf < 0.55:
            log.warning("[%s] Rejected — low confidence %.1f%%", rid, pred_conf*100)
            return jsonify({'success': False,
                'error': f'Model could not make a confident diagnosis (confidence: {pred_conf*100:.1f}%). '
                         'Please upload a clear, standard medical X-ray.'}), 400

        # Severity
        severity = get_severity(pred_cls, pred_conf)
        log.info("[%s] Severity: %s", rid, severity)

        all_probs = {c: float(pred_probs[i]) for i,c in enumerate(class_names)}
        for c in ['Normal','Infection','Fracture','Tumor']:
            all_probs.setdefault(c, 0.0)

        # XAI
        from xai_engine import generate_heatmap, lime_explanation, shap_explanation, generate_explanation

        log.info("[%s] Running Grad-CAM++", rid)
        hm_b64, cam_map = generate_heatmap(img, diag_model, class_idx, pred_cls, DEVICE)
        if pred_cls == 'Normal':
            hm_b64 = ''

        log.info("[%s] Running LIME", rid)
        lime_regions, lime_weights, _ = lime_explanation(img, diag_model, class_idx, DEVICE)

        log.info("[%s] Running SHAP", rid)
        shap_vals, measurements, _ = shap_explanation(img, diag_model, class_idx, DEVICE)

        log.info("[%s] NLG", rid)
        explanation = generate_explanation(
            pred_cls, pred_conf, cam_map,
            lime_regions, lime_weights, shap_vals, measurements)

        # PDF
        log.info("[%s] Generating PDF", rid)
        from report_generator import build_pdf_report
        pdf_fn   = f'report_{rid}.pdf'
        pdf_path = os.path.join(REPORTS_DIR, pdf_fn)
        build_pdf_report(
            report_id=rid, img_array=img,
            predicted_class=pred_cls, confidence=pred_conf,
            all_probs=all_probs,
            class_names=['Normal','Infection','Fracture','Tumor'],
            heatmap_b64=hm_b64, lime_text=explanation,
            output_path=pdf_path, modality=modality,
            backbone=MANIFEST['backbone'],
            lime_regions=lime_regions, lime_weights=[w*100 for w in lime_weights],
            shap_values=shap_vals, measurements=measurements,
            patient_info=patient_info)

        # Save to history
        save_history({
            'report_id':  rid,
            'timestamp':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filename':   f.filename,
            'pat_name':   patient_info['name'],
            'pat_age':    patient_info['age'],
            'pat_gender': patient_info['gender'],
            'pat_id':     patient_info['pat_id'],
            'pat_doctor': patient_info['doctor'],
            'modality':   modality,
            'pred_class': pred_cls,
            'confidence': round(pred_conf*100, 1),
            'severity':   severity,
            'pdf_file':   pdf_fn,
        })
        log.info("[%s] Complete — %s %.1f%% [%s]", rid, pred_cls, pred_conf*100, severity)

        return jsonify({
            'success': True, 'report_id': rid,
            'prediction': {
                'class':          pred_cls,
                'confidence':     pred_conf,
                'confidence_pct': f'{pred_conf*100:.1f}%',
                'all_probs':      all_probs,
                'modality':       modality,
                'backbone':       MANIFEST['backbone'],
                'timestamp':      datetime.now().isoformat(),
                'severity':       severity,
            },
            'images':  {'original': to_b64(img), 'heatmap': hm_b64},
            'xai': {
                'explanation':  explanation,
                'lime_regions': lime_regions[:6],
                'lime_weights': [round(w*100) for w in lime_weights[:6]],
                'shap_values':  shap_vals,
                'measurements': measurements,
            },
            'report_url': f'/download_report/{pdf_fn}'
        })

    except Exception as e:
        tb = traceback.format_exc()
        log.error("[%s] ERROR: %s\n%s", rid, e, tb)
        return jsonify({'success': False, 'error': str(e), 'detail': tb}), 500


@app.route('/download_report/<filename>')
def download_report(filename):
    path = os.path.join(REPORTS_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        return jsonify({'error': 'Not found'}), 404
    return send_file(path, as_attachment=True,
                     download_name=filename, mimetype='application/pdf')


if __name__ == '__main__':
    log.info("MedAI Vision starting — http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)