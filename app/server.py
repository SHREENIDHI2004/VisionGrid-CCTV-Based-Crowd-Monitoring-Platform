import io, sys, time, threading, base64
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analytics import CCTVAnalytics

app = Flask(__name__, static_folder='static')
CORS(app)

PERSON_CLASS = 0
FONT         = cv2.FONT_HERSHEY_SIMPLEX
CYAN         = (0, 212, 255)
RED          = (30,  30, 220)
WHITE        = (255, 255, 255)
BLACK        = (0,   0,   0)

# ── Global stream state ────────────────────────────────────────────────────
class _State:
    def __init__(self):
        self.running        = False
        self.lock           = threading.Lock()
        self.main_frame     = None   # annotated BGR frame
        self.heat_frame     = None   # heatmap overlay frame
        self.analytics      = {}
        self.thread         = None
        self.analytics_eng  = None
        self.model          = None

G = _State()

# ── Model loader ───────────────────────────────────────────────────────────
def _load_model():
    from ultralytics import YOLO
    print("Loading YOLOv5s …")
    m = YOLO("yolov5s.pt")
    print("Model ready.")
    return m

# ── Drawing ────────────────────────────────────────────────────────────────
def _draw_frame(frame, persons, analytics, eng):
    h, w = frame.shape[:2]
    suspicious = analytics.get('suspicious', {})

    # Density grid colour overlay
    for r in range(4):
        for c in range(4):
            cnt = analytics.get('density_grid', [[0]*4]*4)[r][c]
            if cnt == 0:
                continue
            x1 = int(c * w/4);  y1 = int(r * h/4)
            x2 = int((c+1)*w/4); y2 = int((r+1)*h/4)
            alpha = min(0.08 + cnt * 0.05, 0.38)
            col = (0,0,200) if cnt >= 5 else (0,140,255) if cnt >= 3 else (0,180,0)
            ov = frame.copy()
            cv2.rectangle(ov, (x1,y1),(x2,y2), col, -1)
            cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

    # Grid lines
    for i in range(1, 4):
        cv2.line(frame,(int(i*w/4),0),(int(i*w/4),h),(80,80,80),1)
        cv2.line(frame,(0,int(i*h/4)),(w,int(i*h/4)),(80,80,80),1)

    # Per-person boxes + trails
    for p in persons:
        tid  = p.get('track_id', -1)
        x1,y1,x2,y2 = [int(v) for v in p['box']]
        conf = p['confidence']
        susp = str(tid) in suspicious
        col  = RED if susp else CYAN

        # Movement trail
        hist = list(eng.track_history.get(tid, []))
        pts  = [(int(x), int(y)) for x,y,_ in hist]
        for i in range(1, min(len(pts), 25)):
            fade = int(255 * i / 25)
            tc = (min(col[0],fade), min(col[1],fade), min(col[2],fade))
            cv2.line(frame, pts[-i-1], pts[-i], tc, 2)

        # Box + corner ticks
        cv2.rectangle(frame,(x1,y1),(x2,y2),col,3)
        tick, tk = 16, 4
        for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame,(px,py),(px+dx*tick,py),col,tk)
            cv2.line(frame,(px,py),(px,py+dy*tick),col,tk)

        # Label pill
        reason = suspicious.get(str(tid),'')
        lbl = f" ID:{tid} {conf:.0%}" + (f" [{reason}]" if reason else "")
        fs,ft = 0.55, 2
        (lw,lh),bl = cv2.getTextSize(lbl, FONT, fs, ft)
        by1 = max(0, y1-lh-bl-8)
        cv2.rectangle(frame,(x1,by1),(x1+lw+4,y1),col,-1)
        cv2.putText(frame,lbl,(x1+2,y1-bl-4),FONT,fs,BLACK,ft,cv2.LINE_AA)

    # Count badge (top-left)
    total = analytics.get('total', 0)
    susp_cnt = analytics.get('suspicious_count', 0)
    l1 = f"  {total}"
    l2 = "  PERSONS"
    l3 = f"  ⚠ {susp_cnt} SUSPICIOUS" if susp_cnt else "  ALL CLEAR"
    l3_col = RED if susp_cnt else (0,200,0)

    fs1,ft1 = 1.5,3
    fs2,ft2 = 0.5,1
    (nw,nh),nb = cv2.getTextSize(l1,FONT,fs1,ft1)
    (sw,sh),_  = cv2.getTextSize(l2,FONT,fs2,ft2)
    (aw,ah),_  = cv2.getTextSize(l3,FONT,fs2,ft2)
    bw = max(nw,sw,aw)+36; bh = nh+sh+ah+44
    mx,my = 18,18

    ov = frame.copy()
    cv2.rectangle(ov,(mx,my),(mx+bw,my+bh),(12,12,12),-1)
    cv2.addWeighted(ov,0.82,frame,0.18,0,frame)
    cv2.rectangle(frame,(mx,my),(mx+bw,my+bh),CYAN,2)
    cv2.rectangle(frame,(mx,my),(mx+6,my+bh),CYAN,-1)

    cv2.putText(frame,l1,(mx+14,my+14+nh),FONT,fs1,CYAN,ft1,cv2.LINE_AA)
    div = my+14+nh+nb+6
    cv2.line(frame,(mx+8,div),(mx+bw-8,div),CYAN,1)
    cv2.putText(frame,l2,(mx+14,div+sh+8),FONT,fs2,WHITE,ft2,cv2.LINE_AA)
    cv2.putText(frame,l3,(mx+14,div+sh+ah+20),FONT,fs2,l3_col,ft2,cv2.LINE_AA)

    return frame

# ── Camera thread ──────────────────────────────────────────────────────────
def _camera_loop(source, conf, iou):
    eng = CCTVAnalytics()
    G.analytics_eng = eng

    # Parse source
    try:
        src = int(source)
    except ValueError:
        src = source   # RTSP URL

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        G.running = False
        G.analytics = {'error': f'Cannot open: {source}'}
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_idx = 0
    placeholder = None

    while G.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.3)
            cap.release()
            cap = cv2.VideoCapture(src)
            continue

        frame_idx += 1
        if frame_idx % 2 != 0:   # process every 2nd frame
            continue

        try:
            results = G.model.track(
                frame, persist=True,
                conf=conf, iou=iou,
                classes=[PERSON_CLASS],
                verbose=False
            )

            persons = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    tid  = int(box.id[0]) if box.id is not None else -1
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    persons.append({
                        'track_id':   tid,
                        'confidence': float(box.conf[0]),
                        'box':        [x1, y1, x2, y2]
                    })

            stats = eng.update(persons, frame.shape)

            ann   = _draw_frame(frame.copy(), persons, stats, eng)
            heat  = eng.get_heatmap_overlay(frame.copy())

            with G.lock:
                G.main_frame  = ann
                G.heat_frame  = heat
                G.analytics   = stats

        except Exception as e:
            print(f"[WARN] Inference error: {e}")

    cap.release()
    print("Camera thread stopped.")

# ── MJPEG generator ────────────────────────────────────────────────────────
_NO_SIGNAL = None

def _get_no_signal(w=640, h=360):
    global _NO_SIGNAL
    if _NO_SIGNAL is None:
        img = np.zeros((h, w, 3), np.uint8)
        cv2.putText(img, "NO SIGNAL", (w//2-90, h//2),
                    FONT, 1.2, CYAN, 2, cv2.LINE_AA)
        _NO_SIGNAL = img
    return _NO_SIGNAL

def _mjpeg(get_frame_fn):
    while True:
        frame = get_frame_fn()
        if frame is None:
            frame = _get_no_signal()
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
        time.sleep(0.033)

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:p>')
def static_proxy(p):
    return send_from_directory(app.static_folder, p)

@app.route('/video_feed')
def video_feed():
    def _get():
        with G.lock: return G.main_frame
    return Response(_mjpeg(_get),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/heatmap_feed')
def heatmap_feed():
    def _get():
        with G.lock: return G.heat_frame
    return Response(_mjpeg(_get),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def api_stats():
    with G.lock:
        return jsonify({**G.analytics, 'running': G.running})

@app.route('/api/start', methods=['POST'])
def api_start():
    data   = request.get_json(silent=True) or {}
    source = data.get('source', '0')
    conf   = float(data.get('conf', 0.35))
    iou    = float(data.get('iou',  0.45))

    if G.running:
        return jsonify({'status': 'already running'})

    if G.model is None:
        try:
            G.model = _load_model()
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    G.running = True
    G.analytics_eng = None
    t = threading.Thread(target=_camera_loop, args=(source, conf, iou), daemon=True)
    G.thread = t
    t.start()
    return jsonify({'status': 'started', 'source': source})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    G.running = False
    with G.lock:
        G.main_frame = None
        G.heat_frame = None
    return jsonify({'status': 'stopped'})

@app.route('/api/reset_heatmap', methods=['POST'])
def api_reset_heatmap():
    if G.analytics_eng:
        G.analytics_eng.reset_heatmap()
    return jsonify({'status': 'heatmap reset'})

# Also keep single-image detect endpoint
@app.route('/detect', methods=['POST'])
def detect_image():
    import io as _io
    from PIL import Image
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    if G.model is None:
        try:
            G.model = _load_model()
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    file  = request.files['image']
    conf  = float(request.form.get('conf', 0.25))
    iou_v = float(request.form.get('iou',  0.45))
    img   = Image.open(_io.BytesIO(file.read())).convert('RGB')
    img_np= np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    results = G.model(img_np, conf=conf, iou=iou_v,
                      classes=[PERSON_CLASS], verbose=False)
    persons = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            persons.append({'name':'person','confidence':float(box.conf[0]),
                            'box':[x1,y1,x2,y2],'track_id':-1})
    from analytics import CCTVAnalytics
    tmp_eng = CCTVAnalytics(*img_bgr.shape[1::-1])
    ann = _draw_frame(img_bgr.copy(), persons,
                      {'total':len(persons),'suspicious':{},'density_grid':[[0]*4]*4,
                       'suspicious_count':0,'alerts':[]}, tmp_eng)
    _, buf = cv2.imencode('.jpg', ann, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64 = base64.b64encode(buf).decode()
    return jsonify({'image': f'data:image/jpeg;base64,{b64}',
                    'detections': persons,
                    'total_persons': len(persons),
                    'model_loaded': 'YOLOv5s — COCO'})


# ── Video processing state ─────────────────────────────────────────────────
class _VP:
    status   = 'idle'   # idle | processing | done | error
    progress = 0        # 0-100
    error    = None
    out_path = None

VP = _VP()
_OUT_VIDEO = Path(__file__).parent / 'static' / 'processed_output.mp4'

def _process_video_thread(input_path, conf, iou):
    VP.status   = 'processing'
    VP.progress = 0
    VP.error    = None

    try:
        if G.model is None:
            G.model = _load_model()

        cap   = cv2.VideoCapture(str(input_path))
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(str(_OUT_VIDEO), fourcc, fps, (w, h))
        eng    = CCTVAnalytics(w, h)
        idx    = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            VP.progress = int(idx / total * 100)

            results = G.model.track(frame, persist=True, conf=conf, iou=iou,
                                    classes=[PERSON_CLASS], verbose=False)
            persons = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    tid = int(box.id[0]) if box.id is not None else -1
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    persons.append({'track_id': tid,
                                    'confidence': float(box.conf[0]),
                                    'box': [x1,y1,x2,y2]})
            stats = eng.update(persons, frame.shape)
            ann   = _draw_frame(frame.copy(), persons, stats, eng)
            out.write(ann)

        cap.release()
        out.release()
        VP.progress = 100
        VP.status   = 'done'
        VP.out_path = str(_OUT_VIDEO)
        print(f"[VIDEO] Processing complete → {_OUT_VIDEO}")

    except Exception as e:
        VP.status = 'error'
        VP.error  = str(e)
        print(f"[VIDEO ERROR] {e}")
        import traceback; traceback.print_exc()


@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    if VP.status == 'processing':
        return jsonify({'error': 'Already processing a video'}), 409

    file       = request.files['video']
    conf       = float(request.form.get('conf', 0.35))
    iou        = float(request.form.get('iou',  0.45))
    input_path = Path(__file__).parent / 'static' / 'temp_input_video.mp4'
    file.save(str(input_path))

    t = threading.Thread(target=_process_video_thread,
                         args=(input_path, conf, iou), daemon=True)
    t.start()
    return jsonify({'status': 'started'})


@app.route('/video_progress')
def video_progress():
    return jsonify({'status': VP.status,
                    'progress': VP.progress,
                    'error':    VP.error})


@app.route('/download_video')
def download_video():
    if not _OUT_VIDEO.exists():
        return jsonify({'error': 'No processed video available'}), 404
    return send_from_directory(str(_OUT_VIDEO.parent),
                               _OUT_VIDEO.name, as_attachment=True,
                               download_name='person_detection_output.mp4')


if __name__ == '__main__':
    print("Starting CCTV Analytics Server on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
