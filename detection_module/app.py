# from flask import Flask, render_template, Response, jsonify, send_from_directory
# import cv2
# import os
# import datetime
# import threading
# import time
# from typing import Optional
# from email.message import EmailMessage
# import smtplib
# import numpy as np
# from PIL import Image 
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)
#  # ✅ Replacement for imghdr
# 6
# # --- Helper: detect image type (replaces imghdr.what) ---
# def get_image_format(data_bytes):
#     try:
#         from io import BytesIO
#         with Image.open(BytesIO(data_bytes)) as img:
#             return img.format.lower()
#     except Exception:
#         return 'jpg'

# # Try to import optional dependencies
# try:
#     import pywhatkit
#     PYWHATKIT_AVAILABLE = True
# except ImportError:
#     print("⚠️ pywhatkit not available. WhatsApp alerts will be disabled.")
#     PYWHATKIT_AVAILABLE = False

# try:
#     from model_inference import PPEModel
#     MODEL_AVAILABLE = True
# except ImportError as e:
#     print(f"❌ Error importing PPEModel: {e}")
#     MODEL_AVAILABLE = False
#     # Dummy fallback class
#     class PPEModel:
#         def __init__(self, *args, **kwargs): pass
#         def detect(self, frame): return []
#         def check_ppe_per_person(self, detections, iou_threshold=0.05): return []
#         def annotate_frame(self, frame, detections, ppe_results=None): return frame

# # Constants
# WEIGHTS = os.getenv('WEIGHTS_PATH', 'runs/train_ppe/ppe_detector/weights/best1.pt')
# DEVICE = os.getenv('DEVICE', 'cpu')
# CONF = float(os.getenv('CONF', 0.5))
# IOU = float(os.getenv('IOU', 0.45))
# MANAGER_PHONE = "+919156008476"
# ALERT_COOLDOWN_SECONDS = 300
# EMAIL_INTERVAL_SECONDS = 200

# # Globals
# # Real-time summary + thread lock
# # Use keys total, wearing_ppe, not_wearing_ppe as per frontend contract
# latest_summary = {"total": 0, "wearing_ppe": 0, "not_wearing_ppe": 0, "timestamp": None}
# summary_lock = threading.Lock()

# # Captured images list and directory
# CAPTURE_DIR = os.path.join(os.path.dirname(__file__), "captured_images")
# os.makedirs(CAPTURE_DIR, exist_ok=True)
# captured_images = []   # list of filenames, newest appended
# captured_images_lock = threading.Lock()

# try:
#     if MODEL_AVAILABLE:
#         model = PPEModel(weights_path=WEIGHTS, device=DEVICE, conf=CONF, iou=IOU)
#         print("✅ PPE Model loaded successfully")
#     else:
#         model = PPEModel()
#         print("⚠️ Using dummy model - detection will not work properly")
# except Exception as e:
#     print(f"❌ Error loading PPE model: {e}")
#     model = PPEModel()

# try:
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("⚠️ Could not open webcam at index 0, trying index 1...")
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():  
#             print("⚠️ Could not open webcam. Please check your camera connection.")
# except Exception as e:
#     print(f"❌ Error initializing camera: {e}")
#     cap = None

# last_alert_time = 0

# # Email CONFIG
# CONFIG = {
#     'email_alerts': True,
#     'email_sender': 'navaleprathmesh14@gmail.com',
#     'email_receiver': 'navaleprathmesh123@gmail.com',
#     'email_smtp': 'smtp.gmail.com',
#     'email_port': 587,
#     'email_password': 'dwzi bpwb iynb nphw'
# }

# # --- Functions ---
# def send_whatsapp_alert(message):
#     if not PYWHATKIT_AVAILABLE:
#         print("⚠️ WhatsApp alerts disabled - pywhatkit not available")
#         return

#     def worker():
#         try:
#             print("⏳ Waiting 30 seconds before sending WhatsApp alert...")
#             time.sleep(30)
#             now = datetime.datetime.now()
#             hour, minute = now.hour, now.minute + 1
#             pywhatkit.sendwhatmsg(
#                 MANAGER_PHONE,
#                 message,
#                 hour,
#                 minute,
#                 wait_time=10,
#                 tab_close=True,
#                 close_time=3
#             )
#             print(f"✅ WhatsApp Alert scheduled: {message}")
#         except Exception as e:
#             print(f"❌ Failed to send WhatsApp alert: {e}")
#     threading.Thread(target=worker, daemon=True).start()


# def send_email_alert(subject: str, body: str, image_path: Optional[str] = None):
#     if not CONFIG['email_alerts']:
#         print('Email alerts disabled in config')
#         return False
#     try:
#         msg = EmailMessage()
#         msg['Subject'] = subject
#         msg['From'] = CONFIG['email_sender']
#         msg['To'] = CONFIG['email_receiver']
#         msg.set_content(body)

#         if image_path and os.path.exists(image_path):
#             with open(image_path, 'rb') as f:
#                 data = f.read()
#             ext = get_image_format(data)
#             msg.add_attachment(data, maintype='image', subtype=ext, filename=os.path.basename(image_path))

#         server = smtplib.SMTP(CONFIG['email_smtp'], CONFIG['email_port'])
#         server.starttls()
#         server.login(CONFIG['email_sender'], CONFIG['email_password'])
#         server.send_message(msg)
#         server.quit()
#         print('✅ Email alert sent to', CONFIG['email_receiver'])
#         return True
#     except Exception as e:
#         print('❌ Failed to send email alert:', e)
#         return False


# def email_scheduler():
#     while True:
#         try:
#             subject = f"System Heartbeat 💖 at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
#             body = "This is an automated system check to confirm the email service is operational."
#             send_email_alert(subject=subject, body=body, image_path=None)
#         except Exception as e:
#             print(f"❌ Error in email scheduler: {e}")
#         time.sleep(EMAIL_INTERVAL_SECONDS)


# def gen_frames():
#     global last_alert_time, latest_summary, captured_images

#     if cap is None:
#         placeholder = cv2.imread('placeholder.jpg') if os.path.exists('placeholder.jpg') else None
#         if placeholder is None:
#             placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
#             cv2.putText(placeholder, "Camera Not Available", (200, 240),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         ret, buffer = cv2.imencode('.jpg', placeholder)
#         if ret:
#             yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#         return

#     while True:
#         try:
#             success, frame = cap.read()
#             if not success or frame is None:
#                 # small sleep to avoid tight loop if camera fails
#                 time.sleep(0.05)
#                 continue

#             frame_small = cv2.resize(frame, (640, 480))

#             if MODEL_AVAILABLE:
#                 detections = model.detect(frame_small)
#                 ppe_results = model.check_ppe_per_person(detections, iou_threshold=0.05)
#                 annotated = model.annotate_frame(frame_small, detections, ppe_results)

#                 # --- summary counts ---
#                 total_people = len(ppe_results) if ppe_results is not None else 0
#                 not_wearing = sum(1 for pr in ppe_results if pr.get('missing'))
#                 wearing = total_people - not_wearing

#                 # update shared summary
#                 with summary_lock:
#                     latest_summary['total'] = total_people
#                     latest_summary['wearing_ppe'] = wearing
#                     latest_summary['not_wearing_ppe'] = not_wearing
#                     latest_summary['timestamp'] = datetime.datetime.utcnow().isoformat() + 'Z'

#                 # --- save annotated image if violation found ---
#                 if not_wearing > 0 and (time.time() - last_alert_time) > ALERT_COOLDOWN_SECONDS:
#                     ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
#                     filename = f"ppe_violation_{ts}.jpg"
#                     image_path = os.path.join(CAPTURE_DIR, filename)
#                     cv2.imwrite(image_path, annotated)
#                     with captured_images_lock:
#                         captured_images.append(filename)
#                         # optionally keep only last N images
#                         if len(captured_images) > 200:
#                             # remove oldest file on disk and from list
#                             old = captured_images.pop(0)
#                             try:
#                                 os.remove(os.path.join(CAPTURE_DIR, old))
#                             except Exception:
#                                 pass

#                     # send email alert (existing function)
#                     send_email_alert(
#                         f"PPE Missing Alert",
#                         f"Alert: The following PPE is missing from {not_wearing} person(s).",
#                         image_path
#                     )
#                     last_alert_time = time.time()

#             else:
#                 annotated = frame_small.copy()
#                 cv2.putText(annotated, "Model Not Available", (10, 30),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 with summary_lock:
#                     latest_summary['total'] = 0
#                     latest_summary['wearing_ppe'] = 0
#                     latest_summary['not_wearing_ppe'] = 0
#                     latest_summary['timestamp'] = datetime.datetime.utcnow().isoformat() + 'Z'

#             ret, buffer = cv2.imencode('.jpg', annotated)
#             if not ret:
#                 continue
#             yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

#         except Exception as e:
#             print(f"❌ Error in gen_frames: {e}")
#             # return a small error frame
#             error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
#             cv2.putText(error_frame, f"Error: {str(e)[:50]}", (10, 240),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#             ret, buffer = cv2.imencode('.jpg', error_frame)
#             if ret:
#                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             break

# # --- Flask Routes ---
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/status')
# def status():
#     return jsonify({"status": "running"}), 200

# @app.route('/detection_summary')
# def detection_summary_route():
#     with summary_lock:
#         return jsonify(latest_summary), 200

# @app.route('/get_captured_images')
# def get_captured_images():
#     with captured_images_lock:
#         return jsonify({"images": list(captured_images)}), 200

# @app.route('/captured_images/<filename>')
# def serve_captured_image(filename):
#     return send_from_directory(CAPTURE_DIR, filename)

# if __name__ == '__main__':
#     email_thread = threading.Thread(target=email_scheduler, daemon=True)
#     email_thread.start()
#     app.run(host='0.0.0.0', port=5000, debug=False)


from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Global model variable (DO NOT load at import time)
# --------------------------------------------------
model = None


# --------------------------------------------------
# Load model AFTER server starts
# --------------------------------------------------
@app.before_first_request
def load_model():
    global model
    try:
        from ultralytics import YOLO

        WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "weights/best.pt")
        DEVICE = os.getenv("DEVICE", "cpu")

        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")

        model = YOLO(WEIGHTS_PATH)
        model.to(DEVICE)

        print("✅ PPE model loaded successfully")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None


# --------------------------------------------------
# Health check (IMPORTANT for Render)
# --------------------------------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "PPE Detection Backend Running",
        "model_loaded": model is not None
    })


# --------------------------------------------------
# Prediction API (image upload)
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    global model

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image_path = "temp.jpg"
    image_file.save(image_path)

    try:
        results = model(image_path)

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0])
                })

        os.remove(image_path)

        return jsonify({
            "detections": detections,
            "count": len(detections)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# REQUIRED for Render (DO NOT hardcode port)
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
