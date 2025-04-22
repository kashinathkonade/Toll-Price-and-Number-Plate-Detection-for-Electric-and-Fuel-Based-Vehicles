# vehicle_detection_flask/app.py

from flask import Flask, render_template, request, redirect, url_for, Response, send_file
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import base64
import tempfile
from io import BytesIO

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("/home/user3/EVs and Fuel Vehicles Detection Project /New_Project/Model yolov8x/detect/train/weights/best.pt")
EXPECTED_CLASSES = ['Green', 'White', 'Yellow']

TOLL_PRICES = {
    "Green": 30,
    "Yellow": 60,
    "White": 50
}

VEHICLE_TYPE_MAPPING = {
    "Green": "Electric Vehicle",
    "White": "Fuel Vehicle (Private)",
    "Yellow": "Fuel Vehicle (Commercial)"
}

# Function to perform detection
def detect_objects(image):
    results = model(image, conf=0.25)
    detected_data = []
    unique_detections = set()

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, class_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[class_id]
            vehicle_type = VEHICLE_TYPE_MAPPING.get(class_name, "Unknown")
            toll_price = TOLL_PRICES.get(class_name, "N/A")

            if vehicle_type != "Unknown" and vehicle_type not in unique_detections:
                unique_detections.add(vehicle_type)

            detected_data.append([vehicle_type, class_name, toll_price])

            color = (0, 255, 0) if class_name == "Green" else (255, 255, 255) if class_name == "White" else (0, 255, 255)
            cv2.rectangle(image, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), color, 4)
            cv2.putText(image, class_name, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 4)

    return image, detected_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image', methods=['POST'])
def detect_image():
    file = request.files.get('image')
    if not file or file.filename == '':
        return redirect(url_for('index'))

    image = Image.open(file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    processed_image, detected_data = detect_objects(image_bgr)

    img_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    df = pd.DataFrame(detected_data, columns=["Vehicle Type", "Class", "Toll Price"])
    table_html = df.to_html(index=False, classes='table table-bordered table-striped text-center')

    return render_template('results_image.html', image_data=image_base64, table_html=table_html)

@app.route('/video', methods=['POST'])
def detect_video():
    file = request.files.get('video')
    if not file or file.filename == '':
        return redirect(url_for('index'))

    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    tfile.close()

    # OpenCV video capture
    cap = cv2.VideoCapture(tfile.name)
    output_path = os.path.join('static', 'processed_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    detected_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame, detections = detect_objects(frame)
        detected_data.extend(detections)
        out.write(frame)  # âœ… Save processed frame
    cap.release()
    out.release()
    os.unlink(tfile.name)

    df = pd.DataFrame(detected_data, columns=["Vehicle Type", "Class", "Toll Price"])
    table_html = df.to_html(index=False, classes='table table-bordered table-striped text-center')

    return render_template('results_video.html', video_path='processed_video.mp4', table_html=table_html)

# Live Camera Streaming
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame, _ = detect_objects(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/live')
def detect_live():
    return render_template('live_camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

