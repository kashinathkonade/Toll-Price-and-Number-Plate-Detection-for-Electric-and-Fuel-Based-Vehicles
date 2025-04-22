import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import tempfile
import os
from io import BytesIO

# Load the trained YOLOv8 model
model = YOLO("C:\\Users\\kashinath konade\\Downloads\\New_Project\\Model yolov8x\\detect\\train\\weights\\best.pt")

# Print Model Class Names for Debugging
print("Model Class Names:", model.names)

# Ensure model class names match data.yaml
EXPECTED_CLASSES = ['Green', 'White', 'Yellow']
if list(model.names.values()) != EXPECTED_CLASSES:
    st.error("Model class names do not match expected classes from data.yaml. Please check your training data.")

# Define vehicle categories and toll prices
TOLL_PRICES = {
    "Green": 30,  # Electric Vehicle
    "Yellow": 60, # Fuel Vehicle (Commercial)
    "White": 50,  # Fuel Vehicle (Private)
}

VEHICLE_TYPE_MAPPING = {
    "Green": "Electric Vehicle",
    "White": "Fuel Vehicle (Private)",
    "Yellow": "Fuel Vehicle (Commercial)"
}
# ✅ Streamlit UI
st.set_page_config(layout="wide")  # ✅ Set UI to wide mode
st.markdown(
    """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <style>
        /* Title Container */
        .title-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            margin-bottom: 20px;
        }

        /* Title Box with Gradient & Shadow */
        .title-box {
            background: linear-gradient(135deg,   #6ae2b8, #51d642 100%);
            padding: 7px 20px;
            border-radius: 50px;
            box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.3);
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            color: white;
            display: inline-block;
            text-transform: uppercase;
            transition: all 0.3s ease-in-out;
            border: 2px solid #fff;
        }

        /* Hover Effect */
        .title-box:hover {
            transform: scale(1.05);
            box-shadow: 0px 10px 25px rgba(255, 255, 255, 0.5);
        }

        /* Icon Animations */
        .title-icon {
            animation: bounce 1.5s infinite alternate;
        }

        @keyframes bounce {
            from { transform: translateY(0px); }
            to { transform: translateY(-5px); }
        }
    </style>

    <div class="title-container">
        <div class="title-box">
            <span class="title-icon">🚗</span> 
            EV & FUEL VEHICLE DETECTION  
            <span class="title-icon">🚕</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
# Inject custom CSS to enhance table UI and hide stVideo element
st.markdown("""
    <style>
    [data-testid="stVideo"] {
        display: none;
    }

    /* Style output table */
    .stDataFrame {
        border-radius: 15px !important;
        border: 2px solid #000 !important;
        background-color: #fff !important;
        padding: 8px !important;
        box-shadow: 3px 3px 12px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Center table text */
    .stDataFrame th, .stDataFrame td {
        text-align: center !important;
        font-weight: normal !important;
        font-size: 18px !important;
        color: #000 !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

import base64

# Define correct image path
image_path = r"C:\\Users\\kashinath konade\\Downloads\\New_Project\\AiSPRY logo.jpg"

# Function to encode image in base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert image to base64 format
try:
    image_base64 = encode_image(image_path)
    image_src = f"data:image/png;base64,{image_base64}"
except FileNotFoundError:
    st.sidebar.error("Image not found. Check the file path.")
    image_src = None  # Prevents error if image is missing

# Custom CSS for Styling
st.markdown("""
    <style>
        /* Sidebar Image Container */
        .sidebar-image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 15px;
            background: linear-gradient(135deg,  #39dcce, #fc44da); /* Gradient Background */
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3); /* Box Shadow */
            transition: transform 0.3s ease-in-out;
        }

        /* Hover Effect */
        .sidebar-image-container:hover {
            transform: scale(1.05);
            box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.4);
        }

        /* Bounce Animation */
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }

        .sidebar-image {
            animation: bounce 2s infinite;
        }
        
    </style>
""", unsafe_allow_html=True)

# Display Sidebar Image with Animation (Only if Image is Found)
if image_src:
    st.sidebar.markdown(
        f"""
        <div class="sidebar-image-container">
            <img class="sidebar-image" src="{image_src}" width="255">
        </div>
        """, unsafe_allow_html=True
    )

# Sidebar Options
st.sidebar.markdown("### Select Input Type:")
option = st.sidebar.radio("", ["Image", "Video", "Live Camera"])

# Function to show popup notification below video/image
def show_popup(vehicle_type, toll_price):
    st.toast(f"🚗 {vehicle_type} detected! Toll Price: ₹{toll_price}")



# Function to perform detection
def detect_objects(image):
    """Processes image with YOLOv8 for object detection"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure correct color format
    results = model(image_rgb, conf=0.25)  # Increased confidence threshold for better accuracy
    detected_data = []
    unique_detections = set()
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, class_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure correct class name mapping from YOLO model
            class_name = model.names[class_id]
            vehicle_type = VEHICLE_TYPE_MAPPING.get(class_name, "Unknown")
            toll_price = TOLL_PRICES.get(class_name, "N/A")
            
            if vehicle_type != "Unknown" and vehicle_type not in unique_detections:
                unique_detections.add(vehicle_type)
                show_popup(vehicle_type, toll_price)
            
            detected_data.append([vehicle_type, class_name, toll_price])
            
            # Draw bounding box with corrected class name and wider bounding box
            color = (0, 255, 0) if class_name == "Green" else (255, 255, 255) if class_name == "White" else (0, 255, 255)
            thickness = 4  # Increased thickness for better visibility
            cv2.rectangle(image, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), color, thickness)
            cv2.putText(image, class_name, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 4)
    
    return image, detected_data

# Handling Image Upload
import io
if option == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)

            processed_image, detected_data = detect_objects(image_np)

            # Convert image to bytes for download
            img_download = Image.fromarray(processed_image)
            img_buffer = io.BytesIO()
            img_download.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            # **Center Image Using Streamlit Columns**
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(processed_image, caption="Detected Image", use_container_width=True)

            # Download button centered
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Download Processed Image",
                data=img_bytes,
                file_name="detected_image.png",
                mime="image/png",
                label = "Download Processed Image"
                
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Detection Results Table
            st.write("### Detection Results")
            df = pd.DataFrame(detected_data, columns=["Vehicle Type", "Class", "Toll Price (₹)"])
            st.table(df)  # Improved display format

        except (OSError, UnidentifiedImageError):
            st.error("Uploaded image file is corrupted or invalid. Please upload a different image.")
            
            
# Function to perform detection
def detect_objects(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure correct color format
    results = model(image_rgb, conf=0.7)  # Increased confidence threshold for better accuracy
    detected_data = []
    unique_detections = set()
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, class_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure correct class name mapping from YOLO model
            class_name = model.names[class_id]
            vehicle_type = VEHICLE_TYPE_MAPPING.get(class_name, "Unknown")
            toll_price = TOLL_PRICES.get(class_name, "N/A")
            
            if vehicle_type != "Unknown" and vehicle_type not in unique_detections:
                unique_detections.add(vehicle_type)
                show_popup(vehicle_type, toll_price)
            
            detected_data.append([vehicle_type, class_name, toll_price])
            
            # Draw bounding box with corrected class name and wider bounding box
            color = (0, 255, 0) if class_name == "Green" else (255, 255, 255) if class_name == "White" else (0, 255, 255)
            thickness = 4  # Increased thickness for better visibility
            cv2.rectangle(image, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), color, thickness)
            cv2.putText(image, class_name, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 4)
    
    return image, detected_data

# Handling Video Upload
if option == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)

        output_path = "processed_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        detected_data = []

        # **Center Video Using Columns**
        col1, col2, col3 = st.columns([1, 2, 1])  # Center alignment
        with col2:
            stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame, detections = detect_objects(frame)
            detected_data.extend(detections)

            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Show video in the center
            with col2:
                stframe.image(frame, channels="RGB", use_container_width=True)

        cap.release()
        out.release()
        os.unlink(tfile.name)

        # **Detection Results Centered**
        st.markdown("<h3 style='text-align: center;'>Detection Results</h3>", unsafe_allow_html=True)
        df = pd.DataFrame(detected_data, columns=["Vehicle Type", "Class", "Toll Price (₹)"])

        
        col1, col2, col3 = st.columns([1, 2, 1])  # Center alignment for table
        with col2:
            st.table(df)

        # **Download button centered**
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        with open(output_path, "rb") as file:
            st.download_button(
                label="📥 Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4",
            )
        st.markdown("</div>", unsafe_allow_html=True)

# Live Camera Feed
elif option == "Live Camera":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    detected_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Ensure correct color handling
        frame, detections = detect_objects(frame)
        detected_data.extend(detections)
        stframe.image(frame, channels="RGB", use_container_width=True)

    cap.release()

    st.write("### Detection Results")
    df = pd.DataFrame(detected_data, columns=["Vehicle Type", "Class", "Toll Price (₹)"])
    st.dataframe(df.style.set_properties(**{'text-align': 'center'}).set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'center')]}]
    ))
    

