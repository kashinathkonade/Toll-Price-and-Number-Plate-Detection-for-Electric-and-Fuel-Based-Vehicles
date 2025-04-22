
# Toll Price and Number Plate Detection for Electric and Fuel-Based Vehicles 🚗⚡⛽

## 📌 Project Overview

This project aims to build an intelligent toll collection system that uses automatic number plate recognition (ANPR) to identify vehicle types—Electric or Fuel-based—and calculates the toll amount accordingly. Electric vehicles are charged a different toll rate than fuel-based vehicles to promote the use of clean energy transportation.

---

## 🎯 Objective

To develop an AI-driven system that:
- Detects the number plate of a vehicle using image processing and OCR.
- Classifies the vehicle type as Electric or Fuel-based.
- Calculates and displays the toll price based on the vehicle category.

---

## 🔧 Technologies Used

- **Python** 
- **OpenCV** for image processing
- **YOLO / PaddleOCR** for number plate detection
- **Flask** or **Streamlit** for web interface (optional)
- **SQLite** or **Pandas** for storing vehicle data
- **Machine Learning / Rule-based logic** for classification

---

## 📸 How It Works

1. **Capture Image or Video** of the vehicle entering the toll booth.
2. **Detect and Extract Number Plate** using YOLO or PaddleOCR.
3. **Compare the number plate** with the database to determine the vehicle type.
4. **Calculate the toll price**:
   - If **Electric**: Apply reduced toll rate (e.g., ₹30)
   - If **Fuel-Based**: Apply standard toll rate (e.g., ₹60)
5. **Display or Store** the toll charge and vehicle details for records.

---

## 🧪 Sample Output

Vehicle Number: MH12AB1234
Vehicle Type: Electric
Toll Price: ₹30

yaml
Copy
Edit

---

## 💡 Key Features

- Real-time detection and classification.
- Environmentally aware pricing model.
- Scalable and can be integrated with smart city infrastructure.
- Could integrate with FASTag/RFID in the future.

---

## 📂 Folder Structure

Toll-Detection-System/ │ ├── data/ # Vehicle info (CSV/DB) ├── images/ # Sample vehicle images ├── src/ │ ├── detection.py # Number plate detection logic │ ├── classification.py # Electric vs Fuel logic │ └── app.py # Main app (Flask/Streamlit) ├── README.md └── requirements.txt

yaml
Copy
Edit

---

## ✅ Future Improvements

- Integrate with a cloud database for larger deployments.
- Add voice assistant or display panel at toll gates.
- Include vehicle type detection using ML and visual cues.
- Enable SMS notifications and online payment integration.

---

## 🙏 Message

This project is dedicated to fostering smart transportation and clean energy awareness. By identifying the type of vehicle at toll booths and assigning toll charges accordingly, we aim to promote electric vehicles while ensuring fair toll operations.

Let’s build smarter roads for a sustainable future. 🌱
