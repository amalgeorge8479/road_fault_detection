import streamlit as st
import sqlite3
import pandas as pd
import os
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

model = YOLO("best.pt")  

conn = sqlite3.connect("road_faults.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS faults (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fault_type TEXT,
    confidence REAL,
    timestamp TEXT
)
""")
conn.commit()

st.set_page_config(page_title="Road Fault Detection", layout="wide")
st.markdown(
    """
    <style>
        /* Force light theme */
        .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
        }
        /* Style titles */
        h1, h2, h3, h4 {
            color: #2c3e50 !important;
        }
        /* Light theme for file uploader drop area */
        div[data-testid="stFileUploaderDropzone"] {
            background-color: #f8f9fa !important;
            border: 2px dashed #ccc !important;
            border-radius: 10px !important;
            color: #000000 !important;
        }
        /* Uploaded file name + size styling */
        div[data-testid="stFileUploaderDropzone"] * {
            color: #000000 !important;
            font-weight: 500 !important;
        }
        /* Style DataFrame headers */
        .stDataFrame th {
            background-color: #f1f1f1 !important;
            color: #000000 !important;
        }
        /* Custom alert boxes */
        .success-box {
            padding: 12px;
            background-color: #d4edda;
            border-left: 6px solid #28a745;
            border-radius: 6px;
            color: #155724;
            font-weight: bold;
        }
        .error-box {
            padding: 12px;
            background-color: #f8d7da;
            border-left: 6px solid #dc3545;
            border-radius: 6px;
            color: #721c24;
            font-weight: bold;
        }
        /* Fix for Streamlit warning text visibility */
        div[data-baseweb="notification"][class*="stAlert"] {
            background-color: #fff3cd !important;
            color: #000000 !important;
            font-weight: 600 !important;
            border: 1px solid #ffeeba !important;
            border-radius: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚧 Road Fault Detection System")

uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run YOLO detection
    results = model(image)

    # Extract detections
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            detections.append((label, conf, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    st.markdown("###  Detection Results")

    if detections:
        # Save to DB
        cursor.executemany(
            "INSERT INTO faults (fault_type, confidence, timestamp) VALUES (?, ?, ?)",
            detections
        )
        conn.commit()

        # Convert detections into DataFrame
        df = pd.DataFrame(detections, columns=["Fault Type", "Confidence", "Timestamp"])

        # Show results in table
        st.dataframe(df, use_container_width=True)

    else:
        st.warning(" No faults detected in this image.")
