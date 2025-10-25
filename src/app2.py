import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your trained CNN model
MODEL_PATH = "E:/video_mood_stress/models/emotion_model.h5"
model = load_model(MODEL_PATH)

# Update this list based on your dataset folder structure
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

st.title("🎥 Real-Time Mood & Stress Detection (Custom Model)")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Camera not found.")
        break

    # Convert to grayscale (optional — depends on how dataset was trained)
    face = cv2.resize(frame, (224, 224))  # keep 3 color channels (RGB)

    # Prepare image for prediction
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0) / 255.0

    # Predict emotion
    preds = model.predict(face)
    emotion = CLASS_NAMES[np.argmax(preds)]

    # Simple stress calculation (average of negative emotions)
    stress = int((preds[0][0] + preds[0][2] + preds[0][5]) * 100 / 3)

    # Display emotion + stress on frame
    cv2.putText(frame, f"{emotion.upper()} | Stress: {stress}%", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Convert color for Streamlit display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

# Release camera
camera.release()