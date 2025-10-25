import streamlit as st
import cv2
from fer import FER
import time

st.set_page_config(page_title="Emotion Detector", layout="wide")

# --- CSS ---
CUSTOM_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.page-content {margin-top: 60px;}

/* Ribbon styling */
.ribbon {
    width: 100%;
    background-color: #404040;
    padding: 14px 20px;
    display: flex;
    align-items: baseline;
    justify-content: flex-start;
    color: white;
    font-size: 55px;
    font-weight: bold;
}

h1 { font-size: 38px; }
h2 { font-size: 30px; }
h3 { font-size: 26px; }
p, span { font-size: 26px; }

/* Calm down alert */
.alert {
    background-color: #ffcccc;
    color: #b30000;
    font-weight: bold;
    text-align: center;
    border-radius: 10px;
    padding: 10px;
    margin-top: 10px;
    font-size: 24px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------- RIBBON -----------------
st.markdown('<div class="ribbon">Emotion Detector</div>', unsafe_allow_html=True)

# ----------------- PAGE CONTENT -----------------
st.markdown('<div class="page-content">', unsafe_allow_html=True)
st.title("🎥 Real-Time Mood & Stress Detection with Live Graph")

col1, col2 = st.columns([1, 1])

with col1:
    consent = st.checkbox("I allow the app to access my camera")
    if consent:
        run = st.checkbox("Start Camera")
        FRAME_WINDOW = st.empty()
        ALERT_BOX = st.empty()
    else:
        st.warning("Camera access is required to use real-time mood & stress detection.")
        run = False
        FRAME_WINDOW = None
        ALERT_BOX = None

with col2:
    st.markdown("#### 📈 Stress Level Over Time")
    st.markdown("<div style='margin-top: 75px;'></div>", unsafe_allow_html=True)
    chart = st.line_chart([])

if consent and run:
    detector = FER(mtcnn=False)
    camera = cv2.VideoCapture(0)

    stress_values = []
    timestamps = []

    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Camera not found.")
            break

        result = detector.detect_emotions(frame)
        if result:
            emotions = result[0]["emotions"]
            dominant = max(emotions, key=emotions.get)
            stress = int((emotions.get("angry", 0) + emotions.get("sad", 0) + emotions.get("fear", 0)) * 100 / 3)

            stress_values.append(stress)
            timestamps.append(time.time())

            # Keep last 30 seconds
            start_time = timestamps[-1] - 30
            while timestamps and timestamps[0] < start_time:
                timestamps.pop(0)
                stress_values.pop(0)

            chart.line_chart(stress_values)

            # Display alert if stress > 25
            if stress > 25:
                ALERT_BOX.markdown('<div class="alert">😟 Calm down! Take a deep breath.</div>', unsafe_allow_html=True)
            else:
                ALERT_BOX.empty()

            cv2.putText(frame, f"{dominant.upper()} | Stress: {stress}%", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    camera.release()

st.markdown('</div>', unsafe_allow_html=True)
