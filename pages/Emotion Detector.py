import streamlit as st
import cv2
from fer import FER
import time
from collections import Counter
import pandas as pd

st.set_page_config(page_title="Emotion Detector", layout="wide")

# ----------------- EMOJI MAPPING -----------------
emoji_map = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😱",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

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

/* Summary Panel */
.summary {
    background-color: #e6f7ff;
    border: 2px solid #0099cc;
    border-radius: 12px;
    padding: 15px;
    margin-top: 20px;
    font-size: 22px;
    color: #002b36;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------- RIBBON -----------------
st.markdown('<div class="ribbon">Emotion Detector</div>', unsafe_allow_html=True)

# ----------------- INITIALIZE STATE -----------------
for key in ["stress_values", "emotion_log", "start_time", "duration", "avg_stress",
            "dominant_emotion", "user_feedback", "feedback_submitted"]:
    if key not in st.session_state:
        if key in ["stress_values", "emotion_log"]:
            st.session_state[key] = []
        elif key in ["duration", "avg_stress"]:
            st.session_state[key] = 0
        elif key == "user_feedback":
            st.session_state[key] = None
        elif key == "feedback_submitted":
            st.session_state[key] = False
        else:
            st.session_state[key] = None

st.session_state.dominant_emotion = "N/A"

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

# ----------------- RESET FEEDBACK WHEN CAMERA STARTS -----------------
if consent and run:
    st.session_state.user_feedback = None
    st.session_state.feedback_submitted = False

# ----------------- CAMERA LOGIC -----------------
if consent and run:
    detector = FER(mtcnn=False)
    camera = cv2.VideoCapture(0)

    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    stress_values = []
    timestamps = []

    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Camera not found.")
            break

        frame_small = cv2.resize(frame, (640, 480))
        result = detector.detect_emotions(frame_small)
        if result:
            emotions = result[0]["emotions"]
            dominant = max(emotions, key=emotions.get).lower()

            stress = (emotions.get("angry", 0) + emotions.get("sad", 0) + emotions.get("fear", 0)) / 3 * 100
            stress = 0 if stress < 1 else round(stress, 2)

            stress_values.append(stress)
            timestamps.append(time.time())
            st.session_state.stress_values.append(stress)
            st.session_state.emotion_log.append(dominant)

            # Keep last 30 seconds for chart
            start_window = timestamps[-1] - 30
            while timestamps and timestamps[0] < start_window:
                timestamps.pop(0)
                stress_values.pop(0)

            chart.line_chart(stress_values)

            # Alert if stress > 25
            if stress > 25:
                ALERT_BOX.markdown('<div class="alert">😟 Calm down! Take a deep breath.</div>', unsafe_allow_html=True)
            else:
                ALERT_BOX.empty()

            cv2.putText(frame, f"{dominant.upper()} | Stress: {stress}%", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    camera.release()

# ----------------- AFTER CAMERA STOP -----------------
if consent and not run:
    # --- SESSION SUMMARY PANEL ---
    if st.session_state.stress_values or st.session_state.emotion_log:
        st.session_state.duration = int(time.time() - st.session_state.start_time) if st.session_state.start_time else st.session_state.duration
        st.session_state.start_time = None

        if st.session_state.stress_values:
            st.session_state.avg_stress = round(sum(st.session_state.stress_values) / len(st.session_state.stress_values), 2)

        if st.session_state.emotion_log:
            emotion_counts = Counter(st.session_state.emotion_log)
            dominant_emotion_name = emotion_counts.most_common(1)[0][0].capitalize()
            dominant_emotion_emoji = emoji_map.get(dominant_emotion_name.lower(), "😐")
            st.session_state.dominant_emotion = f"{dominant_emotion_emoji} {dominant_emotion_name}"

        st.markdown(f"""
        <div class="summary">
        <h3>🧾 Session Summary</h3>
        <p><b>Average Stress Level:</b> {st.session_state.avg_stress}%</p>
        <p><b>Dominant Emotion Detected:</b> {st.session_state.dominant_emotion}</p>
        <p><b>Duration of Monitoring:</b> {st.session_state.duration} seconds</p>
        </div>
        """, unsafe_allow_html=True)

        # ----------------- EMOTION DISTRIBUTION -----------------
        if st.session_state.emotion_log:
            emotion_counts = Counter(st.session_state.emotion_log)
            total = sum(emotion_counts.values())
            emotion_percentages = {k.capitalize(): round(v / total * 100, 2) for k, v in emotion_counts.items()}
            df_emotions = pd.DataFrame.from_dict(emotion_percentages, orient='index', columns=['Percentage'])
            df_emotions = df_emotions.sort_values(by='Percentage', ascending=False)

            st.markdown("<h3 style='text-align: center; margin-top: 20px;'>📊 Emotion Distribution</h3>", unsafe_allow_html=True)
            st.bar_chart(df_emotions)

    # ----------------- USER FEEDBACK -----------------
    st.markdown("<h3 style='text-align: center; margin-top: 20px;'>📝 Feedback</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; font-size: 22px; color: #333;'>
    Please select your current emotion(s) after this session.
    </p>
    """, unsafe_allow_html=True)

    if not st.session_state.feedback_submitted:
        user_emotions = st.multiselect(
            "Your Emotion(s):",
            ["Happy", "Sad", "Angry", "Surprised", "Fear", "Disgust", "Neutral"]
        )
        if st.button("Submit Feedback") and user_emotions:
            st.session_state.user_feedback = user_emotions
            st.session_state.feedback_submitted = True

    if st.session_state.feedback_submitted:
        selected_emotions = ", ".join(st.session_state.user_feedback)
        st.markdown(f"""
        <p style='text-align: center; font-size: 22px; color: #007700; margin-top: 15px;'>
        🙏 Thank you for your feedback!</b></p>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <p style='text-align: center; font-size: 20px; color: #333; margin-top: 10px;'>
        Click "Start Camera" above to begin a new session.
        </p>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
