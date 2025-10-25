import streamlit as st

st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")

# --- CSS ---
CUSTOM_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}

/* Ribbon styling */
.ribbon {
    width: 100%;
    background-color: #404040;
    padding: 14px 20px;
    display: flex;
    align-items: baseline;
    justify-content: flex-start;
    color: white;
    font-size:55px;
    font-weight: bold;
}

/* Page top margin */
.page-content {margin-top: 60px;}

/* Increase font size for body text and headers */
h1 { font-size: 38px; }
h2 { font-size: 30px; } 
h3 { font-size: 26px; }
p, span { font-size: 26px; }

/* Increase sidebar label font size */
[data-testid="stSidebar"] label {
    font-size: 30px !important;
    font-weight: bold;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------- RIBBON -----------------
st.markdown('<div class="ribbon">Facial Emotion Recognition</div>', unsafe_allow_html=True)

# ----------------- PAGE CONTENT -----------------
st.markdown('<div class="page-content">', unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center; font-size: 18px;'>
We offer <b>accurate</b> and <b>automated</b> system for identifying human emotions 
from <b>facial expressions</b>, in images, videos, or live-streams.<br>
It detects universal emotions like <b>happy, sad, angry, surprised, fear, disgust,</b> and <b>neutral</b> using Artificial Intelligence.
</p>
""", unsafe_allow_html=True)

# Rainbow horizontal line
st.markdown("""
<hr style="
    border: 0;
    height: 3px;
    background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
    border-radius: 1px;
    margin-top: 2rem;
    margin-bottom: 2rem;
">
""", unsafe_allow_html=True)

# Example sections
st.header("How It Works")
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    st.image(
        "images/feature.png",
        caption="Advanced AI Analysis"
    )
with col2:
    st.subheader("Real-Time Processing")
    st.write("""
    Our system processes video feeds in real-time to detect subtle micro-expressions. 
    Immediate feedback helps in applications like mental health monitoring and customer service.
    """)
    st.subheader("High Accuracy")
    st.write("""
    Our engine achieves over 70% accuracy in identifying the seven core universal emotions. 
    It is trained on diverse datasets for fairness and reliability.
    """)

# Rainbow horizontal line
st.markdown("""
<hr style="
    border: 0;
    height: 3px;
    background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
    border-radius: 1px;
    margin-top: 2rem;
    margin-bottom: 2rem;
">
""", unsafe_allow_html=True)

st.header("Emotion Spectrum")
#st.write("<p style='text-align: center; font-size: 26px;'>Our model detects a wide range of emotions:</p>", unsafe_allow_html=True)

# First row: Happy, Sad, Surprised
img_col1, img_col2, img_col3 = st.columns(3)
with img_col1: st.image("images/happy.png", caption="Happy")
with img_col2: st.image("images/sad.png", caption="Sad")
with img_col3: st.image("images/surprise.png", caption="Surprised")

# Second row: Angry, Scared, Disgust
img_col4, img_col5, img_col6 = st.columns(3)
with img_col4: st.image("images/angry.png", caption="Angry")
with img_col5: st.image("images/scared.png", caption="Fear")
with img_col6: st.image("images/disgust.png", caption="Disgust")

# Third row: Neutral
img_col7, _, _ = st.columns(3)  # center Neutral image
with img_col7: st.image("images/neutral.png", caption="Neutral")


st.markdown("<div style='height: 300px;'></div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
