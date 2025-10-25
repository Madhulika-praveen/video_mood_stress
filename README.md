**🎥 Facial Emotion & Stress Detection App**
This project is a Streamlit-based web application that detects human emotions in real-time using your webcam feed.
It uses a Facial Emotion Recognition (FER) model to analyze facial expressions and estimate stress levels, providing live visualization of stress over time.

**Features**
🧠 Real-time emotion detection using webcam input
📊 Live stress-level chart updated dynamically
⚠️ “Calm Down!” alert when stress exceeds threshold
💡 User permission prompt before accessing camera
🎨 Custom UI with a modern ribbon and adjustable layout

**🏗️ Project Structure**
video_mood_stress/
├── Home.py               # Main Streamlit app (homepage)
├── pages/
│   └── Emotion Detector.py      # Emotion detector page
├── requirements.txt      # Dependencies list
└── README.md             # Project documentation

**🚀 Installation & Setup**
1. Clone this repository
git clone https://github.com/Madhulika-praveen/video_mood_stress.git
cd video_mood_stress

2. Create a virtual environment (optional but recommended)
python -m venv mood_env
mood_env\Scripts\activate   # On Windows
# or
source mood_env/bin/activate  # On macOS/Linux

3. Install dependencies
pip install -r requirements.txt

4. Run the Streamlit app
streamlit run Home.py

**📈 How It Works**
The app uses your webcam to capture video frames in real time.

The FER library (Facial Emotion Recognition) detects emotions like happy, sad, angry, scared, disgust, neutral, etc.

The app calculates a stress score based on the intensity of negative emotions.

If the stress value exceeds 25, a “Calm down!” message appears.

A live chart plots your stress level trend.

**⚙️ Dependencies**
Key packages:

streamlit
opencv-python
fer
tensorflow
mtcnn
numpy
matplotlib

See requirements.txt for exact versions.

🧩 Future Enhancements

🎤 Audio-based emotion detection

💾 Save emotion logs over time

📱 Deploy to Streamlit Cloud or Hugging Face Spaces

🧘 Integration with relaxation or meditation suggestions


**🧑‍💻 Author**
Madhulika Praveen
Lahari BK
BMS College of Engineering
AI & ML Department