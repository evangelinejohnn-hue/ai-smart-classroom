import streamlit as st
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt
from deepface import DeepFace

from ai_engine import predict
from db import init, insert, fetch_all

# OPTIONAL MODULES (safe import)
try:
    from face_attendance import recognize
except:
    recognize = None

try:
    from ml_predictor import predict_marks
except:
    predict_marks = None

# ---------------- INIT ----------------
init()

st.set_page_config(page_title="AI Smart School SaaS", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>
.title {
    font-size: 38px;
    font-weight: bold;
    color: #00ffcc;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🏫 AI Smart School SaaS Platform</div>', unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "user" not in st.session_state:
    st.session_state.user = "student_001"

if "run" not in st.session_state:
    st.session_state.run = False

# ---------------- MENU ----------------
menu = st.sidebar.radio(
    "Navigation",
    ["Live CCTV", "Dashboard", "Teacher Panel", "AI Chatbot"]
)

# =====================================================
# 🎥 LIVE CCTV (FULL SYSTEM)
# =====================================================
if menu == "Live CCTV":

    st.subheader("🎥 Smart AI CCTV Classroom System")

    classroom = st.selectbox("Select Classroom", ["Class A", "Class B", "Class C"])

    col1, col2 = st.columns(2)

    if col1.button("▶ Start System"):
        st.session_state.run = True

    if col2.button("⛔ Stop System"):
        st.session_state.run = False

    frame_box = st.image([])

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while st.session_state.run:

        ret, frame = cap.read()
        if not ret:
            st.error("Camera not found")
            break

        frame = cv2.resize(frame, (640, 480))

        # ---------------- FACE RECOGNITION ----------------
        if recognize:
            faces, names = recognize(frame)

            for (top, right, bottom, left), name in zip(faces, names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(frame, name, (left, top-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # ---------------- EMOTION AI ----------------
        small = cv2.resize(frame, (320, 240))

        try:
            result = DeepFace.analyze(
                small,
                actions=['emotion'],
                enforce_detection=False
            )
            emotion = result[0]['dominant_emotion']
        except:
            emotion = "neutral"

        focus, stress, score = predict(emotion)

        status = "Present" if focus > 50 else "Absent"

        insert(classroom, st.session_state.user, emotion, focus, stress, score, status)

        # ---------------- OVERLAYS ----------------
        cv2.putText(frame, f"{classroom}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.putText(frame, f"Emotion: {emotion}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"Focus: {focus}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.putText(frame, f"Stress: {stress}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # ---------------- ALERTS ----------------
        if stress > 80:
            cv2.putText(frame, "HIGH STRESS ALERT", (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        if focus < 40:
            cv2.putText(frame, "LOW FOCUS WARNING", (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)

        frame_box.image(frame, channels="BGR")

        time.sleep(0.03)

    cap.release()

# =====================================================
# 📊 DASHBOARD (CLOUD STYLE)
# =====================================================
elif menu == "Dashboard":

    st.subheader("📊 Multi-Classroom Analytics")

    data = fetch_all()

    if data:
        df = pd.DataFrame(data,
            columns=["id","classroom","user","emotion","focus","stress","score","status"]
        )

        selected_class = st.selectbox("Select Classroom", df["classroom"].unique())

        class_df = df[df["classroom"] == selected_class]

        st.dataframe(class_df)

        fig, ax = plt.subplots()
        ax.plot(class_df["focus"], label="Focus")
        ax.plot(class_df["stress"], label="Stress")
        ax.plot(class_df["score"], label="Score")
        ax.legend()

        st.pyplot(fig)

        # ---------------- AI PREDICTION ----------------
        if predict_marks:
            avg_focus = class_df["focus"].mean()
            avg_stress = class_df["stress"].mean()
            avg_score = class_df["score"].mean()

            result = predict_marks(avg_focus, avg_stress, avg_score)

            st.success(f"📈 Class Performance Prediction: {result}")

# =====================================================
# 👩‍🏫 TEACHER PANEL
# =====================================================
elif menu == "Teacher Panel":

    st.subheader("👩‍🏫 Teacher Control Panel")

    data = fetch_all()

    df = pd.DataFrame(data,
        columns=["id","classroom","user","emotion","focus","stress","score","status"]
    )

    classroom = st.selectbox("Select Classroom", df["classroom"].unique())

    class_df = df[df["classroom"] == classroom]

    student = st.selectbox("Select Student", class_df["user"].unique())

    student_df = class_df[class_df["user"] == student]

    st.dataframe(student_df)

# =====================================================
# 🤖 CHATBOT
# =====================================================
elif menu == "AI Chatbot":

    st.subheader("🤖 AI Study Assistant")

    msg = st.text_input("Ask anything")

    if msg:
        msg = msg.lower()

        if "focus" in msg:
            reply = "Remove distractions and use Pomodoro technique."
        elif "stress" in msg:
            reply = "Take breaks and relax your mind."
        elif "study" in msg:
            reply = "Study in 25-min focused sessions."
        else:
            reply = "Ask about focus, stress, study tips."

        st.success(reply)