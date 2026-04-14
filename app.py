import streamlit as st
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt
from deepface import DeepFace

from ai_engine import predict
from db import init, insert, fetch_all

# ---------------- INIT ----------------
init()

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Smart Classroom SaaS", layout="wide")

st.title("🏫 AI Smart Classroom SaaS (Cloud Version)")

# ---------------- SESSION ----------------
if "run" not in st.session_state:
    st.session_state.run = False

# ---------------- MENU ----------------
menu = st.sidebar.radio(
    "Navigation",
    ["Live CCTV", "Dashboard", "Teacher Panel"]
)

# =====================================================
# 🎥 LIVE CCTV (STREAMLIT SAFE)
# =====================================================
if menu == "Live CCTV":

    st.subheader("🎥 Live AI CCTV Monitoring (Cloud Safe)")

    classroom = st.selectbox("Select Classroom", ["Class A", "Class B", "Class C"])

    start = st.button("▶ Start Camera")
    stop = st.button("⛔ Stop Camera")

    if start:
        st.session_state.run = True

    if stop:
        st.session_state.run = False

    frame_box = st.image([])

    cap = cv2.VideoCapture(0)

    if st.session_state.run:

        # SAFE LOOP (NO INFINITE LOOP)
        for i in range(200):

            if not st.session_state.run:
                break

            ret, frame = cap.read()
            if not ret:
                st.error("Camera not found")
                break

            frame = cv2.resize(frame, (640, 480))

            # AI EMOTION
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

            insert(classroom, "student_001", emotion, focus, stress, score, status)

            # OVERLAY TEXT
            cv2.putText(frame, f"{classroom}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            cv2.putText(frame, f"Emotion: {emotion}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.putText(frame, f"Focus: {focus}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            cv2.putText(frame, f"Stress: {stress}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            frame_box.image(frame, channels="BGR")

            time.sleep(0.05)

    cap.release()

# =====================================================
# 📊 DASHBOARD
# =====================================================
elif menu == "Dashboard":

    st.subheader("📊 Analytics Dashboard")

    data = fetch_all()

    if data:
        df = pd.DataFrame(data,
            columns=["id","classroom","user","emotion","focus","stress","score","status"]
        )

        st.dataframe(df)

        fig, ax = plt.subplots()
        ax.plot(df["focus"], label="Focus")
        ax.plot(df["stress"], label="Stress")
        ax.plot(df["score"], label="Score")
        ax.legend()

        st.pyplot(fig)

# =====================================================
# 👩‍🏫 TEACHER PANEL
# =====================================================
elif menu == "Teacher Panel":

    st.subheader("👩‍🏫 Teacher Panel")

    data = fetch_all()

    df = pd.DataFrame(data,
        columns=["id","classroom","user","emotion","focus","stress","score","status"]
    )

    classroom = st.selectbox("Select Classroom", df["classroom"].unique())

    class_df = df[df["classroom"] == classroom]

    student = st.selectbox("Select Student", class_df["user"].unique())

    student_df = class_df[class_df["user"] == student]

    st.dataframe(student_df)