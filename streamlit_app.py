import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="Fatigue Detection", layout="centered")

# 🎨 UI
st.markdown("## 🚗 AI Fatigue Detection System")
st.caption("Detecting drowsiness using Computer Vision & AI")

# Sidebar mode selection
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Live Camera (Local)", "Upload Video (Cloud)"]
)

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

LEFT_EYE = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, eye):
    p1 = landmarks[eye[1]]
    p2 = landmarks[eye[5]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[4]]
    p5 = landmarks[eye[0]]
    p6 = landmarks[eye[3]]

    vertical = np.linalg.norm(p2 - p4) + np.linalg.norm(p3 - p6)
    horizontal = np.linalg.norm(p1 - p5)

    return vertical / (2.0 * horizontal)


FRAME_WINDOW = st.image([])
counter = 0
threshold = 0.25


# =========================
# 🎥 MODE 1: LIVE CAMERA
# =========================
if mode == "Live Camera (Local)":
    st.warning("⚠️ Live camera works only on your laptop (not on Streamlit Cloud)")

    run = st.checkbox("Start Camera")

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not working")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    landmarks = []

                    for lm in face.landmark:
                        landmarks.append(np.array([lm.x * w, lm.y * h]))

                    ear = eye_aspect_ratio(landmarks, LEFT_EYE)

                    if ear < threshold:
                        counter += 1
                    else:
                        counter = 0

                    if counter > 15:
                        cv2.putText(frame, "DROWSY ALERT!", (50,100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                        st.error("⚠️ DROWSINESS DETECTED!")

                    st.write(f"EAR: {round(ear, 2)}")

            FRAME_WINDOW.image(frame, channels="BGR")

        cap.release()


# =========================
# 📂 MODE 2: VIDEO UPLOAD
# =========================
elif mode == "Upload Video (Cloud)":

    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

    if uploaded_file:
        tfile = open("temp.mp4", "wb")
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture("temp.mp4")

        st.success("Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    landmarks = []

                    for lm in face.landmark:
                        landmarks.append(np.array([lm.x * w, lm.y * h]))

                    ear = eye_aspect_ratio(landmarks, LEFT_EYE)

                    if ear < threshold:
                        counter += 1
                    else:
                        counter = 0

                    if counter > 15:
                        cv2.putText(frame, "DROWSY ALERT!", (50,100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            FRAME_WINDOW.image(frame, channels="BGR")

        cap.release()
        st.success("Video processing complete!")