import cv2
import mediapipe as mp
import numpy as np

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

def detect_fatigue(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = []

            for lm in face.landmark:
                landmarks.append(np.array([lm.x * w, lm.y * h]))

            ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            return ear, frame

    return None, frame