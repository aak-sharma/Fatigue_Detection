from flask import Flask, render_template, Response
import cv2
import winsound
from fatigue import detect_fatigue

app = Flask(__name__)
cap = cv2.VideoCapture(0)

counter = 0
threshold = 0.25

def play_alarm():
    winsound.Beep(1000, 500)

def generate_frames():
    global counter

    while True:
        success, frame = cap.read()
        if not success:
            break

        ear, frame = detect_fatigue(frame)

        if ear:
            if ear < threshold:
                counter += 1
            else:
                counter = 0

            if counter > 15:
                cv2.putText(frame, "DROWSY ALERT!", (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                play_alarm()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)