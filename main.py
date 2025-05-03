import os
from flask import Flask, render_template, Response, request, jsonify,redirect
import cv2
from deepface import DeepFace

app = Flask(__name__)

def generate_frames():
    webcam = cv2.VideoCapture(0)
    while True:
        ret, frame = webcam.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception as e:
            pass
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video-chat')
def chat():
    return render_template('video-chat.html')

@app.route('/video')
def video():
    return redirect("http://localhost:5001")


if __name__ == '__main__':
    app.run(port=5000,debug=True)