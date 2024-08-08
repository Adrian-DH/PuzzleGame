from flask import Flask, render_template, Response, stream_with_context, jsonify
import cv2
import mediapipe as mp
import time
import threading

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

direction_event = threading.Event()
current_direction = "none"

camera = None
camera_lock = threading.Lock()

def detect_pointing_direction(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    dx = index_tip.x - wrist.x
    dy = index_tip.y - wrist.y
    
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "down" if dy > 0 else "up"


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
    return jsonify({"status": "Camera started"})

def gen_frames():
    global camera
    while True:
        with camera_lock:
            if camera is None:
                continue
            success, frame = camera.read()
        if not success:
            break
        else:
            camera = cv2.VideoCapture(0)
            global current_direction
            while True:
                success, frame = camera.read()
                if not success:
                    break
                else:
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            direction = detect_pointing_direction(hand_landmarks)
                            cv2.putText(frame, f"Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                            current_direction = direction
                            direction_event.set()

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/direction_events')
def direction_events():
    def generate():
        global current_direction
        while True:
            if direction_event.is_set():
                yield f"data: {current_direction}\n\n"
                direction_event.clear()
            time.sleep(0.3)
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)