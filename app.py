from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import base64
import io
from PIL import Image

app = Flask(__name__)

SCREEN_W, SCREEN_H = pyautogui.size()
print(f"Screen Size: {SCREEN_W} x {SCREEN_H}")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

CLICK_THRESHOLD = 30
SMOOTHING = 0.5
last_click_time = 0
running = True

def get_finger_positions(hand_landmarks, frame_shape):
    h, w, _ = frame_shape
    landmarks = {}
    
    finger_tips = {
        'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20
    }
    
    finger_pips = {
        'index': 6, 'middle': 10, 'ring': 14, 'pinky': 18
    }
    
    for name, idx in finger_tips.items():
        lm = hand_landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        landmarks[name] = (x, y)
    
    for name, idx in finger_pips.items():
        lm = hand_landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        landmarks[f'{name}_pip'] = (x, y)
    
    return landmarks

def is_fist(landmarks):
    fingers = ['index', 'middle', 'ring', 'pinky']
    fist_count = 0
    
    for finger in fingers:
        tip_y = landmarks[finger][1]
        pip_y = landmarks[f'{finger}_pip'][1]
        if tip_y > pip_y:
            fist_count += 1
    
    return fist_count >= 3

def is_peace_sign(landmarks):
    index_up = landmarks['index'][1] < landmarks['index_pip'][1]
    middle_up = landmarks['middle'][1] < landmarks['middle_pip'][1]
    ring_down = landmarks['ring'][1] > landmarks['ring_pip'][1]
    pinky_down = landmarks['pinky'][1] > landmarks['pinky_pip'][1]
    
    return index_up and middle_up and ring_down and pinky_down

def generate_frames():
    global last_click_time
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    smooth_x, smooth_y = 0, 0
    
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = get_finger_positions(hand_landmarks, frame.shape)
                
                index_x, index_y = landmarks['index']
                middle_x, middle_y = landmarks['middle']
                
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)
                
                if is_fist(landmarks):
                    cv2.putText(frame, "LOCKED", (w//2 - 40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    continue
                
                if is_peace_sign(landmarks):
                    current_time = time.time()
                    if current_time - last_click_time > 0.5:
                        pyautogui.doubleClick()
                        last_click_time = current_time
                        cv2.putText(frame, "DOUBLE CLICK!", (w//2 - 60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    continue
                
                screen_x = np.interp(index_x, [0, w], [0, SCREEN_W])
                screen_y = np.interp(index_y, [0, h], [0, SCREEN_H])
                
                smooth_x = smooth_x * SMOOTHING + screen_x * (1 - SMOOTHING)
                smooth_y = smooth_y * SMOOTHING + screen_y * (1 - SMOOTHING)
                
                pyautogui.moveTo(smooth_x, smooth_y, duration=0.02)
                
                distance = np.sqrt((index_x - middle_x)**2 + (index_y - middle_y)**2)
                cv2.line(frame, (index_x, index_y), (middle_x, middle_y), (255, 0, 0), 2)
                
                if distance < CLICK_THRESHOLD:
                    current_time = time.time()
                    if current_time - last_click_time > 0.3:
                        pyautogui.click()
                        last_click_time = current_time
                        cv2.putText(frame, "CLICK!", (w//2 - 40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global last_click_time
    
    data = request.json
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    h, w, _ = image.shape
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    gesture = "None"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = get_finger_positions(hand_landmarks, image.shape)
            
            index_x, index_y = landmarks['index']
            middle_x, middle_y = landmarks['middle']
            
            if is_fist(landmarks):
                gesture = "Fist (Locked)"
            elif is_peace_sign(landmarks):
                gesture = "Peace (Double Click)"
                current_time = time.time()
                if current_time - last_click_time > 0.5:
                    pyautogui.doubleClick()
                    last_click_time = current_time
            else:
                screen_x = np.interp(index_x, [0, w], [0, SCREEN_W])
                screen_y = np.interp(index_y, [0, h], [0, SCREEN_H])
                pyautogui.moveTo(screen_x, screen_y, duration=0.02)
                gesture = "Moving"
                
                distance = np.sqrt((index_x - middle_x)**2 + (index_y - middle_y)**2)
                if distance < CLICK_THRESHOLD:
                    current_time = time.time()
                    if current_time - last_click_time > 0.3:
                        pyautogui.click()
                        last_click_time = current_time
                        gesture = "Click"
    
    return jsonify({"gesture": gesture})

@app.route('/stop')
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})

@app.route('/start')
def start():
    global running
    running = True
    return jsonify({"status": "started"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)