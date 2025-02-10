import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, Response

app = Flask(__name__)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Colors and Opacity for Makeup
LIP_COLOR = (0, 0, 255)  # Red for lips
EYELINER_COLOR = (0, 0, 0)  # Black for eyeliner
MAKEUP_OPACITY = 0.7

# Function to overlay makeup on a region
def apply_makeup(image, points, color, opacity):
    overlay = image.copy()
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(overlay, [points], color)
    return cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)

# Function to draw winged eyeliner
def apply_winged_eyeliner(image, eye_points, wing_tip, color, thickness):
    for i in range(len(eye_points) - 1):
        cv2.line(image, eye_points[i], eye_points[i + 1], color, thickness)
    # Add wing
    cv2.line(image, eye_points[-1], wing_tip, color, thickness)
    return image

# Function to draw landmarks and apply makeup
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            height, width, _ = frame.shape

            # Get landmarks
            landmarks = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks.landmark]

            # Define regions for lips and eyeliner
            lips_points = [landmarks[i] for i in range(61, 81)]  # Lips points correctly defined
            left_eye_top = [landmarks[i] for i in [159, 158, 157, 173]]
            left_eye_bottom = [landmarks[i] for i in [144, 145, 153, 154]][::-1]
            right_eye_top = [landmarks[i] for i in [386, 385, 384, 398]]
            right_eye_bottom = [landmarks[i] for i in [373, 374, 380, 381]][::-1]

            # Calculate wing tips for eyeliner
            left_wing_tip = (left_eye_top[-1][0] - 15, left_eye_top[-1][1] - 10)
            right_wing_tip = (right_eye_top[0][0] + 15, right_eye_top[0][1] - 10)

            # Apply lipstick ONLY on lips (correct region)
            frame = apply_makeup(frame, lips_points, LIP_COLOR, MAKEUP_OPACITY)

            # Apply winged eyeliner (no changes here)
            frame = apply_winged_eyeliner(frame, left_eye_top + left_eye_bottom, left_wing_tip, EYELINER_COLOR, 2)
            frame = apply_winged_eyeliner(frame, right_eye_top + right_eye_bottom, right_wing_tip, EYELINER_COLOR, 2)

    return frame

# Video capture and generator function for Flask
def generate_frames():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Error: Cannot open webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame for a better user experience
        frame = process_frame(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return "<h1>Glam AI - Virtual Makeup</h1><img src='/video_feed' style='width: 100%;'>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, Response

app = Flask(__name__)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Colors and Opacity for Makeup
LIP_COLOR = (0, 0, 255)  # Red for lips
EYELINER_COLOR = (0, 0, 0)  # Black for eyeliner
MAKEUP_OPACITY = 0.7

# Function to overlay makeup on a region
def apply_makeup(image, points, color, opacity):
    overlay = image.copy()
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(overlay, [points], color)
    return cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)

# Function to draw winged eyeliner
def apply_winged_eyeliner(image, eye_points, wing_tip, color, thickness):
    for i in range(len(eye_points) - 1):
        cv2.line(image, eye_points[i], eye_points[i + 1], color, thickness)
    # Add wing
    cv2.line(image, eye_points[-1], wing_tip, color, thickness)
    return image

# Function to draw landmarks and apply makeup
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            height, width, _ = frame.shape

            # Get landmarks
            landmarks = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks.landmark]

            # Define regions for lips and eyeliner
            lips_points = [landmarks[i] for i in range(61, 81)]  # Lips points correctly defined
            left_eye_top = [landmarks[i] for i in [159, 158, 157, 173]]
            left_eye_bottom = [landmarks[i] for i in [144, 145, 153, 154]][::-1]
            right_eye_top = [landmarks[i] for i in [386, 385, 384, 398]]
            right_eye_bottom = [landmarks[i] for i in [373, 374, 380, 381]][::-1]

            # Calculate wing tips for eyeliner
            left_wing_tip = (left_eye_top[-1][0] - 15, left_eye_top[-1][1] - 10)
            right_wing_tip = (right_eye_top[0][0] + 15, right_eye_top[0][1] - 10)

            # Apply lipstick ONLY on lips (no makeup on cheeks)
            frame = apply_makeup(frame, lips_points, LIP_COLOR, MAKEUP_OPACITY)

            # Apply winged eyeliner (no changes here)
            frame = apply_winged_eyeliner(frame, left_eye_top + left_eye_bottom, left_wing_tip, EYELINER_COLOR, 2)
            frame = apply_winged_eyeliner(frame, right_eye_top + right_eye_bottom, right_wing_tip, EYELINER_COLOR, 2)

    return frame

# Video capture and generator function for Flask
def generate_frames():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Error: Cannot open webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame for a better user experience
        frame = process_frame(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return "<h1>Glam AI - Virtual Makeup</h1><img src='/video_feed' style='width: 100%;'>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
