from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

app = Flask(__name__)

# -----------------------------
# GLOBALS
# -----------------------------
camera = None
running = False
latest_prediction = "Waiting..."

# Load model
model = load_model("model.h5")

# ⚠️ MUST match training order
labels = ["hello", "thank_you", "sorry"]

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Sequence + smoothing
sequence = []
predictions = []
sequence_length = 30


# -----------------------------
# KEYPOINT EXTRACTION
# -----------------------------
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
    else:
        return np.zeros(21 * 3)  # 63


# -----------------------------
# CAMERA STREAM
# -----------------------------
def generate_frames():
    global camera, running, latest_prediction
    global sequence, predictions

    camera = cv2.VideoCapture(0)

    while running:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Extract landmarks
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        if len(sequence) > sequence_length:
            sequence.pop(0)

        # 🔥 PREDICTION
        if len(sequence) == sequence_length:
            input_data = np.expand_dims(sequence, axis=0)

            pred = model.predict(input_data, verbose=0)[0]
            conf = np.max(pred)
            label = labels[np.argmax(pred)]

            # Only accept confident predictions
            if conf > 0.7:
                predictions.append(label)

            # Keep last 10 predictions
            if len(predictions) > 10:
                predictions.pop(0)

            # Majority voting
            if predictions:
                latest_prediction = max(set(predictions), key=predictions.count)

        # Display text
        cv2.putText(frame, latest_prediction, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/start")
def start():
    global running, sequence, predictions

    running = True
    sequence = []
    predictions = []

    return jsonify({"status": "started"})


@app.route("/stop")
def stop():
    global running, camera
    running = False

    if camera:
        camera.release()

    return jsonify({"status": "stopped"})


@app.route("/prediction")
def prediction():
    return jsonify({"label": latest_prediction})


# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)