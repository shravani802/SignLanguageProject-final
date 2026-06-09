import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

print("🔹 Loading model...")
model = load_model("model.h5")
print("✅ Model loaded")

labels = [
    "hello","thank_you","yes","no","sorry","help","drink","okay","i_love_you","mom"
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

sequence = []
sequence_length = 30

print("📷 Running... Press q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmarks = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

    if len(landmarks) == 63:
        sequence.append(landmarks)

    sequence = sequence[-sequence_length:]

    text = "Show gesture..."

    if len(sequence) == sequence_length:
        input_data = np.array(sequence)

        # ✅ FIXED normalization (consistent)
        max_val = np.max(input_data)
        if max_val == 0:
            max_val = 1
        input_data = input_data / max_val

        input_data = np.expand_dims(input_data, axis=0)

        prediction = model.predict(input_data, verbose=0)[0]

        pred = np.argmax(prediction)
        conf = prediction[pred]

        print("Conf:", conf)

        if conf > 0.5:
            text = f"{labels[pred]} ({conf:.2f})"
        else:
            text = f"Not clear ({conf:.2f})"

    cv2.putText(frame, text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()