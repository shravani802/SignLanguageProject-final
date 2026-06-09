import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

gesture = "thank_you" # CHANGE THIS
sequence_length = 30

folder = f"dataset/{gesture}"
os.makedirs(folder, exist_ok=True)

existing_files = [f for f in os.listdir(folder) if f.endswith(".npy")]
sample_count = len(existing_files)

print("📷 Camera started")
print("Press 's' to record | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collect", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print(f"🎥 Recording sample {sample_count}...")
        time.sleep(1)  # small delay to prepare

        sequence = []

        while len(sequence) < sequence_length:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                    if len(landmarks) == 63:
                        sequence.append(landmarks)

            # show progress
            cv2.putText(frame, f"{len(sequence)}/{sequence_length}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            cv2.imshow("Collect", frame)
            cv2.waitKey(1)

        # skip unstable samples
        if len(sequence) < sequence_length:
            print("⚠ Skipped (unstable detection)")
            continue

        np.save(os.path.join(folder, f"{sample_count}.npy"),
                np.array(sequence))

        print(f"✅ Saved sample {sample_count}")
        sample_count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()