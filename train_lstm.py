import numpy as np
import os
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

labels = {
    "hello": 0,"thank_you":1,"yes":2,"no":3,"sorry":4,"help":5,"drink":6,"okay":7,"i_love_you":8,"mom":9
}

num_classes = len(labels)

X, y = [], []

# Load dataset
for gesture, label in labels.items():
    folder = f"dataset/{gesture}"

    if not os.path.exists(folder):
        print(f"⚠ Missing: {folder}")
        continue

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder, file))

            if data.shape != (30, 63):
                print(f"❌ Skipping {file}, shape={data.shape}")
                continue

            X.append(data)
            y.append(label)

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("❌ No data found")
    exit()

print("Dataset:", X.shape)
print("Distribution:", Counter(y))

# Normalize
max_vals = np.max(X)
if max_vals == 0:
    max_vals = 1
X = X / max_vals

# Shuffle
X, y = shuffle(X, y, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# One-hot
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Model
model = Sequential([
    Input(shape=(30, 63)),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Accuracy: {acc:.2f}")

model.save("model.h5")
print("✅ Model saved")