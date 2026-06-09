# 🤟 Sign Language Recognition System

A real-time Sign Language Recognition System that uses Computer Vision, MediaPipe, and Deep Learning (LSTM) to recognize hand gestures and convert them into meaningful words.

## 📌 Project Overview

This project aims to bridge the communication gap between hearing-impaired individuals and others by recognizing sign language gestures through a webcam and displaying the corresponding text in real time.

The system uses:

- MediaPipe for hand landmark detection
- TensorFlow/Keras LSTM model for gesture classification
- OpenCV for video capture and processing
- Flask for web deployment

---

## 🚀 Features

✅ Real-time gesture recognition

✅ Hand landmark extraction using MediaPipe

✅ Deep Learning based gesture classification

✅ Web-based interface using Flask

✅ Multiple gesture support

✅ Easy to extend with new signs

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| OpenCV | Image Processing |
| MediaPipe | Hand Landmark Detection |
| TensorFlow/Keras | Deep Learning Model |
| NumPy | Numerical Computation |
| Flask | Web Application Framework |
| HTML/CSS | Frontend Development |

---

## 📂 Project Structure

```text
SignLanguageProject_v2/
│
├── dataset/
│   ├── hello/
│   ├── help/
│   ├── i_love_you/
│   ├── mom/
│   ├── no/
│   ├── okay/
│   ├── sorry/
│   ├── thank_you/
│   └── yes/
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── collect_sequences.py
├── train_lstm.py
├── predict_live.py
├── model.h5
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The dataset contains hand gesture sequences for the following signs:

- Hello
- Help
- I Love You
- Mom
- No
- Okay
- Sorry
- Thank You
- Yes

Each gesture is represented using MediaPipe hand landmark coordinates collected across multiple frames.

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/sign-language-recognition.git
cd sign-language-recognition
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

### Step 1: Collect Gesture Sequences

```bash
python collect_sequences.py
```

This captures hand landmarks and stores gesture sequences in the dataset folder.

### Step 2: Train the LSTM Model

```bash
python train_lstm.py
```

After training, the model is saved as:

```text
model.h5
```

---

## ▶️ Running the Project

### Option 1: Live Prediction

```bash
python predict_live.py
```

The webcam will open and recognize gestures in real time.

### Option 2: Run Flask Web Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 🔍 Working Principle

1. Webcam captures live video.
2. MediaPipe detects hand landmarks.
3. Landmark coordinates are collected over a sequence of frames.
4. Sequence data is passed to the trained LSTM model.
5. Model predicts the corresponding sign.
6. Predicted text is displayed on the screen.

---

## 📈 Future Enhancements

- Support for complete sign language sentences
- Speech synthesis from recognized signs
- Mobile application integration
- Support for dynamic gestures
- Multi-hand recognition
- Larger vocabulary dataset

---

## 🎯 Applications

- Communication assistance for hearing-impaired individuals
- Educational tools
- Smart human-computer interaction
- Healthcare and accessibility systems

---

## 👩‍💻 Authors

Developed as an academic project for Sign Language Recognition using Deep Learning and Computer Vision.

---

## 📜 License

This project is developed for educational and research purposes.
