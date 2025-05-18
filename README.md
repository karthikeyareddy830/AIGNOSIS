# 🧠 AIGNOSIS

**AIGNOSIS** is an intelligent medical image diagnostic tool that combines deep learning with Gemini AI to provide medical insights, especially for **bone fractures** and **eye diseases**. It supports interaction in **English**, **Telugu**, and **Hindi**, making it accessible to a broader audience.

---

## 🚀 Features

- 🦴 Detect **bone fractures** and 👁 **eye diseases** using YOLOv8 models.
- 🌐 Supports **multi-language responses** using Google Translate.
- 💬 Chat with an **AI-powered doctor** using Google's Gemini API.
- 🖼 Upload and analyze medical images directly in the app.
- 🎨 Elegant, dark-themed UI using **Streamlit**.

---

## 🧪 Technologies Used

- `Streamlit` – Web app framework
- `YOLOv8` – Object detection
- `Gemini API` – Medical response generation
- `Google Translate API` – Language translation
- `OpenCV`, `Pillow` – Image handling
- `Ultralytics` – YOLO model framework

---

## 📦 Files Included

| File Name       | Description                              |
|------------------|------------------------------------------|
| `error.py`       | Main Streamlit app with all logic        |
| `bone.pt`        | YOLO model for bone fracture detection   |
| `eye_disease.pt` | YOLO model for eye disease classification|

---

## 🛠️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/karthikeyareddy830/AIGNOSIS.git
   cd AIGNOSIS
