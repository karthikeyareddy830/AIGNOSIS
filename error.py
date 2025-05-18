import streamlit as st

# This must be the first Streamlit command
st.set_page_config(layout="wide", page_title="AIgnosis", page_icon="🧠")

import google.generativeai as genai
from PIL import Image
import cv2
import os
import tempfile
import numpy as np
from ultralytics import YOLO
from googletrans import Translator

# Gemini setup
genai.configure(api_key="AIzaSyCEkSKb4mTPe1w6qLFOyuAiPnY6rLFD6rU")
model_gemini = genai.GenerativeModel(
    "gemini-1.5-pro-latest",
    system_instruction="You are a professional medical doctor. Answer clearly and briefly."
)

translator = Translator()

# File saving
def save_uploaded_file(uploaded_file):
    path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
    return path

# YOLO detection with confidence and bounding boxes
def detect_with_yolo(image_path, model_path, box_color):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    results = model(image)
    detected = False

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
            label = f"{conf:.2f}"

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 5, y1), box_color, -1)
            cv2.putText(image, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            detected = True

    result_path = os.path.join(tempfile.gettempdir(), os.path.basename(model_path) + "_result.jpg")
    cv2.imwrite(result_path, image)
    return result_path, detected

# Get translated Gemini response
def get_image_response(image_path, query, lang):
    translated_query = translator.translate(query, src=lang, dest="en").text
    image = Image.open(image_path)
    image.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
    response = model_gemini.generate_content([f"Medical image diagnosis query: {translated_query}", image])
    eng_reply = response.text.strip()
    if lang != "en":
        final_reply = translator.translate(eng_reply, src="en", dest=lang).text
    else:
        final_reply = eng_reply
    return final_reply

# Custom UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
* { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(135deg, #000000, #0a0a2a); }
h1, h2, .streamlit-expanderHeader { color: #0ef; text-shadow: 0 0 10px #0ef; }
.stTextInput>div>div>input, .stSelectbox>div>div>select,
.stFileUploader>div>div>div>div {
    background: rgba(0,0,0,0.7); color: #fff;
    border: 1px solid #0ef; border-radius: 20px;
    box-shadow: 0 0 10px #0ef, inset 0 0 5px #0ef;
}
.stButton>button {
    background: #0ef; color: #000;
    border-radius: 30px; font-weight: 500;
    box-shadow: 0 0 15px #0ef;
}
.stChatMessage { background: rgba(0, 0, 0, 0.7);
    border: 1px solid #0ef; box-shadow: 0 0 15px #0ef; border-radius: 20px;
}
[data-testid="stChatMessageContent"] {
    background: rgba(0, 238, 255, 0.1);
    border: 1px solid rgba(0, 238, 255, 0.3);
    border-radius: 15px; padding: 15px;
    box-shadow: inset 0 0 10px #0ef;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center; font-size: 64px; font-weight: 900; color: #0ef; text-shadow: 0 0 20px #0ef;'>
🧠 AIGNOSIS
</h1>
""", unsafe_allow_html=True)

# Session state init
if "image_type" not in st.session_state:
    st.session_state.image_type = "-- Select --"
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# Language selection
lang_code_map = {"English": "en", "Telugu": "te", "Hindi": "hi"}
lang = st.selectbox("🌐 Choose your language", list(lang_code_map.keys()))
lang_code = lang_code_map[lang]

# Upload and selection
col1, col2 = st.columns([1, 2])
with col1:
    image_type = st.selectbox("Select image type", ["-- Select --", "Bone Fracture", "Eye Disease", "Other Diseases"])

    if image_type != st.session_state.image_type:
        st.session_state.image_type = image_type
        st.session_state.image_path = None
        st.session_state.chat_log = []
        st.rerun()

with col2:
    uploaded = st.file_uploader("Upload a Medical Image", type=["jpg", "jpeg", "png"])

# Processing
if uploaded:
    image_path = save_uploaded_file(uploaded)
    st.session_state.image_path = image_path

    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded, caption="Uploaded Image", use_container_width=True)

    if image_type == "Bone Fracture":
        processed_path, _ = detect_with_yolo(image_path, "bone.pt", (0, 0, 255))
    elif image_type == "Eye Disease":
        processed_path, _ = detect_with_yolo(image_path, "eye_disease.pt", (255, 0, 0))
    else:
        processed_path = image_path

    with col2:
        st.image(processed_path, caption="Detection Result", use_container_width=True)

    st.session_state.processed_path = processed_path

    default_questions = [
        "Possible causes of the detected condition.",
        "Basic medical advice.",
        "When to seek professional consultation."
    ]

    with st.spinner("Analyzing image..."):
        for q in default_questions:
            reply = get_image_response(processed_path, q, lang_code)
            st.session_state.chat_log.append(("assistant", f"{q}\n{reply}"))
            with st.expander(f"🧠 {q}"):
                st.success(reply)

# Chat
if st.session_state.image_path:
    st.markdown("### 💬 Chat with the AI Doctor")
    query = st.chat_input("Ask a question about the uploaded image")

    if query:
        st.session_state.chat_log.append(("user", query))
        if query.lower().strip() in ["bye", "exit"]:
            goodbye = {"en": "Goodbye! Take care. 👋", "te": "వీడ్కోలు! జాగ్రత్తగా ఉండండి. 👋", "hi": "अलविदा! ध्यान रखना। 👋"}[lang_code]
            st.session_state.chat_log.append(("assistant", goodbye))
            for key in ["image_type", "image_path", "chat_log", "processed_path"]:
                st.session_state[key] = None
            st.rerun()
        else:
            with st.spinner("Analyzing your question..."):
                reply = get_image_response(st.session_state.processed_path, query, lang_code)
                st.session_state.chat_log.append(("assistant", reply))

# History
st.markdown("### 📜 Chat History")
for sender, msg in st.session_state.chat_log:
    with st.chat_message(sender):
        if sender == "user":
            st.markdown(f"👤 You: {msg}")
        else:
            st.markdown(f"🤖 AI Doctor: {msg}")