import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
from google import genai
import os 
import gdown


# os.environ["GOOGLE_API_KEY"]  = "AIzaSyDfkkMquLJv_BZXlqyYkAHfkRloR6-y1mw"
a=st.secrets('GeminiAPi')
client = genai.Client(api_key=a)


st.title("🌱 Crop Disease Prediction (Predefined Model)")

# ------------------- Load Predefined Model -------------------
# # Replace the path below with your local model path
# MODEL_PATH = r"https://drive.google.com/file/d/1Kf2P3N5djVM0e4wxM8Ls2OrwOF57mDL_/view?usp=sharing"

# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model(MODEL_PATH)
#     return model

# model = load_model()
# st.success("Model loaded successfully!")
# MODEL_PATH = r"/Users/ayushagrawal/Desktop/dhriti/crop_disease_manual_model.h5"

# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model(MODEL_PATH)
#     return model

# model = load_model()
# st.success("Model loaded successfully!")
FILE_ID = "1Kf2P3N5djVM0e4wxM8Ls2OrwOF57mDL_" 
MODEL_PATH = "my_model.h5"  # local filename

# Download the model if it does not exist
if not os.path.exists(MODEL_PATH):
   url = f"https://drive.google.com/uc?id={FILE_ID}"
   gdown.download(url, MODEL_PATH, quiet=False)

# # Streamlit cache to load model
# @st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()
st.success("Model loaded successfully!")
# ------------------- Upload Image -------------------
img_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if img_file:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

# ------------------- Predict Button -------------------
if st.button("Predict"):

    if img_file is None:
        st.error("Please upload an image.")
        st.stop()

    # Auto-detect input size
    input_w = model.input_shape[1]
    input_h = model.input_shape[2]
    input_channels = model.input_shape[3]

    img = image.resize((input_w, input_h))
    img_arr = np.array(img) / 255.0

    if input_channels == 1:
        img_arr = np.mean(img_arr, axis=2, keepdims=True)

    img_arr = np.expand_dims(img_arr, axis=0)

    # Prediction
    pred = model.predict(img_arr)[0]
    class_index = int(np.argmax(pred))

    # Define your classes here
    class_names = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___healthy','Tomato__Target_Spot',
'Tomato__Tomato_mosaic_virus',
'Tomato__Tomato_YellowLeaf__Curl_Virus',
'Tomato_Bacterial_spot',
'Tomato_Early_blight',
'Tomato_Late_blight',
'Tomato_Leaf_Mold',
'Tomato_Septoria_leaf_spot',
]

    # JSON Output
    output = {
        "label": class_names[class_index],
        "confidence": float(pred[class_index]),
        "probabilities": {class_names[i]: float(pred[i]) for i in range(len(class_names))}
    }

    st.subheader("JSON Response")
    st.code(json.dumps(output, indent=4))

prompt = f"""
You are an expert agricultural assistant.
Given this crop disease prediction JSON:
{json.dumps(output)}
Provide disease solution, treatment, fertilizer/pesticide recommendations, and care instructions.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents = prompt
)


ai_solution = response.text
st.write("AI Solution:\n", ai_solution)


