import streamlit as st
from functions.model import build_model_tuned_densenet201, prediction
from functions.preprocessing import preprocess_raw_image
import tensorflow.compat.v1 as tf
from PIL import Image
import numpy as np
from io import BytesIO
import cv2

tf.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()

st.set_page_config(page_title="Prediction", page_icon=":rocket:")
st.title("Prédictions")

option = st.selectbox(
    "Quel modèle utiliser ?",
    ("-", "DenseNet201", "VGG16"),
    index=0,
    placeholder="Choisissez un modèle..."
)
st.write('Le modèle choisi est :', option)

@st.cache_resource()
def load_model(model_name):
    if model_name == 'DenseNet201':
        model = build_model_tuned_densenet201()
        model.load_weights(r'C:\Users\tomba\Documents\GitHub\MAR24_BDS_Radios_Pulmonaire\models\model_densenet_masked.weights.h5')
        return model
    # Add other models here
    return None

model = load_model(option)

uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'])
if uploaded_file and model:
    with st.spinner('Processing image...'):
        preprocessed_image = preprocess_raw_image(uploaded_file)
        result = prediction(model, preprocessed_image)
        st.success(f"Prediction: {result}")
elif not model:
    st.error("Model is not loaded.")
elif not uploaded_file:
    st.info("Please upload an image file.")
