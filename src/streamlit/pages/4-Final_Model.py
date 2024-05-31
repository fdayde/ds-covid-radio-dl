# File path: streamlit_app.py

import streamlit as st
from functions.model import build_model_tuned_densenet201, prediction, load_model
from functions.preprocessing import preprocess_raw_image, generate_gradcam
import tensorflow.compat.v1 as tf # type: ignore
from PIL import Image
import numpy as np
from io import BytesIO
import cv2

# Configure TensorFlow to run in compatibility mode
tf.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()

# Configure the Streamlit page
st.set_page_config(page_title="Image Prediction", page_icon=":rocket:")
st.title("Final Model: DenseNet201")

# Model selection
option = st.selectbox(
    "Select the model to use:",
    ("-", "DenseNet201", "VGG16"),
    index=0,
    help="Choose a model..."
)
st.write('Selected model:', option)

# Load the model if a valid option is chosen
if option and option != "-":
    model = load_model(option)
    if not model:
        st.error("Failed to load the selected model.")
else:
    model = None
    st.info("Please select a model to enable predictions.")

# File upload
uploaded_files = st.file_uploader("Upload image files", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
if uploaded_files and model:
    for uploaded_file in uploaded_files:
        with st.spinner(f'Processing image {uploaded_file.name}...'):
            try:
                # Preprocess the uploaded image
                preprocessed_image = preprocess_raw_image(uploaded_file)
                
                # Generate a prediction
                result = prediction(model, preprocessed_image)
                
                st.success(f"Prediction for {uploaded_file.name}: {result}")
                
                # Create columns to display images side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(preprocessed_image, use_column_width=True, clamp=True, caption="Preprocessed Image")
                
                # Generate Grad-CAM
                superimposed_img = generate_gradcam(model, preprocessed_image, 'conv5_block32_concat')
                
                with col2:
                    st.image(superimposed_img, use_column_width=True, caption="Superimposed Grad-CAM Image")
            
            except Exception as e:
                st.error(f"An error occurred while processing the image {uploaded_file.name}: {str(e)}")
elif not uploaded_files and model:
    st.info("Please upload image files for prediction.")
