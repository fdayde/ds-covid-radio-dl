# Chemin du fichier : streamlit_app.py

import streamlit as st
from functions.model import build_model_tuned_densenet201, prediction, load_model
from functions.preprocessing import preprocess_raw_image, generate_gradcam
import tensorflow.compat.v1 as tf # type: ignore
from PIL import Image
import numpy as np
from io import BytesIO
import cv2

# Configuration de TensorFlow pour fonctionner en mode compatibilité
tf.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()

# Configuration de la page Streamlit
st.set_page_config(page_title="Prédiction d'image", page_icon=":rocket:")
st.title("Outil de Prédiction d'Image")

# Sélection du modèle
option = st.selectbox(
    "Sélectionnez le modèle à utiliser :",
    ("-", "DenseNet201", "VGG16"),
    index=0,
    help="Choisissez un modèle..."
)
st.write('Modèle sélectionné :', option)

# Charger le modèle si une option valide est choisie
if option and option != "-":
    model = load_model(option)
    if not model:
        st.error("Échec du chargement du modèle sélectionné.")
else:
    model = None
    st.info("Veuillez sélectionner un modèle pour activer les prédictions.")

# Téléchargement des fichiers
uploaded_files = st.file_uploader("Téléchargez des fichiers image", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
if uploaded_files and model:
    for uploaded_file in uploaded_files:
        with st.spinner(f'Traitement de l\'image {uploaded_file.name}...'):
            try:
                # Prétraiter l'image téléchargée
                preprocessed_image = preprocess_raw_image(uploaded_file)
                
                # Générer une prédiction
                result = prediction(model, preprocessed_image)
                
                st.success(f"Prédiction pour {uploaded_file.name} : {result}")
                
                # Créer des colonnes pour afficher les images côte à côte
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(preprocessed_image, use_column_width=True, clamp=True, caption="Image Prétraitée")
                
                # Générer Grad-CAM
                superimposed_img = generate_gradcam(model, preprocessed_image, 'conv5_block32_concat')
                
                with col2:
                    st.image(superimposed_img, use_column_width=True, caption="Image Superposée Grad-CAM")
            
            except Exception as e:
                st.error(f"Une erreur est survenue lors du traitement de l'image {uploaded_file.name} : {str(e)}")
elif not uploaded_files and model:
    st.info("Veuillez télécharger des fichiers image pour la prédiction.")

