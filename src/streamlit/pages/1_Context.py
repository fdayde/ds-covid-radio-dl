import streamlit as st
from pathlib import Path
import os
from PIL import Image
import re

st.set_page_config(
    page_title="Context",
    page_icon=":sparkles:",
)

st.sidebar.title("Context")
pages = ["COVID-19", "Deep Learning", "Data"]
page = st.sidebar.radio("Summary", pages, label_visibility="collapsed")


# Chemin d'accès pour les images du Streamlit
images = Path(r'C:\Users\Nicolas\Documents\DataScience\MAR24_BDS_Radios_Pulmonaire\src\streamlit\pictures')


# Définir le chemin vers le dossier contenant les images pour l'animation
images_animation_lungs = r'C:\Users\Nicolas\Documents\DataScience\MAR24_BDS_Radios_Pulmonaire\src\streamlit\pictures\COVID_lung_animation'

# Fonction pour extraire les numéros des noms de fichiers
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

# Lister toutes les images dans le dossier
image_files = sorted([os.path.join(images_animation_lungs, file) for file in os.listdir(images_animation_lungs)], key = lambda x: extract_number(os.path.basename(x)))



covid_markdown_intro = """
# COVID-19 Overview

**COVID-19**, caused by SARS-CoV-2, emerged in late 2019 and quickly became a global pandemic. It spreads through respiratory droplets and causes symptoms ranging from mild (fever, cough, fatigue) to severe (difficulty breathing, ARDS). Severe cases, especially in older adults and those with underlying conditions, can lead to death.
"""


covid_markdown_health="""
## Lung Impact
- **Inflammation**

- **Pneumonia**

- **Lung Damage**

- **Long Recovery**
"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

covid_markdown_lungs="""
As the infection progresses, lesions start to appear on the lungs :S
"""


##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##


covid_markdown_diagnostic = """
## COVID-19 Diagnostic

- ** ELISA Tests ** : **Quick** (15 minutes) but **low specificity**

- ** PCR ** : **Long** (a day) with **average to high specificity**, not available everywhere

- ** Radiology ** : **Quick** (15 minutes), available in every healthcare facility
"""







##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##
deep_learning_markdown = """
# Introduction to Deep Learning Models

Deep learning, a subset of machine learning, involves neural networks with many layers (hence "deep"). These models are designed to automatically learn representations of data through multiple levels of abstraction.

## Key Concepts

- **Neural Networks**: Composed of layers of interconnected nodes, or neurons, that process input data and learn to make predictions.
- **Layers**: 
  - **Input Layer**: Takes in the raw data.
  - **Hidden Layers**: Perform computations and feature transformations.
  - **Output Layer**: Produces the final prediction or classification.
- **Activation Functions**: Introduce non-linearity, allowing the network to learn complex patterns. Examples include ReLU, sigmoid, and tanh.
- **Training**: Involves adjusting the weights of connections between neurons using a process called backpropagation and an optimization algorithm like gradient descent.
- **Overfitting**: A common issue where the model learns the training data too well, including its noise, and performs poorly on new data. Techniques like dropout and regularization help mitigate this.

## Popular Deep Learning Models

- **Convolutional Neural Networks (CNNs)**: Excellent for image and video recognition tasks, leveraging convolutional layers to capture spatial hierarchies.
- **Recurrent Neural Networks (RNNs)**: Ideal for sequential data like time series or natural language, using loops to maintain context.
- **Transformers**: Advanced models for processing sequential data, particularly in natural language processing, with a mechanism called attention.

Deep learning has revolutionized fields such as computer vision, natural language processing, and speech recognition, enabling significant advancements in technology and research.
"""

data_markdown = """
# Dataset

We worked on the COVID-QU-Ex Dataset available on [Kaggle](https://www.kaggle.com/datasets/cf77495622971312010dd5934ee91f07ccbcfdea8e2f7778977ea8485c1914df).

This data set is a more complete version of the initially proposed dataset. 
It was built by The researchers of Qatar University have compiled the COVID-QU-Ex dataset, which consists of 33,920 chest X-ray (CXR) images including:

- 11,956 COVID-19
- 11,263 Non-COVID infections (Viral or Bacterial Pneumonia)
- 10,701 Normal

Ground-truth lung segmentation masks are provided for the entire dataset.

"""

if page == "COVID-19":
    st.write("### COVID-19")
    st.markdown(covid_markdown_intro)

    ## Séparer l'image et le texte dans 2 colonnes différentes pour les afficher côtes-à-côtes
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(covid_markdown_health)
    with col2:
        for _ in range(2):
            st.write("")
        st.image(str(images / 'COVID19_lung_impact.jpg'), caption = 'COVID19 action on lungs, drawn by Brooke Ring', use_column_width=True)

    for _ in range(2):
        st.write("")

    st.markdown(covid_markdown_lungs)

    ## Animation de l'évolution d'un poumon de patient COVID dans le temps

    ## Initialiser l'index de l'image
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0
    
    ## Fonction pour mettre à jour l'index de l'image
    def next_image():
        st.session_state.image_index = (st.session_state.image_index + 1) % len(image_files)
    
    ## Afficher l'image actuelle
    image_path = image_files[st.session_state.image_index]
    image = Image.open(image_path)
    st.image(image, caption=f'Image {st.session_state.image_index + 1}/{len(image_files)}', use_column_width=True)

    ## Bouton pour passer à l'image suivante
    st.button("Next Image", on_click=next_image)
    st.write("Lungs X-Rays of a COVID-19 patient : Evolution, Rousan L.A. et a., 2020")

    st.markdown(covid_markdown_diagnostic)






if page == "Deep Learning":
    st.write("### Deep Learning")
    st.markdown(deep_learning_markdown)


if page == "Data":
    st.write("### Data")
    st.markdown(data_markdown)


