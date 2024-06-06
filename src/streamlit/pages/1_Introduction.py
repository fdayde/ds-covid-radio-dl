import streamlit as st
from pathlib import Path
import os
from PIL import Image
import re
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64

st.set_page_config(
    page_title="Introduction",
    page_icon=":sparkles:",
    layout="wide"
)

# Define styles for your lines and text
style = """
<style>
.title {
  font-size: 24px;
  font-weight: bold;
  color: white;
}
hr {
  border-top: 2px solid #FF5733;  # You can change the color code as needed
}
</style>
"""

st.sidebar.title("Introduction")
pages = ["Context : COVID", "Context : Deep Learning", "Objective"]
page = st.sidebar.radio("Summary", pages, label_visibility="collapsed")


# Chemin d'accès pour les images du Streamlit
base_path = Path.cwd()
images = base_path / 'pictures'
banner_path = str(images / 'banner_streamlit.jpg')


# Définir le chemin vers le dossier contenant les images pour l'animation
images_animation_lungs = images / 'COVID_lung_animation'

# Fonction pour extraire les numéros des noms de fichiers
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

# Lister toutes les images dans le dossier
image_files = sorted([os.path.join(images_animation_lungs, file) for file in os.listdir(images_animation_lungs)], key = lambda x: extract_number(os.path.basename(x)))

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

covid_markdown_intro = """
## 1. COVID-19 Overview

<div style='text-align: justify;'>
COVID-19, caused by SARS-CoV-2, emerged in late 2019 and quickly became a 
global pandemic. It spreads through respiratory droplets and causes symptoms 
ranging from mild (fever, cough, fatigue) to severe (difficulty breathing, ARDS). 
It mainly affects the lungs and severe cases, especially in older adults and 
those with underlying conditions, can lead to death. It posed a significant
healthcare problem, with a quick spread and a saturation of healtcare facilities.

**One of the most important issue was a rapid diagnostic of the condition**, as it
was not only a way to reduce the spread (isolation) and to better take care of
the patient.
</div>
"""


covid_markdown_health="""
## Lung Impact
- **Inflammation** 

    COVID-19 can cause significant inflammation in the lungs, leading to swelling and 
    irritation of the lung tissue. This inflammation is primarily due to the body's immune response to the virus.

- **Pneumonia**

    COVID-19 can lead to viral pneumonia, where the air sacs (alveoli) in the lungs become inflamed and filled 
    with fluid or pus. This can cause severe breathing difficulties and reduce oxygen levels in the blood.

- **Lung Damage**

    The severe inflammation and pneumonia caused by COVID-19 can result in lasting lung damage. This includes 
    scarring (fibrosis) and long-term loss of lung function, which can persist even after recovery from the acute phase of the illness.

- Long Recovery
    Recovery from severe COVID-19 lung involvement can be prolonged. Patients may experience lingering symptoms such as fatigue, 
    shortness of breath, and reduced lung capacity for weeks or months after the infection has cleared.
"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

covid_markdown_lungs="""
As the infection progresses, lesions start to appear on the lungs :
"""


##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##


covid_markdown_diagnostic = """
## COVID-19 Diagnostic

#### - **ELISA TESTS** : **Quick** (15 minutes) but **low specificity**

#### - **PCR** : **Long** (a day) with **average to high specificity**, not available everywhere

#### - **RADIOLOGY** : **Quick** (15 minutes), available in every healthcare facility
"""




##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##


deep_learning_markdown = """
### Introduction to Deep Learning Models

Deep learning, a subset of machine learning, involves neural networks with many layers (hence "deep"). These models are designed to automatically learn representations of data through multiple levels of abstraction.

Popular Deep Learning Models :

- **Convolutional Neural Networks (CNNs)**: Excellent for image and video recognition tasks, leveraging convolutional layers to capture spatial hierarchies.
- **Recurrent Neural Networks (RNNs)**: Ideal for sequential data like time series or natural language, using loops to maintain context.
- **Transformers**: Advanced models for processing sequential data, particularly in natural language processing, with a mechanism called attention.

Deep learning has revolutionized fields such as computer vision, natural language processing, and speech recognition, enabling significant advancements in technology and research.
"""

deep_learning_use="""
#### Nowadays, deep learning is commonly used in all sorts of advanced tasks, such as **face recognition**, **cancer diagnostic**, **LLM (ChatGPT)**,... and it seems its potential is almost limitless.
"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

objective_markdown = """

In a healthcare facility setting, when a patient presents with signs of a lung infection whether it is a suspected COVID infection or not, it is crucial for the clinician to
differentiate between COVID and other lung afflictions.

We decided that it would be best to develop a Deep Learning model capable of classifying between a COVID infection, other types of pneumonias, or even a healthy lung.

"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##


if page == "Context : COVID":
    st.markdown('<hr>', unsafe_allow_html=True)
    st.write("# Introduction")
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(covid_markdown_intro, unsafe_allow_html=True)

    ## Séparer l'image et le texte dans 2 colonnes différentes pour les afficher côtes-à-côtes
    col1, col2 = st.columns(2)
    st.markdown('<hr>', unsafe_allow_html=True)
    with col1:
        st.markdown(covid_markdown_health)
    with col2:
        for _ in range(4):
            st.write("")
        st.image(str(images / 'COVID19_lung_impact.jpg'), caption = 'COVID19 action on lungs, drawn by Brooke Ring', use_column_width=True)

    for _ in range(2):
        st.write("")

    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown(covid_markdown_lungs)

    ## Animation de l'évolution d'un poumon de patient COVID dans le temps

    ## Initialiser l'index de l'image
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0
    
    ## Afficher l'image actuelle
    image_path = image_files[st.session_state.image_index]
    image = Image.open(image_path)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    # Create three columns
    col1, col2, col3 = st.columns([1,2,1])
    # Using the middle column to place the image
    with col2:
        st.image(image, caption=f'Image {st.session_state.image_index + 1}/{len(image_files)}', use_column_width=False)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

    st.write("<div style='text-align: center;'>" "Lungs X-Rays of a COVID-19 patient : Evolution, Rousan L.A. et a., 2020", unsafe_allow_html=True)

    ## Slider to control the image index
    st.session_state.image_index = st.slider("Image Index", 0, len(image_files) - 1, st.session_state.image_index, step=1)

    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown(covid_markdown_diagnostic)

    st.markdown('<hr>', unsafe_allow_html=True)

if page == "Context : Deep Learning":
    st.markdown('<hr>', unsafe_allow_html=True)
    st.title("Introduction")
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""# 2. Machine Learning and Deep Learning Overview""")
    st.markdown(deep_learning_markdown)
    st.markdown('<hr>', unsafe_allow_html=True)
    # Data for the timeline
    data = {
        "Year": [
            1950, 1957, 1974, 1986, 1989, 1990, 1997, 2006, 2012, 2014, 2016
        ],
        "Event": [
            "Alan Turing proposes the Turing Test",
            "Frank Rosenblatt introduces the Perceptron",
            "Paul Werbos develops Backpropagation algorithm",
            "Backpropagation algorithm gains popularity",
            "Yann LeCun develops Convolutional Neural Networks",
            "First NeurIPS Conference",
            "LSTM networks developed",
            "Geoffrey Hinton introduces Deep Belief Networks",
            "AlexNet wins ImageNet competition",
            "GANs introduced by Ian Goodfellow",
            "AlphaGo defeats world champion Go player"
        ],
        "Description": [
            "A test to determine a machine's ability to exhibit intelligent behavior indistinguishable from a human.",
            "The Perceptron is a type of artificial neural network and one of the earliest models for supervised learning.",
            "Paul Werbos develops the Backpropagation algorithm, a fundamental method for training neural networks.",
            "Backpropagation is a method used to efficiently train neural networks.",
            "Convolutional Neural Networks (CNNs) are a class of deep neural networks, most commonly applied to analyzing visual imagery.",
            "The Neural Information Processing Systems (NeurIPS) conference has become a major event in the field of machine learning and neural networks.",
            "Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning long-range dependencies.",
            "Deep Belief Networks are a type of deep neural network capable of unsupervised learning.",
            "AlexNet demonstrated the superiority of deep learning for image recognition tasks.",
            "Generative Adversarial Networks (GANs) opened new possibilities for generative models.",
            "AlphaGo showcased the potential of deep reinforcement learning by defeating the world champion Go player."
        ]
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    df['Year'] = df['Year'].astype(int)
    df.set_index("Year", inplace=True)

    # Define different lengths for the vertical lines to prevent overlap
    lengths = [1.5, 1, 1.7, 1.2, 1.8, 1.3, 2, 1.5, 1.6, 1.4, 1.9]

    # Streamlit App
    st.markdown("### Timeline of Advancements in Machine Learning")

    # Create the timeline figure
    fig = go.Figure()

    # Add the horizontal line
    fig.add_trace(go.Scatter(
        x=[1940, df.index.max()],
        y=[0, 0],
        mode='lines',
        line=dict(color='red', width=4),
        showlegend=False
    ))

    # Add events to the timeline with alternating positions and lengths
    for i, (year, row) in enumerate(df.iterrows()):
        y_position = lengths[i] if i % 2 == 0 else -lengths[i]
        # Draw vertical lines to the event points
        fig.add_trace(go.Scatter(
            x=[year, year],
            y=[0, y_position],
            mode='lines',
            line=dict(color='red', width=1),
            showlegend=False
        ))
        # Add markers and event texts
        fig.add_trace(go.Scatter(
            x=[year],
            y=[y_position],
            mode='markers+text',
            text=[row["Event"]],
            textposition="top center" if y_position > 0 else "bottom center",
            marker=dict(size=10, color='red'),
            hoverinfo='text',
            name=row["Event"]
        ))

    # Customize the layout
    fig.update_layout(
        title="",
        xaxis_title="Year",
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-2.5, 2.5]
        ),
        xaxis=dict(
            tickmode='linear',
            tick0=1940,
            dtick=10,
        ),
        showlegend=False,
        height=800,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Display the plot
    st.plotly_chart(fig)

    # Display the data with corrected date format and set as index
    st.write("### Key Milestones in Machine Learning and Deep Learning")
    st.dataframe(df, use_container_width=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    for _ in range(2):
                st.write("")

    st.markdown(deep_learning_use)


def get_image_as_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded_string}"

# Function to add a footer with names, LinkedIn links, and an image
def add_footer(image_path):
    image_url = get_image_as_base64(image_path)
    footer = f"""
    <style>
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #333;
        color: #f1f1f1;
        text-align: center;
        padding: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .footer .links {{
        flex: 1;
        text-align: center;
    }}
    .footer .image {{
        margin-right: 20px;
    }}
    .footer a {{
        color: #0a66c2;
        margin: 0 10px;
        text-decoration: none;
    }}
    .footer a:hover {{
        text-decoration: underline;
    }}
    </style>
    <div class="footer">
        <div class="links">
            <p>Projet de Data Science sur la classification des radiographies pulmonaires COVID-19 | © 2024</p>
            <p>
                <a href="https://www.linkedin.com/in/thomas-barret/" target="_blank" rel="noopener noreferrer">Thomas Barret</a> |
                <a href="https://www.linkedin.com/in/nicolas-bouzinbi/" target="_blank" rel="noopener noreferrer">Nicolas Bouzinbi</a> |
                <a href="https://www.linkedin.com/in/florent-dayde/" target="_blank" rel="noopener noreferrer">Florent Daydé</a> |
                <a href="https://www.linkedin.com/in/nicolas-fenassile/" target="_blank" rel="noopener noreferrer">Nicolas Fenassile</a>
            </p>
        </div>
        <div class="image">
            <img src="{image_url}" alt="Logo" height="50">
        </div>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

# Path to the uploaded image
logo_path = str(images / 'Logo_Datascientest.png')

# Add the footer with the image URL
add_footer(logo_path)


if page == "Objective":
    st.title("Introduction")
    st.markdown("""# 3. What we wanted to achieve""")
    st.markdown(objective_markdown)