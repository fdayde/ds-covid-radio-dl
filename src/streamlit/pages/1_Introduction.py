import streamlit as st
from pathlib import Path
import os
from PIL import Image
import re
import pandas as pd
import plotly.graph_objects as go
from functions.footer import add_footer


st.set_page_config(page_title="Introduction", page_icon=":sparkles:", layout="wide")

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 3rem;
                    padding-left: 10rem;
                    padding-right: 10rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


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

st.title("Introduction")
st.sidebar.title("Introduction")
pages = ["Context: COVID", "Context: Deep Learning", "Objective"]
page = st.sidebar.radio("Summary", pages, label_visibility="collapsed")


pictures_path = os.path.join(os.path.dirname(__file__), "..", "pictures")


# Définir le chemin vers le dossier contenant les images pour l'animation
images_animation_lungs = os.path.join(pictures_path, "COVID_lung_animation")


# Fonction pour extraire les numéros des noms de fichiers
def extract_number(filename):
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else 0


# Lister toutes les images dans le dossier
image_files = sorted(
    [
        os.path.join(images_animation_lungs, file)
        for file in os.listdir(images_animation_lungs)
    ],
    key=lambda x: extract_number(os.path.basename(x)),
)

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

covid_markdown_intro = """

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


covid_markdown_health = """

### Inflammation 


### Pneumonia


### Lung Damage


### Long Recovery
"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

covid_markdown_lungs = """
As the infection progresses, lesions start to appear on the lungs :
"""


##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##


covid_markdown_diagnostic = """

- **ELISA TESTS** : **Quick** (15 minutes) but **low specificity**

- **PCR** : **Long** (a day) with **average to high specificity**, not available everywhere

- **RADIOLOGY** : **Quick** (15 minutes), available in every healthcare facility
"""


##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##


deep_learning_markdown = """
#### Introduction to Deep Learning Models

Deep learning, a subset of machine learning, involves neural networks with many layers. These models are designed to automatically learn representations of data through multiple levels of abstraction.

Popular Deep Learning Models :

- **Convolutional Neural Networks (CNNs)**: Excellent for image and video recognition tasks, leveraging convolutional layers to capture spatial hierarchies.
- **Recurrent Neural Networks (RNNs)**: Ideal for sequential data like time series or natural language, using loops to maintain context.
- **Transformers**: Advanced models for processing sequential data, particularly in natural language processing, with a mechanism called attention.

Deep learning has revolutionized fields such as computer vision, natural language processing, and speech recognition, enabling significant advancements in technology and research.
"""

deep_learning_use = """
#### Nowadays, deep learning is commonly used in all sorts of advanced tasks, such as **face recognition**, **cancer diagnostic**, **LLM (ChatGPT)**,... and it seems its potential is almost limitless.
"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

objective_markdown = """

In a healthcare facility setting, when a patient presents with signs of a lung infection whether it is a suspected COVID infection or not, it is crucial for the clinician to
differentiate between COVID and other lung afflictions.

We decided that it would be best to develop a Deep Learning model capable of classifying between a COVID infection, other types of pneumonias, or even a healthy lung.

"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##


if page == "Context: COVID":
    st.header("COVID-19 Overview", divider="rainbow")

    st.markdown(covid_markdown_intro, unsafe_allow_html=True)
    st.subheader("Lung Impact", divider="grey")
    ## Séparer l'image et le texte dans 2 colonnes différentes pour les afficher côtes-à-côtes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("", divider="grey")
        st.image(
            str(os.path.join(pictures_path, "COVID19_lung_impact.jpg")),
            caption="COVID19 action on lungs, drawn by Brooke Ring",
            use_column_width=True,
        )

    with col2:
        st.subheader("", divider="grey")
        ## Animation de l'évolution d'un poumon de patient COVID dans le temps

        # Fonction pour charger une image
        def load_image(image_path):
            return Image.open(image_path)

        # Initialiser l'index de l'image dans la session de Streamlit
        if "img_index" not in st.session_state:
            st.session_state.img_index = 0

        # Afficher l'image actuelle
        current_image = load_image(
            os.path.join(
                images_animation_lungs, image_files[st.session_state.img_index]
            )
        )
        st.image(
            current_image,
            caption=f"Image {st.session_state.img_index + 1} sur {len(image_files)}",
        )

        # Boutons de navigation
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("Previous"):
                if st.session_state.img_index > 0:
                    st.session_state.img_index -= 1

        with col2:
            st.write("")

        with col3:
            if st.button("Next"):
                if st.session_state.img_index < len(image_files) - 1:
                    st.session_state.img_index += 1

        st.write(
            "Lungs X-Rays of a COVID-19 patient : Evolution, Rousan L.A. et a., 2020"
        )

        st.markdown(covid_markdown_lungs)

    st.subheader("COVID-19 Diagnostic", divider="grey")

    st.markdown(covid_markdown_diagnostic)

if page == "Context: Deep Learning":
    st.header("Introduction", divider="rainbow")

    # Data for the timeline
    data = {
        "Year": [1950, 1957, 1974, 1986, 1989, 1990, 1997, 2006, 2012, 2014, 2016],
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
            "AlphaGo defeats world champion Go player",
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
            "AlphaGo showcased the potential of deep reinforcement learning by defeating the world champion Go player.",
        ],
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    df["Year"] = df["Year"].astype(int)
    df.set_index("Year", inplace=True)

    # Define different lengths for the vertical lines to prevent overlap
    lengths = [1.5, 1, 1.7, 1.2, 1.8, 1.3, 2, 1.5, 1.6, 1.4, 1.9]

    st.subheader("Timeline of Advancements in Machine Learning", divider="grey")

    # Create the timeline figure
    fig = go.Figure()

    # Add the horizontal line
    fig.add_trace(
        go.Scatter(
            x=[1940, df.index.max()],
            y=[0, 0],
            mode="lines",
            line=dict(color="red", width=4),
            showlegend=False,
        )
    )

    # Add events to the timeline with alternating positions and lengths
    for i, (year, row) in enumerate(df.iterrows()):
        y_position = lengths[i] if i % 2 == 0 else -lengths[i]
        # Draw vertical lines to the event points
        fig.add_trace(
            go.Scatter(
                x=[year, year],
                y=[0, y_position],
                mode="lines",
                line=dict(color="red", width=1),
                showlegend=False,
            )
        )
        # Add markers and event texts
        fig.add_trace(
            go.Scatter(
                x=[year],
                y=[y_position],
                mode="markers+text",
                text=[row["Event"]],
                textposition="top center" if y_position > 0 else "bottom center",
                marker=dict(size=10, color="red"),
                hoverinfo="text",
                name=row["Event"],
            )
        )

    # Customize the layout
    fig.update_layout(
        title="",
        xaxis_title="Year",
        yaxis=dict(
            showticklabels=False, showgrid=False, zeroline=False, range=[-2.5, 2.5]
        ),
        xaxis=dict(
            tickmode="linear",
            tick0=1940,
            dtick=10,
        ),
        showlegend=False,
        height=800,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # Display the plot
    st.plotly_chart(fig)

    # Display the data with corrected date format and set as index
    st.subheader("Key Milestones in Machine Learning and Deep Learning", divider="grey")
    st.dataframe(df, use_container_width=True)

    for _ in range(2):
        st.write("")

    st.subheader("", divider="grey")

    st.markdown(deep_learning_use)


if page == "Objective":
    st.header("Introduction", divider="rainbow")
    st.subheader("What We Wanted To Achieve", divider="grey")
    st.markdown(objective_markdown)


add_footer()
