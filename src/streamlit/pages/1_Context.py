import streamlit as st

st.set_page_config(
    page_title="Context",
    page_icon=":sparkles:",
)

st.title("Context")
st.sidebar.title("Context")
pages = ["COVID-19", "Deep Learning", "Data"]
page = st.sidebar.radio("Summary", pages, label_visibility="collapsed")


covid_markdown = """
# COVID-19 Overview and Impact on Health

**COVID-19**, caused by SARS-CoV-2, emerged in late 2019 and quickly became a global pandemic. It spreads through respiratory droplets and causes symptoms ranging from mild (fever, cough, fatigue) to severe (difficulty breathing, ARDS). Severe cases, especially in older adults and those with underlying conditions, can lead to death.

## Health Impact
- **Symptoms**: Fever, cough, fatigue, loss of taste/smell, difficulty breathing.
- **Long COVID**: Prolonged symptoms like fatigue and cognitive issues.
- **Healthcare Strain**: Overwhelmed hospitals and disrupted medical care.

## Lung Impact
- **ARDS**: Severe inflammation and respiratory failure.
- **Pneumonia**: Fluid buildup in alveoli, impairing oxygen exchange.
- **Lung Damage**: Scarring (fibrosis) and long-term respiratory issues.
- **Post-Recovery**: Persistent symptoms and reduced lung function.

COVID-19 has significantly impacted global health and respiratory function, with lasting effects on patients and healthcare systems.
"""

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
    st.markdown(covid_markdown)



if page == "Deep Learning":
    st.write("### Deep Learning")
    st.markdown(deep_learning_markdown)


if page == "Data":
    st.write("### Data")
    st.markdown(data_markdown)

