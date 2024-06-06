import streamlit as st

st.set_page_config(
    page_title="Conclusion",
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

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

improving_model_markdown_hp =""""

## 1. Tuning the model : Hyperparameters

### Raise the DropOut rate

### Add L1 and L2 layers

### Optimize the batch size


"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

improving_model_markdown_data =""""

## 2. Tuning the model : Data

### Data Augmentation

### Data Cleaning


"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

improving_model_markdown_tf =""""

## 3. Tuning the model : Transfer Learning

We only used base models which were natively in Keras. With a little more time and computing power, we could have implemented other models, such as ChexNet


"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

improving_model_markdown_int =""""

## 4. interpretability

The lung masks 


"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

st.sidebar.title("Conclusion")
pages = ["Improving The Model", "Discussion"]
page = st.sidebar.radio("Summary", pages, label_visibility="collapsed")

if page == "Improving The Model":
    st.markdown('<hr>', unsafe_allow_html=True)
    st.write("# Conclusion")
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(covid_markdown_intro, unsafe_allow_html=True)