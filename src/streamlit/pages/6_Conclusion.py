import streamlit as st
from pathlib import Path
import os



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

# Chemin d'accès pour les images du Streamlit
base_path = Path.cwd()
images = base_path / 'pictures'
banner_path = str(images / 'banner_streamlit.jpg')
logo_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "logo.PNG")

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

improving_model_markdown_hp ="""

### Tuning the model : Hyperparameters

- Raise the DropOut rate

- Add L1 and L2 layers

- Optimize the batch size


"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

improving_model_markdown_data ="""

### Tuning the model : Data

- Data Augmentation

- Data Cleaning


"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

improving_model_markdown_tf ="""

### Tuning the model : Transfer Learning

We only used base models which were natively in Keras. With a little more time and computing power, we could have implemented other models, such as ChexNet.


"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

improving_model_markdown_int ="""

### Interpretability

The lung masks are essential to steer the model on the lungs : it focuses too much on peripheral artefacts without masks. Neverthless, it affects the performance 
negatively.

We also used the masks provided with the dataset : for a complete application, a lung segmentation step should be added.


"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

discussion_markdown_summary_adv="""

:large_green_circle: The chosen model performs well and is equivalent on all three classes 

:large_green_circle: The recall and f1 scores on all images (masked or not) are superior to 90%

:large_green_circle: No overfitting or very few

:large_green_circle: Good interpretability, the features selected are on the lungs for the most part on masked images

"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

discussion_markdown_summary_inc="""

:red_circle: The DenseNet model is not a light model

:red_circle: Without masks, the interpretability with Grad-Cam is not optimal (features selected not on the lung,...)

:red_circle: Need for more customization on the model like custom metrics

:red_circle: Not enought time/resources for the training

"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

discussion_markdown_what_next="""

- Lung Segmentation seems crucial to really target the lungs for the classification : need to create another specific model

- The model needs more training, even more so with the masked images

- Impossible to correctly do Deep Learning without more resources : :X: Kaggle (30h not enough), :X: Google Colab (not enough units), :X: not enough money to buy a Deep Learning
ready rig...

"""

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

st.sidebar.title("Conclusion")
pages = ["Improving The Model", "Discussion"]
page = st.sidebar.radio("Summary", pages, label_visibility="collapsed")

if page == "Improving The Model":
    st.header("Conclusion", divider = 'rainbow')
    st.subheader("What can we do to improve the model", divider = 'grey')
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.subheader("", divider = 'grey')
        st.markdown(improving_model_markdown_hp, unsafe_allow_html=True)
    with row1_col2:
        st.subheader("", divider = 'grey')
        st.markdown(improving_model_markdown_data, unsafe_allow_html=True)
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.subheader("", divider = 'grey')
        for _ in range(4):
            st.write("")
        st.markdown(improving_model_markdown_tf, unsafe_allow_html=True)
    with row2_col2:
        st.subheader("", divider = 'grey')
        for _ in range(4):
            st.write("")
        st.markdown(improving_model_markdown_int, unsafe_allow_html=True)

if page == "Discussion":
    st.header("Discussion", divider = 'rainbow')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pros", divider = 'grey')
        st.markdown(discussion_markdown_summary_adv)
    with col2:
        st.subheader("Cons", divider = 'grey')
        st.markdown(discussion_markdown_summary_inc)
    st.subheader("To go further : What next", divider = 'grey')
    st.markdown(discussion_markdown_what_next)

# Footer
st.header(" ", divider = 'rainbow')
hcol1, hcol2, hcol3 = st.columns([0.2, 0.5, 0.3])

with hcol1:
    st.markdown("""Thomas Baret  
                    Nicolas Bouzinbi  
                     Florent Daydé  
                     Nicolas Fenassile""")
with hcol2:
    st.markdown(" ")

with hcol3:
    st.image(logo_path, use_column_width=True)