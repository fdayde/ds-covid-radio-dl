import streamlit as st
from pathlib import Path
import os
from functions.footer import add_footer


st.set_page_config(
    page_title="Conclusion",
    page_icon=":sparkles:",
    layout="wide"
)

style = """
<style>
.title {
  font-size: 24px;
  font-weight: bold;
  color: white;
}
hr {
  border-top: 2px solid #FF5733;
}
</style>
"""


improving_model_markdown_hp = """
- Raise the DropOut rate

- Add L1 and L2 layers

- Optimize the batch size
"""

improving_model_markdown_data = """

- Data Augmentation

- Data Cleaning
"""

improving_model_markdown_tf = """
We only used base models which were natively in Keras.  
With a little more time and computing power, we could have implemented other models, such as ChexNet.
"""

improving_model_markdown_int = """
The lung masks are essential to steer the model on the lungs: it focuses too much on peripheral artefacts without masks. Neverthless, it affects the performance 
negatively.  
We also used the masks provided with the dataset: for a complete application, a lung segmentation step should be added.
"""

discussion_markdown_summary_adv = """
:large_green_circle: The chosen model performs well and is equivalent on all three classes 

:large_green_circle: The recall and f1 scores on all images (masked or not) are superior to 90%, same numbers as PCR

:large_green_circle: No or very little overfitting

:large_green_circle: Good interpretability, the features selected are on the lungs for the most part on masked images
"""

discussion_markdown_summary_inc = """
:red_circle: The DenseNet model is not a light model

:red_circle: Without masks, the interpretability with Grad-Cam is not optimal (features selected not on the lung...)

:red_circle: Need for more customization on the model like custom metrics

:red_circle: Not enough time/resources for the training
"""

discussion_markdown_what_next = """
- Lung Segmentation seems crucial to really target the lungs for the classification: need to create another specific model

- The model needs more training, even more so with the masked images

- Impossible to correctly do Deep Learning without more resources: :x: Kaggle (30h not enough), :x: Google Colab (not enough units), :x: too expensive to buy a Deep Learning
ready rig...

- Tensorflow vs Pytorch 
"""


st.sidebar.title("Conclusion")
pages = ["Improving The Model", "Discussion"]
page = st.sidebar.radio("Summary", pages, label_visibility="collapsed")
st.title("Conclusion")


if page == "Improving The Model":
    st.header("Improving The Model", divider='rainbow')
    st.markdown("Although we achieved a good accuracy and an interpretable model, they are still ways to improve our model further:")
    for _ in range(3):
        st.write("")
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("Tuning the model: Hyperparameters", divider='grey')
        st.markdown(improving_model_markdown_hp, unsafe_allow_html=True)
        for _ in range(3):
            st.write("")

    with row1_col2:
        st.subheader("Improve the model: Data", divider = 'grey')
        st.markdown(improving_model_markdown_data, unsafe_allow_html=True)
        for _ in range(3):
            st.write("")
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("Tuning the model: Transfer Learning", divider='grey')
        
        st.markdown(improving_model_markdown_tf, unsafe_allow_html=True)

    with row2_col2:
        st.subheader("Interpretability", divider='grey')
        st.markdown(improving_model_markdown_int, unsafe_allow_html=True)


if page == "Discussion":
    st.header("Conclusion", divider='rainbow')
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pros", divider='grey')
        st.markdown(discussion_markdown_summary_adv)

    with col2:
        st.subheader("Cons", divider='grey')
        st.markdown(discussion_markdown_summary_inc)
        for _ in range(3):
            st.write("")

    st.subheader("To go further: What next", divider='grey')
    st.markdown(discussion_markdown_what_next)


add_footer()