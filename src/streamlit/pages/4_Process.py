import streamlit as st
import os

st.set_page_config(
    page_title="Modeling Process",
    page_icon=":page_facing_up:",
    layout="wide"
)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 10rem;
                    padding-right: 10rem;
                }
        </style>
        """, unsafe_allow_html=True)

logo_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "logo.PNG")

# Main title
st.title("Modeling Process")
st.sidebar.title("Modeling Process")
pages = ["Unfolding", "Images pre-processing", "Transfer Learning", "Training", "Interpretability", "Evaluation of models"]
page = st.sidebar.radio("Summary", pages, label_visibility="collapsed")

if page == "Unfolding":
    st.header("Unfolding", divider = 'rainbow')
    col1, col2, col3 = st.columns([0.4, 0.1, 0.5])
    with col1:
        st.markdown("""As we have seen previously, the first step before starting deep learning will be a pre-processing of all the data.       
                Then, once we have chosen a method, we can start by a quick LeNet model to have a first idea of how our dataset is behaving in deep learning.      
                As it takes quite some time to build and train a model from scratch, we will be using transfer learning to accelerate process.
                We will be training several models, which will be evaluated to select the best one. """)
    with col2:
        st.markdown(" ")
    with col3:
        data_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "unfolding.PNG")
        st.image(data_path, caption="https://deepnote.com/app/a_mas/Data-Preprocessing-Tools-4943e322-768c-4961-b30f-c0e8f63bf0ec", use_column_width=True)



if page == "Images pre-processing":
    st.header("Images pre-processing", divider = 'rainbow')
    st.markdown("We chose to normalize the images with CLAHE .....")

    # afficher des images poiur comparer, des intensités avant apres normalisation...
    # + masques ?


if page == "Transfer Learning":
    st.header("Transfer Learning", divider = 'rainbow')
    st.markdown("")


if page == "Training":
    st.header("Training", divider = 'rainbow')
    st.markdown("")


if page == "Interpretability":
    st.header("Evaluation", divider = 'rainbow')
    st.markdown("")


if page == "Evaluation of models":
    st.header("Choosing the best model", divider = 'rainbow')
    st.markdown("")


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