import streamlit as st
from functions.choosing_best_model import choose_best_model  # Import the function

st.set_page_config(
    page_title="Modeling Process",
    page_icon=":page_facing_up:",
)


# Main title
st.title("Modeling Process")


# Step 1: Model Choice
if st.checkbox("Step 1: Models choice"):
    st.write("### Step 1: Choosing the right models")
    st.markdown("Based on the literature and resource limits we chose the following models...")

# Step 2: Data Processing
if st.checkbox("Step 2: Image pre-processing"):
    st.write("### Step 2: Image pre-procesing")
    st.markdown("We chose to normalize the images with CLAHE .....")

    # afficher des images poiur comparer, des intensités avant apres normalisation...
    # + masques ?


# Step 3: Transfer learning
if st.checkbox("Step 3: Transfer Learning"):
    st.write("### Step 3: Transfer Learning")
    st.markdown("")

# Step 4: Tuning
if st.checkbox("Step 4: Tuning"):
    st.write("### Step 4: Tuning")
    st.markdown("")

# Step 5: Training
if st.checkbox("Step 5: Training"):
    st.write("### Step 5: Training")
    st.markdown("")

# Step 6: Fine Tuning
if st.checkbox("Step 6: Fine Tuning"):
    st.write("### Step 6: Fine Tuning")
    st.markdown("")

# Step 7: Model Evaluation
if st.checkbox("Step 7: Model Evaluation"):
    st.write("### Step 7: Model Evaluation")
    st.markdown("")

# Step 8: Choosing the Best Model
if st.checkbox("Step 8: Choosing the Best Model"):
    st.write("### Step 8: Choosing the Best Model")
    table_md, conclusion_md = choose_best_model()
    st.markdown(table_md)
    st.markdown("<br>", unsafe_allow_html=True) 
    st.markdown(conclusion_md)
