
import streamlit as st

st.set_page_config(
    page_title="DS Project",
    page_icon=":sparkles:",
)

st.title("Data Science Project: COVID-19 radiographies classification")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)