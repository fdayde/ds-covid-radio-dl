import streamlit as st
from pathlib import Path
from functions.footer import add_footer




st.set_page_config(
    page_title="DS Project",
    page_icon=":sparkles:",
    layout='wide'
)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 3rem;
                    padding-left: 22rem;
                    padding-right: 22em;
                }
        </style>
        """, unsafe_allow_html=True)



st.title("Data Science Project: COVID Lung X-Rays Classification")
st.header(" ", divider = 'rainbow')


base_path = Path.cwd()
images = base_path / 'pictures'
banner_path = str(images / 'banner_streamlit.jpg')

st.image(banner_path, use_column_width=True)


context_md = """
This project was completed during the Data Scientist Bootcamp at [Datascientest](https://datascientest.com/) from March to June 2024.  

Using the [COVID-QU-Ex Dataset](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu) from Kaggle, the goal was to build a deep learning model to classify chest X-ray into 3 categories: COVID_19, Non-COVID infections or Normal.  

This version has been forked from the original project: 
- [original](https://github.com/DataScientest-Studio/MAR24_BDS_Radios_Pulmonaire)
- [forked version](https://github.com/fdayde/ds-covid-radio-dl)

"""

st.markdown(context_md)

add_footer()