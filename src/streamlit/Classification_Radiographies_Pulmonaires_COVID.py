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


add_footer()