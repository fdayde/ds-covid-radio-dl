import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="DS Project",
    page_icon=":sparkles:",
    layout='wide'
)

st.title("Data Science Project: COVID Lung X-Rays Classification")

base_path = Path.cwd()
streamlit_path = Path.cwd() / 'src' / 'streamlit'
images = streamlit_path / 'pictures'
banner_path = str(images / 'banner_streamlit.jpg')

st.image(banner_path, use_column_width=True)