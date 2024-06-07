import streamlit as st
from pathlib import Path
import base64
import os
from PIL import Image

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
                    padding-left: 25rem;
                    padding-right: 25rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.title("Data Science Project: COVID Lung X-Rays Classification")
st.header(" ", divider = 'rainbow')

base_path = Path.cwd()
images = base_path / 'pictures'
banner_path = str(images / 'banner_streamlit.jpg')
logo_path = str(images / 'logo.PNG')

st.image(banner_path, use_column_width=True)

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