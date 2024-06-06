import streamlit as st
from pathlib import Path
import base64

st.set_page_config(
    page_title="DS Project",
    page_icon=":sparkles:",
    layout='wide'
)

st.title("Data Science Project: COVID Lung X-Rays Classification")

base_path = Path.cwd()
images = base_path / 'pictures'
banner_path = str(images / 'banner_streamlit.jpg')

st.image(banner_path, use_column_width=True)

def get_image_as_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded_string}"

# Function to add a footer with names, LinkedIn links, and an image
def add_footer(image_path):
    image_url = get_image_as_base64(image_path)
    footer = f"""
    <style>
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #333;
        color: #f1f1f1;
        text-align: center;
        padding: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .footer .links {{
        flex: 1;
        text-align: center;
    }}
    .footer .image {{
        margin-right: 20px;
    }}
    .footer a {{
        color: #0a66c2;
        margin: 0 10px;
        text-decoration: none;
    }}
    .footer a:hover {{
        text-decoration: underline;
    }}
    </style>
    <div class="footer">
        <div class="links">
            <p>Projet de Data Science sur la classification des radiographies pulmonaires COVID-19 | © 2024</p>
            <p>
                <a href="https://www.linkedin.com/in/thomas-barret/" target="_blank" rel="noopener noreferrer">Thomas Barret</a> |
                <a href="https://www.linkedin.com/in/nicolas-bouzinbi/" target="_blank" rel="noopener noreferrer">Nicolas Bouzinbi</a> |
                <a href="https://www.linkedin.com/in/florent-dayde/" target="_blank" rel="noopener noreferrer">Florent Daydé</a> |
                <a href="https://www.linkedin.com/in/nicolas-fenassile/" target="_blank" rel="noopener noreferrer">Nicolas Fenassile</a>
            </p>
        </div>
        <div class="image">
            <img src="{image_url}" alt="Logo" height="50">
        </div>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

# Path to the uploaded image
logo_path = str(images / 'Logo_Datascientest.png')

# Add the footer with the image URL
add_footer(logo_path)