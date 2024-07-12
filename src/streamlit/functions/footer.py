import streamlit as st
import os
import base64


current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "..", "pictures", "logo.PNG")
in_bluelogo_path = os.path.join(current_dir, "..", "pictures", "profile_icons", "In-Blue-14@2x.png")
git_blacklogo_path = os.path.join(current_dir, "..", "pictures", "profile_icons", "github-mark.png")

in_blue_logo_base64 = base64.b64encode(open(in_bluelogo_path, "rb").read()).decode("utf-8")
git_blacklogo_base64 = base64.b64encode(open(git_blacklogo_path, "rb").read()).decode("utf-8")

   
def profile(name, linkedin_url=None, github_url=None):
    icon_col1, icon_col2, name_col3 = st.columns([1, 1, 14])

    if linkedin_url:
        with icon_col1:
            st.markdown(f"<a href='{linkedin_url}' target='_blank'><img src='data:image/png;base64,{in_blue_logo_base64}' class='icon' width='16'></a>", unsafe_allow_html=True)
    if github_url:
        with icon_col2:
            st.markdown(f"<a href='{github_url}' target='_blank'><img src='data:image/png;base64,{git_blacklogo_base64}' class='icon' width='16'></a>", unsafe_allow_html=True)
    if name:
        with name_col3:
            st.write(f"**{name}**")

def add_main_footer():
    st.header(" ", divider='rainbow')
    hcol1, hcol2, hcol3 = st.columns([0.5, 0.5, 0.3])

    with hcol1:
        profile(name="Thomas Baret", linkedin_url="https://linkedin.com/in/thomas-baret-080050107", github_url="https://github.com/tom-b974")
        profile(name="Nicolas Bouzinbi", linkedin_url="https://linkedin.com/in/nicolas-bouzinbi-7916481b4", github_url="https://github.com/NicolasBouzinbi")
        profile(name="Florent Daydé", linkedin_url="https://linkedin.com/in/florent-daydé-16431469", github_url="https://github.com/fdayde")
        profile(name="Nicolas Fenassile", linkedin_url="https://linkedin.com/in/nicolasfenassile", github_url="https://github.com/NicoFena")

    with hcol2:
        st.markdown(" ")

    with hcol3:
        st.image(logo_path, use_column_width=True)


def add_footer():
    st.header(" ", divider='rainbow')
    hcol1, hcol2, hcol3 = st.columns([0.7, 0.5, 0.15])

    with hcol1:
        profile(name="*COVID Lung X-Ray Classification Using a Deep Learning Model, 2024.*", github_url="https://github.com/fdayde/ds-covid-radio-dl")

    with hcol2:
        st.markdown(" ")

    with hcol3:
        st.image(logo_path, use_column_width=True)

# Use: at the end each script:
# add_footer()
