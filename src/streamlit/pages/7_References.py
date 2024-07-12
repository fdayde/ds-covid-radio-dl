# References for bibliography
# sources for images etc.

import streamlit as st
from functions.footer import add_footer


st.set_page_config(
    page_title="References",
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
  border-top: 2px solid #FF5733;  # You can change the color code as needed
}
</style>
"""

st.title("References & Sources")


references_md = """
DenseNet201 Model: Gao Huang and Zhuang Liu and Laurens van der Maaten and Kilian Q. Weinberger, 2018. Densely Connected Convolutional Networks.
"""

sources_md = """

"""


st.markdown(references_md)


add_footer()