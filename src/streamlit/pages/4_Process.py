from functions.choosing_best_model import df_clasif_report, choose_best_model, create_accuracy_plot, plot_model_similarity_graph, knn_model_similarity
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
    st.header("Images Normalization", divider = 'rainbow')
    st.markdown('''In image processing, normalization is a technique used to adjust the range and distribution of pixel intensity values to enhance the visual quality and performance of image analysis tasks.  
                For our datasets, during the development, we have chosen to test different normalization methods''')
    st.subheader("Min-Max Normalization", divider = 'gray')
    st.markdown('''
Min-Max Normalization scales the pixel values of an image to a fixed range, typically [0, 1] or [0, 255]. This method ensures that the minimum and maximum pixel values in the image correspond to the minimum and maximum values of the desired range, respectively.

**Formula:**

''')
    st.latex(r'''
        I_{norm}(x, y) = \frac{I(x, y) - I_{min}}{I_{max} - I_{min}} 
        ''')
    st.markdown(r'''
- I(x, y) is the original pixel value.
- Imin and Imax are the minimum and maximum pixel values in the original image.


**Benefits:**
- Simple to implement.
- Ensures that all pixel values are within a specific range, which can be useful for certain machine learning algorithms.''')

    st.subheader("Histogram Normalization", divider = 'gray')
    st.markdown('''
Histogram Normalization, also known as Histogram Equalization, adjusts the contrast of an image by modifying its intensity distribution. This method spreads out the most frequent intensity values, enhancing the global contrast of the image.

**Procedure:**
1. Compute the histogram of the image.
2. Calculate the cumulative distribution function (CDF) of the histogram.
3. Use the CDF to map the original pixel values to the new intensity values.

**Benefits:**
- Improves the contrast of an image, making features more distinguishable.
- Useful for images with backgrounds and foregrounds that are both bright or both dark.''')

    st.subheader("Contrast Limited Adaptive Histogram Equalization (CLAHE)", divider = 'gray')
    st.markdown(''')
CLAHE is a variant of histogram equalization that operates on small regions in the image called tiles. It limits the amplification of noise by clipping the histogram at a predefined value (clip limit). The contrast in each tile is enhanced, and then the tiles are combined using bilinear interpolation.

**Procedure:**
1. Divide the image into non-overlapping tiles.
2. Apply histogram equalization to each tile independently.
3. Clip the histogram of each tile at the clip limit to reduce noise.
4. Interpolate the boundaries of the tiles to eliminate artificial boundaries.

**Benefits:**
- Enhances local contrast while preventing noise amplification.
- Suitable for medical images and other scenarios where local contrast enhancement is critical.''')
    image_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "preprocessing_comparison.png")
    st.image(image_path, caption="Comparison of normalization methods - images/pixel intensity repartition", use_column_width=True)
    
    st.subheader("Conclusion", divider = 'gray')
    st.markdown(r'''
We chose to normalize the images with CLAHE because it provides a balanced approach to enhancing local contrast without amplifying noise excessively. This method is particularly beneficial for our dataset, which contains images where fine details are crucial for analysis. Min-Max Normalization and Histogram Normalization also play significant roles in different contexts, providing global contrast enhancement and fixed range scaling, respectively. By using these normalization techniques, we aim to improve the performance and accuracy of our image analysis tasks.
    ''')


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

    st.write("### Models performances summary")
    table_md, conclusion_md = choose_best_model()
    st.markdown(table_md)
    st.markdown("<br>", unsafe_allow_html=True) 
    st.markdown(conclusion_md)
    
    # Display the Plotly figure
    fig = create_accuracy_plot()
    st.plotly_chart(fig)

    st.write("### Classification Report Metrics")
    # similarity graph
    nb_neighbors = st.number_input("Number of Neighbors for KNN", min_value=1, max_value=10, value=3, step=1) # nb of neighbors for knn
    neighbors = knn_model_similarity(nb_neighbors=nb_neighbors)
    st.pyplot(plot_model_similarity_graph(neighbors, nb_neighbors=nb_neighbors))
    # Display the DataFrame
    st.write(df_clasif_report)



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