from functions.choosing_best_model import df_clasif_report, df_densenet_classif_report, table_markdown, create_accuracy_plot, plot_model_similarity_graph, knn_model_similarity
import streamlit as st
import os
import xml.etree.ElementTree as ET
from functions.footer import add_footer

st.set_page_config(
    page_title="Modeling Process",
    page_icon=":page_facing_up:",
    layout="wide"
)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 3rem;
                    padding-left: 10rem;
                    padding-right: 10rem;
                }
        </style>
        """, unsafe_allow_html=True)

diagram_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "Training_process.SVG")
overfitting_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "Overfitting.png")
training_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "Training_plot.png")


unfolding_markdown = """
As we have seen previously, the first essential step before delving into deep learning is pre-processing our data. 
This prepares our dataset for effective model training and evaluation.

Once our data is ready, we can start with a quick test using a LeNet model to get an initial understanding of how our dataset performs in deep learning.

Given that building and training a model from scratch is time-consuming, we will leverage transfer learning to accelerate the process. 
By using pre-trained models, we can significantly reduce training time and improve performance.

We will train several models and evaluate each one to select the best performer.
We thus need to select the appropriate metric and ensure the model chosen is interpretable.
This approach ensures we choose the most suitable model for our specific dataset and objectives.
"""

# Main title
st.title("Modeling Process")
st.sidebar.title("Modeling Process")
pages = ["Unfolding", "Images pre-processing", "Transfer Learning", "Training", "Interpretability", "Evaluation of models"]
page = st.sidebar.radio("Summary", pages, label_visibility="collapsed")

if page == "Unfolding":
    st.header("Unfolding", divider = 'rainbow')
    col1, col2, col3= st.columns([0.4, 0.1, 0.5])
    with col1:
        st.markdown(unfolding_markdown)
        st.subheader("Metric selected", divider = 'gray')
        st.markdown("""As you will see during the presentation, we have chosen the accuracy to evaluate the model.  
                    Since we have balanced data, it is a good general metric to ensure our predictions do not shift too much towards one label or another""")
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
    st.subheader("Fine-Tuning a Pre-trained Model: DenseNet201 Example", divider = 'gray')
    st.markdown("""
Fine-tuning is a technique in transfer learning where a pre-trained model is adapted for a new task. Let's take DenseNet201 as an example. Here’s a simplified process focusing on unfreezing the last two convolutional blocks:

- **Load Pre-trained Model**: Use DenseNet201 with pre-trained weights (e.g., ImageNet).""")
    code = """base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))"""
    st.code(code, language = 'python')
    st.markdown("""
- **Freeze Layers**: Initially freeze all layers to retain pre-trained knowledge.
- **Replace Top Layers**: Adjust the top layers to match the new task's classes.""")
    code = """ x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(3, activation='softmax')(x) # COVID, Non-COVID and Normal

model = Model(inputs=base_model.input, outputs=output)"""
    st.code(code, language = 'python')
    st.markdown("""
- **Unfreeze Layers**: Unfreeze the last two convolutional blocks for fine-tuning.""")
    code = """#Extract from the model building function:
for layer in base_model.layers:
        layer.trainable = False
for layer in base_model.layers[137:] # Unfreezing the last two convolutional blocks
        layer.trainable = True   """
    st.code(code, language = 'python')
    st.markdown("""
- **Compile Model**: Use an appropriate optimizer (here, Adam) and loss function (sparse_categorical_crossentropy).
- **Train Model**: Train on the new dataset with a lower learning rate.
- **Evaluate and Adjust**: Assess performance and adjust as needed.""")
    
             


if page == "Training":
    st.header("Training", divider = 'rainbow')
    st.subheader("Principle", divider = 'grey')
    st.markdown("""
The data has already been divided into 3 sets : Train, Validation and Test. The model will use the Train and Validation test for the training.""")
    st.write("")
    col1, col2, col3 = st.columns([1,2,1]) 
    with col2:
        with open(diagram_path, "r") as training_process:
            svg_content = training_process.read()
        st.image(svg_content, use_column_width=False)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Example of Overfitting", divider = 'grey')
        st.image(overfitting_path, caption='Example of overfitting DenseNet201')
    with  col2:
        st.subheader("Training plots", divider = 'grey')
        st.image(training_path, caption='Accuracy and Loss Functions VGG19')


if page == "Interpretability":
    st.header("Evaluation", divider = 'rainbow')
    st.subheader("What is Grad-CAM?", divider = 'gray')
    st.markdown("""
Grad-CAM (Gradient-weighted Class Activation Mapping) helps visualize which parts of an image a CNN focuses on when making a prediction.""")
    st.subheader("Why Use Grad-CAM?", divider = 'gray')
    st.markdown("""
- **Debugging Models**: Check if the model focuses on correct image parts.
- **Understanding Errors**: See why a model made a wrong prediction.
- **Transparency**: Provide interpretable explanations for model predictions.
""")

    st.subheader("Example Visualization", divider = 'gray')

    col1, col2 = st.columns(2)

    with col1:
        st.image(os.path.join(os.path.dirname(__file__), "..", "pictures", "grad_cam_ex_nomask.png"), 
                 caption="Grad-CAM on Non-Masked Image", width=300)
        st.image(os.path.join(os.path.dirname(__file__), "..", "pictures", "grad_cam_ex2_nomask.png"),
                caption="Grad-CAM on Non-Masked Image", width=300)

    with col2:
        st.image(os.path.join(os.path.dirname(__file__), "..", "pictures", "grad_cam_ex_mask.png"), 
                 caption="Grad-CAM on Masked Image", width=300)
        st.image(os.path.join(os.path.dirname(__file__), "..", "pictures", "grad_cam_ex2_mask.png"), 
                 caption="Grad-CAM on Masked Image", width=300)

    st.subheader("Conclusion", divider = 'gray')
    st.markdown("""
Grad-CAM offers a simple yet powerful way to interpret CNN decisions, improving model transparency and trustworthiness.
""")


if page == "Evaluation of models":
    st.header("Choosing the best model", divider = 'rainbow')
    st.subheader("Performances summary", divider = 'gray')
    table_md = table_markdown()
    st.markdown(table_md)
    st.markdown("<br>", unsafe_allow_html=True) 
    
    # Display the Plotly figure
    fig = create_accuracy_plot()
    st.plotly_chart(fig)

    st.markdown("""
:arrow_forward: The **DenseNet201** has the best trade-off between performance and interpretability:   
- 91.1% accuracy on masked images (95.9% on unmasked).  
- particularly good interpretability on masked images. 
""")
    
    # Checkbox to hide/show the DenseNet201 Classification Report
    st.markdown("<br>", unsafe_allow_html=True) 
    if st.checkbox("DenseNet201 Classification Report"):
        st.write(df_densenet_classif_report)

    st.markdown("<br><br>", unsafe_allow_html=True) 
    st.markdown("""
:arrow_forward: DenseNet201 is a complex model with 713 layers, often seen in the literature, either alone or in combination with other models, for medical imaging classification (Chowdhury et al., 2020 ; Bhosale et al. 2023).  
Compared to other CNNs, the dense layer architecture of Densenet201 is designed to improve accuracy with more parameters without performance degradation or overfitting, and benefit from feature reuse, compact representations, and reduced redundancy (Huang et al., 2017).
""")

    # Checkbox to hide/show the Classification Report Metrics section
    st.markdown("<br><br>", unsafe_allow_html=True) 
    if st.checkbox(":gift: Classification Report Metrics & Models' Similarity"):
        st.subheader("Classification Report Metrics", divider = 'gray')
        # similarity graph
        nb_neighbors = st.number_input("Number of Neighbors for KNN", min_value=1, max_value=10, value=3, step=1) # nb of neighbors for knn
        neighbors = knn_model_similarity(nb_neighbors=nb_neighbors)
        st.pyplot(plot_model_similarity_graph(neighbors, nb_neighbors=nb_neighbors))
        # Display the DataFrame
        st.write(df_clasif_report)



add_footer()