from functions.choosing_best_model import df_clasif_report, df_densenet_classif_report, table_markdown, create_accuracy_plot, plot_model_similarity_graph, knn_model_similarity
import streamlit as st
import os
from functions.footer import add_footer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

diagram_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "Training_process.svg")
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
        data_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "unfolding.png")
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
    st.markdown('''
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
        # Dummy data for the example
        epochs = list(range(1, 31))
        train_loss = [
    0.7883333333333337,
    0.44083333333333347,
    0.4583333333333334,
    0.749166666666667,
    0.36083333333333345,
    0.3550000000000001,
    0.2583333333333334,
    0.36250000000000004,
    0.21333333333333337,
    0.23250000000000004,
    0.1841666666666667,
    0.29000000000000004,
    0.16500000000000004,
    0.09083333333333343,
    0.10249999999999992,
    0.043333333333333224,
    0.08000000000000007,
    0.012499999999999956,
    0.06666666666666665,
    0.015000000000000013,
    0.04916666666666658,
    0.05499999999999994,
    0.044166666666666576,
    0.060833333333333295,
    0.03166666666666662,
    0.04249999999999998,
    0.015833333333333366,
    0.04083333333333328,
    0.020000000000000018
]
        val_loss = [
    0.4075000000000001,
    0.3958333333333335,
    0.5916666666666669,
    0.2533333333333334,
    0.3141666666666667,
    0.20833333333333337,
    0.3008333333333334,
    0.1725,
    0.1875,
    0.24416666666666675,
    0.19833333333333336,
    0.2583333333333334,
    0.2058333333333333,
    0.2600000000000001,
    0.22666666666666668,
    0.25583333333333336,
    0.2466666666666667,
    0.34916666666666674,
    0.2566666666666667,
    0.3400000000000001,
    0.2650000000000001,
    0.34750000000000014,
    0.27333333333333343,
    0.34416666666666673,
    0.27333333333333343,
    0.35250000000000015,
    0.2758333333333334,
    0.35750000000000004,
    0.2758333333333380
]
        train_accuracy = [
    0.606622487872488,
    0.7840176715176717,
    0.8073523042273044,
    0.7207726957726959,
    0.8544655232155234,
    0.8462686243936246,
    0.9029344248094251,
    0.8469854469854472,
    0.9221218814968818,
    0.9094204781704784,
    0.9318542099792102,
    0.8781574844074846,
    0.9420348232848236,
    0.9730206167706171,
    0.9648237179487182,
    0.9732913201663205,
    0.9736529799029803,
    1.0037400381150385,
    0.9802256583506587,
    1.0035602910602914,
    0.9881561850311854,
    1.0033805440055443,
    0.9893234580734585,
    0.9712166493416496,
    0.9913916320166324,
    1.0030188842688847,
    0.9907679313929317,
    1.0028348059598065,
    0.99148691961192,
    1.002657224532225
]
        val_accuracy = [
    0.8377057345807348,
    0.8889726264726268,
    0.7929357241857244,
    0.8905946812196814,
    0.9286101004851008,
    0.9136542792792796,
    0.9149190055440058,
    0.9576186763686767,
    0.9173466735966739,
    0.9411317567567571,
    0.9356375606375609,
    0.9436547124047128,
    0.9223904192654195,
    0.9412205474705478,
    0.9280665280665283,
    0.9428426022176025,
    0.909864431739432,
    0.940406271656272,
    0.9119391025641028,
    0.9411274255024258,
    0.912205474705475,
    0.9395963270963275,
    0.9156314968814971,
    0.9398648648648652,
    0.9109472453222456,
    0.9396851178101181,
    0.9107631670131673,
    0.9396851178101200,
    0.9107631670131700,
    0.9396851178101210
]

        # Create subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Model loss by epoch", "Model accuracy by epoch"))

        # Plot for loss
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='train loss'), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='val loss'), row=1, col=1)

        # Plot for accuracy
        fig.add_trace(go.Scatter(x=epochs, y=train_accuracy, mode='lines', name='train accuracy'), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=val_accuracy, mode='lines', name='val accuracy'), row=1, col=2)

        # Update layout
        fig.update_layout(title_text="Model Loss and Accuracy by Epoch", width=1000, height=400)

        # Streamlit app
        st.plotly_chart(fig)
    with  col2:
        st.subheader("Training plots", divider = 'grey')
        # Dummy data for the example
        epochs = list(range(1, 26))
        train_loss = [
    0.9156627450980399,
    0.6580705882352945,
    0.5991294117647062,
    0.5588627450980396,
    0.5319294117647063,
    0.5183372549019611,
    0.5060627450980396,
    0.48179607843137295,
    0.4735294117647062,
    0.46659607843137285,
    0.4543294117647062,
    0.44472941176470626,
    0.44179607843137303,
    0.4361960784313731,
    0.4266039215686279,
    0.421003921568628,
    0.41006274509803964,
    0.4124705882352946,
    0.40952941176470625,
    0.3945960784313729,
    0.39166274509803967,
    0.387403921568628,
    0.381796078431373,
    0.3815372549019612,
    0.3799294117647064
]
        val_loss = [
    1.0889960784313735,
    0.7713960784313731,
    0.4684627450980394,
    0.3895294117647061,
    0.46659607843137285,
    0.41566274509803947,
    0.408745098039216,
    0.5031294117647063,
    0.44020392156862787,
    0.45059607843137284,
    0.3329960784313727,
    0.4020627450980394,
    0.37379607843137286,
    0.3268627450980395,
    0.3852627450980396,
    0.4316627450980397,
    0.5780627450980398,
    0.2937960784313729,
    0.3348627450980396,
    0.2919372549019612,
    0.3410039215686279,
    0.3593960784313729,
    0.3244705882352945,
    0.3228627450980397,
    0.29059607843137303
]
        train_accuracy = [
    0.5876543209876541,
    0.7234567901234565,
    0.7506172839506169,
    0.7691358024691355,
    0.7802469135802466,
    0.787654320987654,
    0.7938271604938267,
    0.7999999999999996,
    0.8024691358024687,
    0.807407407407407,
    0.8135802469135798,
    0.8185185185185181,
    0.8197530864197526,
    0.824691358024691,
    0.824691358024691,
    0.8308641975308637,
    0.8345679012345675,
    0.8308641975308637,
    0.8345679012345675,
    0.8395061728395058,
    0.844444444444444,
    0.8432098765432094,
    0.8469135802469131,
    0.8481481481481477,
    0.8469135802469131
]
        val_accuracy = [
    0.15555555555555578,
    0.7024691358024688,
    0.8111111111111107,
    0.8506172839506169,
    0.8049382716049379,
    0.8296296296296293,
    0.8333333333333329,
    0.8123456790123453,
    0.8135802469135798,
    0.835802469135802,
    0.8703703703703699,
    0.8370370370370366,
    0.8641975308641971,
    0.8740740740740736,
    0.8382716049382711,
    0.8259259259259255,
    0.7864197530864194,
    0.8888888888888884,
    0.8691358024691354,
    0.8851851851851847,
    0.8679012345679008,
    0.8567901234567896,
    0.8679012345679008,
    0.8827160493827155,
    0.890123456790123
]

        # Create subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Model loss by epoch", "Model accuracy by epoch"))

        # Plot for loss
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='train loss'), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='val loss'), row=1, col=1)

        # Plot for accuracy
        fig.add_trace(go.Scatter(x=epochs, y=train_accuracy, mode='lines', name='train accuracy'), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=val_accuracy, mode='lines', name='val accuracy'), row=1, col=2)

        # Update layout
        fig.update_layout(title_text="Model Loss and Accuracy by Epoch", width=1000, height=400)

        # Streamlit app
        st.plotly_chart(fig)


if page == "Interpretability":
    st.header("Interpretation", divider = 'rainbow')
    col1, col2, col3 = st.columns([0.35,0.15,0.5])
    with col1:
        st.subheader("What is Grad-CAM?", divider = 'gray')
        st.markdown("""
Grad-CAM (Gradient-weighted Class Activation Mapping) helps visualize which parts of an image a CNN focuses on when making a prediction.""")
        st.subheader("Why Use Grad-CAM?", divider = 'gray')
        st.markdown("""
- **Debugging Models**: Check if the model focuses on correct image parts.
- **Understanding Errors**: See why a model made a wrong prediction.
- **Transparency**: Provide interpretable explanations for model predictions.
""")
        for _ in range(10):
            st.write("")
        st.subheader("Conclusion", divider = 'gray')
        st.markdown("""
Grad-CAM offers a simple yet powerful way to interpret CNN decisions, improving model transparency and trustworthiness.
""")

    with col3:
        st.subheader("Example Visualization", divider = 'gray')
        scol1, scol2 = st.columns([0.5, 0.5])

        with scol1:
            st.image(os.path.join(os.path.dirname(__file__), "..", "pictures", "grad_cam_ex_nomask.png"), 
                    caption="Grad-CAM on Non-Masked Image")
            st.image(os.path.join(os.path.dirname(__file__), "..", "pictures", "grad_cam_ex2_nomask.png"),
                    caption="Grad-CAM on Non-Masked Image")

        with scol2:
            st.image(os.path.join(os.path.dirname(__file__), "..", "pictures", "grad_cam_ex_mask.png"), 
                    caption="Grad-CAM on Masked Image")
            st.image(os.path.join(os.path.dirname(__file__), "..", "pictures", "grad_cam_ex2_mask.png"), 
                    caption="Grad-CAM on Masked Image")



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