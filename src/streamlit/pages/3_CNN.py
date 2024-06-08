import streamlit as st
import os
from functions.footer import add_footer


st.set_page_config(page_title="CNN", layout="wide")

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


# CNN images path
input_img_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "input_data.webp")
relu_img_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "relu.png")
flatten_img_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "flatten.webp")
dense_img_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "dense.webp")
architecture_img_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "cnn_architecture.png")
convolution_img_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "convolution.gif")
pooling_img_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "pooling.gif")

# image sources :
# https://medium.com/@raycad.seedotech/convolutional-neural-network-cnn-8d1908c010ab
# https://www.quora.com/What-is-the-ReLU-layer-in-CNN
# https://pub.mdpi-res.com/electronics/electronics-09-02152/article_deploy/html/images/electronics-09-02152-g013.png?1608039755
# https://av-eks-lekhak.s3.amazonaws.com/media/__sized__/article_images/fully_connected_layer-thumbnail_webp-600x300.webp
# https://www.mathworks.com/videos/introduction-to-deep-learning-what-are-convolutional-neural-networks--1489512765771.html
# https://mlnotebook.github.io/post/CNN1/
# https://towardsdatascience.com/an-introduction-to-convolutional-neural-networks-bdf692352c7

# CNN steps text :
home_text = "CNN Steps"
input_long_txt = """1 - **Input Data**: \n\nThe input data, such as an image, is represented as a 3D matrix (height, width, depth). 
\n\nFor a color image, the depth is 3, corresponding to the RGB channels."""
convolution_long_txt = """\n\n2 - **Convolution**: 
\n\nA kernel or filter (a small matrix with weights) slides across the input image to extract features such as patterns, shapes, or edges. 
\n\nThis is done by performing element-wise multiplication and summing the results to produce a feature map. 
\n\nEach neuron in the convolutional layer has its own set of weights (i.e., its own filter).
\n\nEach convolutional layer has a number of parameters defined by: [the size of the kernel] X [the number of channels of the input] X [the number of channels of the output] X the bias,
\n\nThe number of parameters is directly linked to the model's capacity to learn complex pattern in the data.
\n\nThe filter weights are adjusted and updated through gradient backpropagation, allowing the model to learn from errors and improve its performance."""
activation_long_txt = """\n\n3 - **Activation Function**: 
\n\nAn activation function, often a Rectified Linear Unit (ReLU), is applied to the feature map. 
\n\nReLU replaces negative values with zero and keeps positive values unchanged, introducing non-linearity to the model.
\n\nThis non-linearity allows the model to learn complex patterns and relationships within the data."""
pooling_long_txt = """\n\n4 - **Pooling**: \n\nThe resulting feature maps are reduced in their dimensions with a pooling layer, commonly max pooling. 
\n\nPooling summarizes the features by taking the maximum (or average) value in a specified window, reducing the spatial dimensions (height and width) 
while retaining the most important features."""
flatten_long_txt = """\n\n5 - **Flatten**: 
\n\nAfter several convolutional and pooling layers, the resulting feature maps are flattened into a single vector. 
\n\nThis transforms the 2D feature maps into a 1D vector, preparing it for the fully connected layers."""
dense_long_txt = """\n\n6 - **Dense (Fully Connected) Layers**: 
\n\nThe flattened vector is fed into one or more dense layers (also known as fully connected layers). 
\n\nEach neuron in these layers is connected to every neuron in the previous layer, allowing the network to learn complex patterns and relationships."""
output_long_txt = """\n\n7 - **Output Layer**: 
\n\nThe final dense layer produces the output. 
\n\nFor classification tasks, this output layer typically uses a softmax activation function to generate probabilities for each class, 
indicating the likelihood of the input image belonging to each class."""

input_short_txt = "1 - Input data: a 3D matrix (height, width, and color channels RGB) for a picture."
convolution_short_txt = "\n\n2 - Convolution: kernels slide across the image to extract features like edges and textures, creating feature maps."
activation_short_txt = "\n\n3 - Activation function (e.g., ReLU): is applied to introduce non-linearity by setting negative values to zero."
pooling_short_txt = "\n\n4 - Pooling: Feature maps are downsampled using pooling layers, reducing their dimensions while retaining important features."
flatten_short_txt = "\n\n5 - Flatten: 2D feature maps are flattened into a 1D vector."
dense_short_txt = "\n\n6 - Dense: the 1D vector is passed through fully connected layers to learn complex patterns."
output_short_txt = "\n\n7 - Output: the final layer that outputs probabilities for each class, providing the classification result."

texts = [
    home_text,
    input_long_txt,
    input_short_txt + convolution_long_txt,
    input_short_txt + convolution_short_txt + activation_long_txt,
    input_short_txt + convolution_short_txt + activation_short_txt + pooling_long_txt,
    input_short_txt + convolution_short_txt + activation_short_txt + pooling_short_txt + flatten_long_txt,
    input_short_txt + convolution_short_txt + activation_short_txt + pooling_short_txt + flatten_short_txt + dense_long_txt,
    input_short_txt + convolution_short_txt + activation_short_txt + pooling_short_txt + flatten_short_txt + dense_short_txt + output_long_txt,
    input_short_txt + convolution_short_txt + activation_short_txt + pooling_short_txt + flatten_short_txt + dense_short_txt + output_short_txt
]

cnn_concepts = [
    architecture_img_path,
    input_img_path,
    convolution_img_path,
    relu_img_path,
    pooling_img_path,
    flatten_img_path,
    dense_img_path,
    dense_img_path,
    architecture_img_path
]



# Title and description
st.title("Convolutional Neural Network")
st.header("CNN Architecture", divider = 'rainbow')

n_max = len(cnn_concepts)

# Initialize session state indexex
if 'index' not in st.session_state:
    st.session_state['index'] = 0

col1, col2 = st.columns(2)

# Display the text in the first column
col1.markdown(texts[st.session_state['index']])

# Display the image in the second column
col2.image(cnn_concepts[st.session_state['index']])

# Add next and prev buttons to the sidebar
if st.sidebar.button('Next'):
    if st.session_state.index >= n_max - 1:
        st.session_state.index = n_max - 1
    else:
        st.session_state.index += 1

if st.sidebar.button('Prev'):
    if st.session_state.index <= 0:
        st.session_state.index = 0
    else:
        st.session_state.index -= 1


add_footer()