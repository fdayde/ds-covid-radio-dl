import streamlit as st
from streamlit_extras.colored_header import colored_header
from graphviz import Digraph
import os

# Set the page title
st.set_page_config(page_title="CNN Architecture", page_icon=":construction:",)

# Title and description
st.title("Convolutional Neural Network Architecture")

# CNN picture
architecture_img_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "cnn_architecture.png")
st.image(architecture_img_path, caption="Architecture of a CNN.", use_column_width=True)
st.markdown("<font size='1'><div style='text-align: right;'>[image source](https://www.mathworks.com/videos/introduction-to-deep-learning-what-are-convolutional-neural-networks--1489512765771.html)", unsafe_allow_html=True)

# Selection box to choose the architecture
architecture = st.selectbox(
    "Illustration of different CNN architectures:",
    ("Simple CNN", "Key Concepts", "Simplified DenseNet201", "Overfitting")
)



# CNN imgages path
convolution_img_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "convolution.gif")
pooling_img_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "pooling.gif")
fullcnn_img_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "full_cnn.gif")
dropout_img_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "dropout.gif")
earlystoping_img_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "early_stopping.webp")
dataaugmentation_img_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "data_augmentation.jpg")
# sources 
# convolution : https://mlnotebook.github.io/post/CNN1/
# pooling : https://towardsdatascience.com/an-introduction-to-convolutional-neural-networks-bdf692352c7
# full cnn : https://adatis.co.uk/wp-content/uploads/CNNsFullGifff.gif
# dropout : https://miro.medium.com/v2/resize:fit:640/format:webp/1*znOtHWgqnEtpXWk2iQcK-Q.gif
# early stopping : https://miro.medium.com/v2/resize:fit:640/format:webp/1*nhmPdWSGh3ziatQKOmVq0Q.png
# data augmentation : https://www.sciencedirect.com/science/article/pii/S174680942100923X#f0010


def key_concepts():
    # Colored header
    colored_header(
        label="CNN Key Concepts",
        description="Visual representation of some CNN key concepts.",
        color_name="red-70",
    )

    # Create a layout with two columns
    col1, col2 = st.columns([1, 2])


    # URLs for the images to be displayed
    image_paths = [
        convolution_img_path,
        pooling_img_path
    ]

    # Render the graph in the first column
    with col1:
        for url in image_paths:
            st.image(url, use_column_width=True)
            for _ in range(26):
                st.write("")

    # Add explanations for each step in the second column
    with col2:
        st.markdown("""
        - **Convolution**:   
            - mathematical operation performed between a small matrix called a **kernel** (or filter) and a subset of the input data,\n
            - the kernel slides over the input data and a dot product is computed at each position,\n
            - the kernel's weights depends on the model, and are updated during the learning phase,\n
            - then a **bias** is added to the result,\n
            - a new matrix, the **feature map** is created, which higlights specific feature of the data.\n
        - **Parameters**:   
            - each convolutional layer has a number of parameters defined by:\n
                - [the size of the kernel] X [the number of channels of the input] X [the number of channels of the output] X the bias,\n
                - the number of parameters is directly linked to the model's capacity to learn complex pattern in the data.\n
        - **Pooling**:    
            - Summary Statistic: replaces network outputs at certain locations with a summary of nearby outputs\n
            - Spatial Size Reduction: It decreases the representation’s spatial size, reducing computation and weights\n
            - Individual Slices: Pooling operates on each representation slice individually\n
            - Max Pooling: The most common pooling method, retaining the maximum value from the local neighborhood.\n
        """)



def simple_cnn_diagram():
    # Create a graph object
    dot = Digraph()

    # Add nodes for each step
    dot.node("Input", "Input Layer\n(32x32x3)")
    dot.node("Conv", "Conv Layer\n(28x28x8)")
    dot.node("ReLU", "ReLU Layer\n(28x28x8)")
    dot.node("Pool", "Pooling Layer\n(14x14x8)")
    dot.node("FC", "Fully Connected Layer\n(784 -> 128)")
    dot.node("Output", "Output Layer\n(128 -> 10)")

    # Add edges to link the nodes
    dot.edge("Input", "Conv")
    dot.edge("Conv", "ReLU")
    dot.edge("ReLU", "Pool")
    dot.edge("Pool", "FC")
    dot.edge("FC", "Output")

    # Display the graph with a colored header
    colored_header(
        label="CNN Architecture Diagram",
        description="Visual representation of the simplified CNN architecture.",
        color_name="blue-70",
    )

    # Create a layout with two columns
    col1, col2 = st.columns([1, 2])

    # Render the graph in the first column
    with col1:
        st.graphviz_chart(dot)

    # Add explanations for each step in the second column
    with col2:
        st.subheader("Explanation:")
        st.markdown("""
        - **Input Layer**: This is the initial layer that takes input images of size 32x32 with 3 channels (RGB).\n
        - **Conv Layer**: Convolutional layer with 8 filters, resulting in feature maps of size 28x28.\n
        - **ReLU Layer**: Applies the Rectified Linear Unit activation function to introduce non linearity.\n
        - **Pooling Layer**: Performs max pooling, reducing the spatial dimensions to 14x14.\n
        - **Fully Connected (or Dense) Layer**: Flattens the feature maps and passes through a dense layer with 784 neurons, followed by ReLU activation.\n
        - **Output Layer**: Final classification layer with 10 neurons, representing the output classes.\n
        """)

    st.image(fullcnn_img_path, use_column_width=True)



def simplified_densenet201_diagram():
    # Create a graph object
    dot = Digraph()

    # Add nodes for DenseNet201 layers
    dot.node("Input", "Input Layer\n(224x224x3)")
    dot.node("Init_Conv", "Initial Conv + Pooling\n(112x112x64)")

    dot.node("DenseBlock1", "Dense Block 1\n(6 Convs + BNs + ReLUs)")
    dot.node("Trans1", "Transition Layer 1\n(56x56x128)")

    dot.node("DenseBlock2", "Dense Block 2\n(12 Convs + BNs + ReLUs)")
    dot.node("Trans2", "Transition Layer 2\n(28x28x256)")

    dot.node("DenseBlock3", "Dense Block 3\n(48 Convs + BNs + ReLUs)")
    dot.node("Trans3", "Transition Layer 3\n(14x14x512)")

    dot.node("DenseBlock4", "Dense Block 4\n(32 Convs + BNs + ReLUs)")
    dot.node("FC", "Fully Connected Layer\n(7x7x1024)")

    dot.node("Output", "Output Layer\n(1000 Classes)")

    # Add edges to link the nodes
    dot.edge("Input", "Init_Conv")
    dot.edge("Init_Conv", "DenseBlock1")
    dot.edge("DenseBlock1", "Trans1")
    dot.edge("Trans1", "DenseBlock2")
    dot.edge("DenseBlock2", "Trans2")
    dot.edge("Trans2", "DenseBlock3")
    dot.edge("DenseBlock3", "Trans3")
    dot.edge("Trans3", "DenseBlock4")
    dot.edge("DenseBlock4", "FC")
    dot.edge("FC", "Output")

    # Display the graph with a colored header
    colored_header(
        label="Simplified DenseNet201 Architecture Diagram",
        description="Visual representation of the simplified DenseNet201 architecture.",
        color_name="green-70",
    )

    # Create a layout with two columns
    col1, col2 = st.columns([1, 2])

    # Render the graph in the first column
    with col1:
        st.graphviz_chart(dot)

    # Add explanations for each step in the second column
    with col2:
        st.subheader("Explanation:")
        st.markdown("""
        - **Input Layer**: Initial input images of size 224x224 with 3 channels (RGB).\n
        - **Initial Conv + Pooling**: Initial convolutional layer with 64 filters, followed by max pooling, resulting in feature maps of size 112x112.\n
        - **Dense Block 1**: Dense block consisting of 6 convolutional layers with batch normalization and ReLU activation.\n
        - **Transition Layer 1**: Transition layer with convolution and pooling, resulting in feature maps of size 56x56.\n
        - **Dense Block 2**: Dense block with 12 convolutional layers.\n
        - **Transition Layer 2**: Transition layer with convolution and pooling, resulting in feature maps of size 28x28.\n
        - **Dense Block 3**: Dense block with 48 convolutional layers.\n
        - **Transition Layer 3**: Transition layer with convolution and pooling, resulting in feature maps of size 14x14.\n
        - **Dense Block 4**: Dense block with 32 convolutional layers.\n
        - **Fully Connected Layer**: Final fully connected layer with 7x7x1024 neurons.\n
        - **Output Layer**: Output layer with 1000 classes for classification.
        """)





def overfitting():
    # Colored header
    colored_header(
        label="Overfitting",
        description="Overfitting solutions.",
        color_name="orange-70",
    )

    # Create a layout with two columns
    col1, col2 = st.columns([1, 2])


    # URLs for the images to be displayed
    image_paths = [
        dropout_img_path,
        earlystoping_img_path,
        dataaugmentation_img_path
    ]

    # Render the graph in the first column
    with col1:
        for url in image_paths:
            st.image(url, use_column_width=True)
            for _ in range(4):
                st.write("")

    # Add explanations for each step in the second column
    with col2:
        st.markdown("""
        - **Dropouts**:   
            - 
        - **Early Stopping**:   
            - 
        - **Data Augmentation**:    
            - 
        """)




if architecture == "Simple CNN":
    simple_cnn_diagram()
elif architecture == "Key Concepts":
    key_concepts()
elif architecture == "Simplified DenseNet201":
    simplified_densenet201_diagram()
else : 
    overfitting()
