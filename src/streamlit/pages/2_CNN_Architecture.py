import streamlit as st
from streamlit_extras.colored_header import colored_header
from graphviz import Digraph
import os

# Set the page title
st.set_page_config(page_title="CNN Architecture", page_icon=":construction:", layout="wide")

# Title and description
st.title("Convolutional Neural Network Architecture")

# CNN picture
image_path = os.path.join(os.path.dirname(__file__), "..", "pictures", "cnn_architecture.png")
st.image(image_path, caption="Architecture of a CNN.", use_column_width=True)
st.markdown("<font size='1'><div style='text-align: right;'>[image source](https://www.mathworks.com/videos/introduction-to-deep-learning-what-are-convolutional-neural-networks--1489512765771.html)", unsafe_allow_html=True)

# Selection box to choose the architecture
architecture = st.selectbox(
    "Illustration of different CNN architectures:",
    ("Simple CNN", "Simplified DenseNet201")
)

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
        - **ReLU Layer**: Applies the Rectified Linear Unit activation function.\n
        - **Pooling Layer**: Performs max pooling, reducing the spatial dimensions to 14x14.\n
        - **Fully Connected Layer**: Flattens the feature maps and passes through a dense layer with 784 neurons, followed by ReLU activation.\n
        - **Output Layer**: Final classification layer with 10 neurons, representing the output classes.\n
        """)



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



if architecture == "Simple CNN":
    simple_cnn_diagram()
else:
    simplified_densenet201_diagram()
