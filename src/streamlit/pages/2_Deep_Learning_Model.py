import streamlit as st

st.set_page_config(
    page_title="Context",
    page_icon=":mag:",
)

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import visualkeras
from PIL import Image


# Main title
st.title("Constructing a Simple Deep Learning Model")

# Introduction
st.write("""
In this tutorial, we will build a simple neural network using TensorFlow and Keras.
We'll walk through the following steps:
1. Defining the model architecture
2. Compiling the model
3. Summarizing the model
4. Visualizing the model
""")



# Initialize model
model = Sequential()
is_model_defined = False

# Step 1: Define the model architecture
if st.checkbox("Step 1: Define the model architecture"):
    st.write("### Step 1: Define the Model Architecture")
    model = Sequential()
    model.add(Dense(32, input_shape=(784,), activation='relu'))
    model.add(Dense(10, activation='softmax'))
    is_model_defined = True
    
    st.code("""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model
model = Sequential()
model.add(Dense(32, input_shape=(784,), activation='relu'))
model.add(Dense(10, activation='softmax'))
""", language='python')
    st.write("We've created a simple neural network with one hidden layer of 32 neurons and an output layer of 10 neurons.")

# Step 2: Compile the model
if st.checkbox("Step 2: Compile the model"):
    st.write("### Step 2: Compile the Model")
    if is_model_defined:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        st.code("""
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
""", language='python')
        st.write("We've compiled the model using the Adam optimizer and sparse categorical crossentropy loss.")
    else:
        st.error("Please complete Step 1 first.")

# Step 3: Summarize the model
if st.checkbox("Step 3: Summarize the model"):
    st.write("### Step 3: Summarize the Model")
    if is_model_defined:
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        summary_str = "\n".join(model_summary)
        st.text(summary_str)
        
        st.write("The summary of our model shows the architecture and the number of parameters.")
    else:
        st.error("Please complete Step 1 and Step 2 first.")

# Step 4: Visualize the model
if st.checkbox("Step 4: Visualize the model"):
    st.write("### Step 4: Visualize the Model")
    if is_model_defined:
        # Ensure the model is built by passing dummy input
        model.build(input_shape=(None, 784))
        
        # Check the input and output shapes of each layer
        layer_shapes = []
        for layer in model.layers:
            try:
                layer_shapes.append(f"{layer.name} - Input Shape: {layer.input_shape}, Output Shape: {layer.output_shape}")
            except AttributeError as e:
                layer_shapes.append(f"{layer.name} - Error: {e}")
        
        st.write("### Layer Shapes")
        for shape in layer_shapes:
            st.write(shape)
        
        # Use visualkeras to visualize the model
    #     try:
    #         img = visualkeras.layered_view(model, legend=True)  # Generate the visualization
    #         st.image(img, caption='Model Architecture')
    #         st.write("The visualization shows the architecture of our neural network, including the layers and their connections.")
    #     except Exception as e:
    #         st.error(f"Error visualizing the model: {e}")
    # else:
    #     st.error("Please complete Step 1 first.")
