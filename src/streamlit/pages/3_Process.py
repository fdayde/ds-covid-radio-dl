import streamlit as st
import os

st.set_page_config(
    page_title="Modeling Process",
    page_icon=":page_facing_up:", layout="wide"
)


# Main title
st.title("Modeling Process")


# Step 1: Model Choice
if st.checkbox("Step 1: Models choice"):
    st.write("### Step 1: Choosing the right models")
    st.markdown("Based on the literature and resource limits we chose the following models...")

# Step 2: Data Processing
if st.checkbox("Step 2: Image pre-processing"):
    st.write("### Step 2: Image pre-processing")
    
    st.markdown('''
    ## Image Normalization Methods

In image processing, normalization is a technique used to adjust the range and distribution of pixel intensity values to enhance the visual quality and performance of image analysis tasks. For our datasets, during the development, we have chosen to test the following normalization methods:

### 1. Min-Max Normalization

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
- Ensures that all pixel values are within a specific range, which can be useful for certain machine learning algorithms.

### 2. Histogram Normalization

Histogram Normalization, also known as Histogram Equalization, adjusts the contrast of an image by modifying its intensity distribution. This method spreads out the most frequent intensity values, enhancing the global contrast of the image.

**Procedure:**
1. Compute the histogram of the image.
2. Calculate the cumulative distribution function (CDF) of the histogram.
3. Use the CDF to map the original pixel values to the new intensity values.

**Benefits:**
- Improves the contrast of an image, making features more distinguishable.
- Useful for images with backgrounds and foregrounds that are both bright or both dark.

### 3. Contrast Limited Adaptive Histogram Equalization (CLAHE)

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
    st.image(image_path, caption="Comparison of normalization methodes - images/pixel intensity repartition", use_column_width=True)
    
    st.markdown(r'''

### Conclusion

We chose to normalize the images with CLAHE because it provides a balanced approach to enhancing local contrast without amplifying noise excessively. This method is particularly beneficial for our dataset, which contains images where fine details are crucial for analysis. Min-Max Normalization and Histogram Normalization also play significant roles in different contexts, providing global contrast enhancement and fixed range scaling, respectively. By using these normalization techniques, we aim to improve the performance and accuracy of our image analysis tasks.
    ''')


# Step 3: Transfer learning
if st.checkbox("Step 3: Transfer Learning"):
    st.write("### Step 3: Transfer Learning")
    st.markdown("""
### Fine-Tuning a Pre-trained Model: DenseNet201 Example

Fine-tuning a pre-trained model is a powerful technique in transfer learning, allowing a model trained on a large dataset to be adapted for a specific task with a smaller dataset. Transfer learning leverages the knowledge gained from one task and applies it to a different but related task, significantly improving performance and reducing the time required for training.

Let's take DenseNet201 as an example to illustrate this process. DenseNet201, part of the Dense Convolutional Network (DenseNet) family, is pre-trained on a vast dataset like ImageNet. The process of fine-tuning involves adjusting this pre-trained model to better suit a new, often more specialized dataset.

Here’s a step-by-step outline of the fine-tuning process with DenseNet201, specifically focusing on unfreezing the last two convolutional blocks:

1. **Load the Pre-trained Model**: First, we load the DenseNet201 model with weights pre-trained on a large benchmark dataset such as ImageNet. This provides a robust starting point as the model has already learned to detect a wide variety of features.

2. **Freezing Layers**: Initially, all the layers of DenseNet201 are frozen to prevent their weights from being updated during the training process. This ensures that the pre-trained knowledge is retained.

3. **Replace the Top Layers**: Since the original DenseNet201 model is designed for classification over a different number of classes, the top (fully connected) layers are replaced to match the number of classes in the new dataset.

4. **Unfreeze Specific Layers**: To adapt the model to the new task, we selectively unfreeze the last two convolutional blocks. Unfreezing these layers allows them to learn and fine-tune to the new dataset while keeping the earlier layers frozen to preserve their general feature detection capabilities.

5. **Compile the Model**: The model is then compiled with a suitable optimizer (such as Adam or SGD) and a loss function appropriate for the new task, often categorical cross-entropy for classification problems.

6. **Training**: The model is trained on the new dataset, with the unfreezed layers adjusting their weights to learn task-specific features. During this stage, a lower learning rate is typically used to prevent large updates that could disrupt the pre-trained weights significantly.

7. **Evaluation and Iteration**: After training, the model’s performance is evaluated. Depending on the results, further fine-tuning may be done, possibly by unfreezing more layers or adjusting hyperparameters.

By following this process, DenseNet201 can be fine-tuned to effectively perform on a new, specialized task, leveraging both the extensive pre-trained knowledge and the specific features of the new dataset. Fine-tuning thus strikes a balance between preserving learned features and adapting to new ones, making it a versatile approach in deep learning and transfer learning.
""")

# Step 4: Training
if st.checkbox("Step 4: Training"):
    st.write("### Step 4: Training")
    st.markdown("")

# Step5: Model Evaluation
if st.checkbox("Step 5: Model Evaluation"):
    st.write("### Step 5: Model Evaluation")
    st.markdown("")

# Step 6: Choosing the Best Model
if st.checkbox("Step 6: Choosing the Best Model"):
    st.write("### Step 6: Choosing the Best Model")
    st.markdown("")