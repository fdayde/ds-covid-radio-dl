import streamlit as st
import os
from streamlit_extras.app_logo import add_logo
from functions.footer import add_footer


st.set_page_config(
    page_title="Data Exploration",
    page_icon=":sparkles:",
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

st.title("Data Exploration")
st.sidebar.title("Data Exploration")
pages = ["Dataset Analysis", "Data Visualization"]
page = st.sidebar.radio("Summary", pages, label_visibility="collapsed")

data_markdown = """
We worked on the COVID-QU-Ex Dataset available on [Kaggle](https://www.kaggle.com/datasets/cf77495622971312010dd5934ee91f07ccbcfdea8e2f7778977ea8485c1914df).

This data set is a more complete version of the initially proposed dataset.  
The researchers of Qatar University have created a new dataset containing the previous data,
to which they added new images.  
They compiled all the data in the "COVID-QU-Ex dataset", which consists of 33,920 chest X-ray (CXR) images including:

- 11,956 COVID-19
- 11,263 Non-COVID infections (Viral or Bacterial Pneumonia)
- 10,701 Normal

We also have a separation of the data in 3 sets: Train, Test and Val from the get-go.  
We'll keep this separation when using the ImageGenerator as this approach allows us to:
 - Train the model with the Training set: Utilize the training data to fit the model.
 - Validate the model with the Validation set: Use the validation data to tune hyperparameters and make adjustments, ensuring the model performs well on unseen data.
 - Evaluate the model with the Test set: Conduct the final assessment of the model's performance to ensure an unbiased evaluation.
This practice is standard in machine learning and deep learning, providing a robust framework for model development and assessment.


"""

path_images = os.path.join(os.path.dirname(__file__), "..", "pictures")


if page == "Dataset Analysis":
    st.header("Dataset Analysis", divider = 'rainbow')
    
    column1, column2, column3, column4, column5, column6 = st.columns([0.40, 0.04, 0.14, 0.14, 0.14, 0.14])

    with column1:
        st.subheader("COVID-QU-Ex-Dataset", divider = 'grey')
        st.markdown(data_markdown)
        st.subheader("Samples from the dataset", divider = 'grey')
        st.markdown("""On the right are shown several raw images with their associated masks, for each label.""")
        st.markdown(" ")
        st.markdown("""We can see that radios are vastly different from each other.  
                    Some are very dark, others very light, we can find annotations on certain radios, and even rotated/dezoomed ones.  
                    We can see some probes as well.  
                    We can see a lot of "unwanted" information, that could potentially mislead a deep learning model.
                    This is the reason why we will be investigating both full images and masked images for a deep learning model.""")  
        st.markdown(" ")
        st.markdown("""A preprocessing will be mandatory to smooth the data and get a better model.""")

    with column2:
        st.markdown(" ")

    with column3:
        st.markdown("Label: COVID")
        image_path1 = os.path.join(path_images, "data_exploration", "COVID-19", "images", "covid_200.png")
        st.image(image_path1, caption="image covid_200.png")
        st.markdown("Label: Non-COVID")
        image_path2 = os.path.join(path_images, "data_exploration", "Non-COVID", "images", "non_COVID (170).png")
        st.image(image_path2, caption="image non_covid (170).png")
        st.markdown("Label: Normal")
        image_path3 = os.path.join(path_images, "data_exploration", "Normal", "images", "Normal (3935).png")
        st.image(image_path3, caption="Normal (3935).png")

    with column4:
        st.markdown("""Mask associated""")
        mask_path1 = os.path.join(path_images, "data_exploration", "COVID-19", "lung masks", "covid_200.png")
        st.image(mask_path1, caption="mask covid_200.png")
        st.markdown("""Mask associated""")
        mask_path2 = os.path.join(path_images, "data_exploration", "Non-COVID", "lung masks", "non_COVID (170).png")
        st.image(mask_path2, caption="mask non_covid (170).png")
        st.markdown("""Mask associated""")
        mask_path3 = os.path.join(path_images, "data_exploration", "Normal", "lung masks", "Normal (3935).png")
        st.image(mask_path3, caption="Normal (3935).png")

    with column5:
        st.markdown("Label: COVID")
        image_path4 = os.path.join(path_images, "data_exploration", "COVID-19", "images", "covid_11.png")
        st.image(image_path4, caption="image covid_11.png")
        st.markdown("Label: Non-COVID")
        image_path5 = os.path.join(path_images, "data_exploration", "Non-COVID", "images", "non_COVID (764).png")
        st.image(image_path5, caption="image non_covid (764).png")
        st.markdown("Label: Normal")
        image_path6 = os.path.join(path_images, "data_exploration", "Normal", "images", "Normal (8398).png")
        st.image(image_path6, caption="Normal (8398).png")

    with column6:
        st.markdown("""Mask associated""")
        mask_path4 = os.path.join(path_images, "data_exploration", "COVID-19", "lung masks", "covid_11.png")
        st.image(mask_path4, caption="mask covid_11.png")
        st.markdown("""Mask associated""")
        mask_path5 = os.path.join(path_images, "data_exploration", "Non-COVID", "lung masks", "non_COVID (764).png")
        st.image(mask_path5, caption="mask non_covid (764).png")
        st.markdown("""Mask associated""")
        mask_path6 = os.path.join(path_images, "data_exploration", "Normal", "lung masks", "Normal (8398).png")
        st.image(mask_path6, caption="Normal (8398).png")

if page == "Data Visualization":
    st.header("Data Visualization", divider = 'rainbow')

    col1, col2, col3 = st.columns([0.45, 0.1, 0.45])

    with col1:
        # DataViz - Image Count
        st.subheader("Image count", divider = 'gray')
        data_path1 = os.path.join(path_images, "data_repartition.PNG")
        st.image(data_path1, caption="Repartition of images in each label", use_column_width=True)
        st.markdown("""
                As seen above, with this new dataset, we worked on more data, which usually leads to more accurate models. 
                In our case, we also have access to balanced data, which simplifies the preprocessing and reduce bias in models.  
                We now have 35% of the dataset pertaining to the COVID label, while the original dataset contained only 17% of COVID labels.

                """)
    with col2:
        st.markdown(" ")

    with col3:
        # DataViz - Intensity Mean
        st.subheader("Mean intensity", divider = 'grey')
        data_path2 = os.path.join(path_images, "mean_intensity.PNG")
        st.image(data_path2, caption="Mean intensity of images and masks in each label", use_column_width=True)
        st.markdown("""
                    Masks have roughly the same intensity spread for each label, but we can see that COVID images are more intense in average.  
                    This tends to show that COVID-infected lungs have clear light areas compared to healthy lungs and other infections.
                    """)
        

add_footer()