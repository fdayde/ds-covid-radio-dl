import streamlit as st

def table_markdown():
    table_md = """
| Model             | Data                                      | Accuracy after Fine Tuning           | Limitations                                               |
|-------------------|-------------------------------------------|--------------------------------------|-----------------------------------------------------------|
| **Xception**      | normalized images (HE)                    | 96.1%                                | Interpretability: sometimes focuses on annotations        |
| _2017, 132 layers_| normalized images HE and masked           | 90.3%                                | Pronounced overfitting                                    |
| **DenseNet201**   | normalized images CLAHE                   | 95.9%                                | Interpretability: sometimes focuses on annotations        |
| _2017, 713 layers_| normalized images CLAHE and masked        | 91.1%                                |                                                           |
| **MobileNetV2**   | normalized images CLAHE (hyperparameters on masked images) | 93.3%               | Overfitting                                               |
| _2018, 53 layers_ | normalized images CLAHE and masked        | 85.8%                                | Interpretability: focuses on areas that are too large and sometimes on annotations |
| **VGG19**         | normalized images CLAHE                   | 93.7%                                | Slight overfitting                                        |
| _2015, 19 layers_ | normalized images CLAHE and masked        | 91.6%                                | Slight overfitting                                        |
"""
    return table_md

def model_choice():
    ccl_md = """The **DenseNet201** has the best treade-off between performance and interpretability."""
    return ccl_md

def choose_best_model():
    return table_markdown(), model_choice()
