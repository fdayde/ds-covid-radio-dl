import pandas as pd
import plotly.graph_objects as go

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


data_accuracy_global = {
    "model": ["VGG19", "VGG19", "MobileNetV2", "MobileNetV2", "DenseNet201", "DenseNet201", "Xception"],
    "masques": [1, 0, 1, 0, 1, 0, 0],
    "Accuracy": [0.92, 0.94, 0.86, 0.93, 0.91, 0.96, 0.96]
}

df_accuracy_global = pd.DataFrame(data_accuracy_global)

data_classification_report = {
    "Class": ["COVID", "Non COVID", "Normal", "COVID", "Non COVID", "Normal",
              "COVID-19", "Non-COVID", "Normal", "COVID-19", "Non-COVID", "Normal",
              "COVID-19", "Normal", "Non-COVID", "COVID-19", "Normal", "Non-COVID",
              "COVID-19", "Normal", "Non-COVID"],
    "Precision": [0.91, 0.91, 0.93, 0.96, 0.91, 0.95, 0.91, 0.85, 0.82, 0.98, 0.89, 0.93,
                  0.93, 0.89, 0.91, 0.99, 0.95, 0.94, 0.99, 0.95, 0.93],
    "Recall": [0.94, 0.92, 0.88, 0.97, 0.94, 0.89, 0.81, 0.88, 0.89, 0.97, 0.94, 0.88,
               0.90, 0.92, 0.92, 0.98, 0.93, 0.96, 0.98, 0.93, 0.96],
    "f1-score": [0.93, 0.92, 0.91, 0.96, 0.93, 0.92, 0.86, 0.86, 0.86, 0.97, 0.92, 0.90,
                 0.91, 0.90, 0.91, 0.99, 0.94, 0.95, 0.99, 0.94, 0.95],
    "Support": [2395, 2253, 2140, 2395, 2253, 2140, 2395, 2253, 2140, 2395, 2253, 2140,
                2395, 2140, 2253, 2395, 2140, 2253, 2395, 2140, 2253],
    "model": ["VGG19", "VGG19", "VGG19", "VGG19", "VGG19", "VGG19",
              "MobileNetV2", "MobileNetV2", "MobileNetV2", "MobileNetV2", "MobileNetV2", "MobileNetV2",
              "DenseNet201", "DenseNet201", "DenseNet201", "DenseNet201", "DenseNet201", "DenseNet201",
              "Xception", "Xception", "Xception"],
    "masques": [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
}

df_clasif_report = pd.DataFrame(data_classification_report)

def create_accuracy_plot(data=df_accuracy_global):
    fig = go.Figure()

    # Add traces for each combination of masques and model
    for masque_value in set(data["masques"]):
        for model_name in set(data["model"]):
            filtered_data = data[(data["masques"] == masque_value) & (data["model"] == model_name)]
            fig.add_trace(go.Bar(x=filtered_data["model"], y=filtered_data["Accuracy"], 
                                    name=f"Masques={masque_value}, Model={model_name}"))

    # Update layout
    fig.update_layout(
        title="Accuracy by Model and Masques",
        xaxis_title="Model",
        yaxis_title="Accuracy",
        hovermode="closest"
    )

    return fig

