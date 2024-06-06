import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np

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
    ccl_md = """The **DenseNet201** has the best trade-off between performance and interpretability."""
    return ccl_md

def choose_best_model():
    return table_markdown(), model_choice()


data_accuracy_global = {
    "model": ["VGG19", "VGG19", "MobileNetV2", "MobileNetV2", "DenseNet201", "DenseNet201", "Xception", "Xception"],
    "masked": [1, 0, 1, 0, 1, 0, 1, 0],
    "Accuracy": [0.916, 0.937, 0.858, 0.933, 0.911, 0.959, 0.903, 0.961]
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
    "model": ["VGG19", "VGG19", "VGG19", "VGG19", "VGG19", "VGG19",
              "MobileNetV2", "MobileNetV2", "MobileNetV2", "MobileNetV2", "MobileNetV2", "MobileNetV2",
              "DenseNet201", "DenseNet201", "DenseNet201", "DenseNet201", "DenseNet201", "DenseNet201",
              "Xception", "Xception", "Xception"],
    "masked": [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
}

df_clasif_report = pd.DataFrame(data_classification_report)

def create_accuracy_plot(data=df_accuracy_global):
    fig = go.Figure()

    custom_colors = {
        ("VGG19", 0): "#d96aff",   # Dark blue
        ("VGG19", 1): "#e8a6ff",   # Light blue
        ("MobileNetV2", 0): "#c1a885",   # Dark green
        ("MobileNetV2", 1): "#dacbb6",   # Light green
        ("DenseNet201", 0): "#ce010e",   # Dark red
        ("DenseNet201", 1): "#e2676e",   # Light red
        ("Xception", 0): "#ffc525",   # Dark yellow
        ("Xception", 1): "#ffdc7c"    # Light yellow
    }

    # Add traces for each combination of masked and model
    for masque_value in set(data["masked"]):
        for model_name in set(data["model"]):
            filtered_data = data[(data["masked"] == masque_value) & (data["model"] == model_name)]
            color = custom_colors.get((model_name, masque_value), "blue")
            fig.add_trace(go.Bar(x=filtered_data["model"], y=filtered_data["Accuracy"], 
                                    name=f"masked={masque_value}, Model={model_name}",
                                    marker=dict(color=color)
                                    ))

    fig.update_layout(
        title="Accuracy by Model and masked status",
        xaxis_title="Model",
        yaxis_title="Accuracy",
        hovermode="closest"
    )
    fig.update_yaxes(range=[0.5, 1])

    return fig



def knn_model_similarity(data=df_clasif_report, nb_neighbors=3):
    """
    Calculate the similarity between models using k-Nearest Neighbors.

    Parameters:
    - data (DataFrame): DataFrame containing classification report data.

    Returns:
    - neighbors (dict): Dictionary containing the indices of the nearest neighbors for each model.
    """
    # Relevant columns for k-NN
    features = data[['Precision', 'Recall']]

    # Normalize features
    normalized_features = (features - features.mean()) / features.std()

    # Fit k-NN model
    nn_model = NearestNeighbors(n_neighbors=nb_neighbors, algorithm='auto')
    nn_model.fit(normalized_features)

    # Find nearest neighbors for each model
    neighbors = {}
    for i, model in enumerate(data['model']):
        _, indices = nn_model.kneighbors([normalized_features.iloc[i]])
        neighbors[model] = data.iloc[indices[0]].index.tolist()

    return neighbors



def plot_model_similarity_graph(neighbors, nb_neighbors):
    """
    Plot the similarity between models based on k-Nearest Neighbors.

    Parameters:
    - neighbors (dict): Dictionary containing the indices of the nearest neighbors for each model.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add edges between nearest neighbors
    for model, neighbor_indices in neighbors.items():
        for neighbor_index in neighbor_indices:
            neighbor_model = df_clasif_report.loc[neighbor_index, 'model']
            G.add_edge(model, neighbor_model)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)  # Define the layout
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrowsize=20)
    plt.title(f"""Models similarity based on knn ({nb_neighbors} neighbors) on classification report metrics""")
    return plt.gcf()
