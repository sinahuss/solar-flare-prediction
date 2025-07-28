# Solar Flare Prediction with Supervised Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sinahuss/solar-flare-prediction/blob/main/notebooks/solar_flare_analysis.ipynb)

## 1. Project Overview

This project is a data science application developed as part of the WGU Computer Science Capstone. It addresses the real-world challenge of space weather forecasting by using supervised machine learning to predict the intensity of solar flares. The model is designed to provide a 24-hour advance warning for a space weather agency like NOAA, helping to protect critical satellite, communication, and power grid infrastructure from the adverse effects of significant solar events.

## 2. Key Features

* **Comprehensive EDA:** Conducts a thorough Exploratory Data Analysis (EDA) on the UCI Solar Flare dataset to understand feature distributions and relationships.
* **Model Comparison:** Trains and evaluates three distinct classification models (**Random Forest**, **XGBoost**, and **SVM**) to identify the optimal algorithm for the task.
* **Robust Evaluation:** Implements a rigorous evaluation process using metrics suitable for imbalanced datasets, including a detailed **classification report** (Precision, Recall, F1-Score) and a **confusion matrix**.
* **Feature Importance Analysis:** Utilizes SHAP (SHapley Additive exPlanations) to interpret the final model's predictions and identify the most influential sunspot characteristics for flare prediction.
* **Interactive GUI:** Includes a simple GUI built with `ipywidgets` for predicting the flare class of a new, user-defined solar active region.

## 3. Tech Stack

* **Language:** Python 3.10
* **Core Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
* **Development Environment:** Jupyter Notebook, Google Colab
* **Model Interpretation:** SHAP

## 4. Methodology

The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology, progressing through the phases of Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.

## 5. Key Results & Performance

The final **Random Forest Classifier** was selected for its superior performance and interpretability. The model's performance on the held-out test set demonstrates its effectiveness in identifying critical flare events.

| Metric        | Overall | C-Class | M-Class | X-Class |
| :------------ | :-----: | :-----: | :-----: | :-----: |
| **Precision** |  0.86   |  0.94   |  0.58   |  0.67   |
| **Recall**    |  0.88   |  0.92   |  0.65   |  0.67   |
| **F1-Score**  |  0.87   |  0.93   |  0.61   |  0.67   |

The model successfully achieved a high recall for the most dangerous X-class flares, meeting a key success criterion for a practical early warning system.

## 6. Installation & Usage

### Prerequisites
- Python 3.10+
- An environment manager like Conda or venv.

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sinahuss/solar-flare-prediction.git](https://github.com/sinahuss/solar-flare-prediction.git)
    cd solar-flare-prediction
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  In the Jupyter interface, navigate to the `notebooks/` folder and open `solar_flare_analysis.ipynb`. The entire analysis can be reproduced by clicking **Cell > Run All**.