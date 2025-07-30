# Solar Flare Prediction with Supervised Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sinahuss/solar-flare-prediction/blob/main/notebooks/solar_flare_analysis.ipynb)

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [Tech Stack](#3-tech-stack)
4. [Methodology](#4-methodology)
5. [Key Results & Performance](#5-key-results--performance)
6. [Installation & Usage](#6-installation--usage)
7. [Repository Structure](#7-repository-structure)
8. [Data Documentation](#8-data-documentation)
9. [Citation](#9-citation)

## 1. Project Overview

This project is a data science application developed as part of the WGU Computer Science Capstone. It addresses the real-world challenge of space weather forecasting by using supervised machine learning to predict the intensity of solar flares. The model is designed to provide a 24-hour advance warning for a space weather agency like NOAA, helping to protect critical satellite, communication, and power grid infrastructure from the adverse effects of significant solar events.

## 2. Key Features

* **Advanced Imbalance Handling:** Implements SMOTEENN (SMOTE + Edited Nearest Neighbors) to address extreme class imbalance where X-class flares represent only 1% of observations.
* **Comprehensive EDA:** Conducts thorough Exploratory Data Analysis (EDA) on the UCI Solar Flare dataset (via Kaggle) with multi-dimensional risk analysis and target variable distribution assessment.
* **Systematic Model Comparison:** Trains and evaluates three distinct classification models (**Random Forest**, **XGBoost**, and **SVM**) with hyperparameter tuning and stratified cross-validation.
* **Production-Ready Evaluation:** Implements rigorous evaluation using metrics suitable for imbalanced datasets, including macro-averaged F1-score, class-specific recall analysis, and ROC-AUC curves.
* **Model Interpretability:** Utilizes SHAP (SHapley Additive exPlanations) for feature importance analysis and business insight generation.
* **Interactive Prediction System:** Includes a user-friendly prediction interface built with `ipywidgets` for real-time solar flare risk assessment.

## 3. Tech Stack

* **Language:** Python 3.12+
* **Core Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Plotly
* **Machine Learning:** Random Forest, XGBoost, Support Vector Machine (SVM)
* **Data Handling:** imbalanced-learn (SMOTEENN), StandardScaler
* **Model Interpretation:** SHAP (SHapley Additive exPlanations)
* **Development Environment:** Jupyter Notebook, Google Colab
* **Environment Management:** Conda

## 4. Methodology

The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology, progressing through the phases of Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.

### 4.1. Dataset Characteristics

* **Size**: 1,391 sunspot observations from the UCI Solar Flare dataset (accessed via Kaggle)
* **Features**: 10 morphological characteristics of solar active regions
* **Target Distribution**: Severely imbalanced classes
  - No Flare: ~85% of observations
  - C-class: ~14% of observations  
  - M-class: ~4% of observations
  - X-class: ~1% of observations (only 14 total cases)
* **Challenge**: Extreme rarity of critical X-class events requiring specialized handling

### 4.2. Technical Approach

#### Data Preprocessing & Feature Engineering
* **Ordinal Encoding**: Applied to ranked features (`largest spot size`, `spot distribution`) preserving natural ordering from smallest to largest/least to most compact
* **One-Hot Encoding**: Used for nominal categorical feature (`modified Zurich class`) to handle magnetic complexity classifications
* **Binary Standardization**: Converted features to intuitive 0/1 encoding where higher values indicate greater complexity/size
* **Feature Scaling**: StandardScaler normalization for SVM optimization

#### Class Imbalance Strategy  
* **SMOTEENN**: Combined SMOTE (Synthetic Minority Oversampling Technique) with Edited Nearest Neighbors to address severe class imbalance
  - Generates synthetic examples of rare X-class flares
  - Removes noisy majority class samples to improve boundary definition
  - Results in 1.8x data amplification with balanced representation
* **Class Weighting**: Implemented balanced class weights to penalize minority class misclassification more heavily during training

#### Model Development & Validation
* **Algorithms**: Systematic comparison of Random Forest, XGBoost, and SVM classifiers chosen for their effectiveness with imbalanced, structured data
* **Hyperparameter Optimization**: 
  - GridSearchCV and RandomizedSearchCV for parameter tuning
  - F1-macro scoring to ensure equal consideration of all classes regardless of frequency
* **Cross-Validation**: Stratified 5-fold CV preserving class distributions across folds to ensure reliable performance estimates with rare events
* **Evaluation Metrics**: Focus on macro-averaged F1, X-class recall (>60% target), and ROC-AUC for comprehensive performance assessment

#### Model Interpretability
* **SHAP Analysis**: Applied SHapley Additive exPlanations to identify most influential features for X-class flare prediction
* **Feature Importance**: Quantified contribution of each sunspot characteristic to model predictions
* **Business Insights**: Revealed that `largest spot size` and `spot distribution` are primary predictors of severe flare activity

#### Technical Challenges Addressed
* **Extreme Class Imbalance**: X-class flares represent only 1% of observations (14 total cases)
* **Small Dataset Constraints**: Limited to 1,391 observations for multi-class classification
* **Rare Event Detection**: Optimized for critical X-class flare recall (75% achieved vs 60% target)
* **Feature Engineering**: Preserved ordinal relationships while handling mixed data types

### 4.3. Business Validation
* **Success Criterion**: >60% X-class flare recall (achieved 75%)
* **Operational Impact**: Successfully identifies 3 out of 4 severe flare events
* **Risk Mitigation**: Balanced approach minimizing false negatives for critical events

## 5. Key Results & Performance

The final **Random Forest Classifier** was selected for its superior performance and interpretability. The model's performance on the held-out test set demonstrates its effectiveness in identifying critical flare events.

**Model Performance Summary:**
- **F1-macro Score**: 0.471 (best among all models tested)
- **Overall Accuracy**: 75.3%
- **ROC-AUC Score**: 0.806 (strong discrimination capability)
- **Performance Context**: Achieved despite dataset constraints (1,391 observations, 1% X-class events)

**Critical X-Class Flare Detection:**
- **X-class Recall**: 75% (meeting key success criterion of >60%)
- **Business Impact**: Successfully identifies 3 out of 4 severe flare events

| Model           | Accuracy | F1-macro | X-class Recall | ROC-AUC |
| :-------------- | :------: | :------: | :------------: | :-----: |
| **Random Forest** |  75.3%   |  0.471   |     75.0%      |  0.806  |
| XGBoost         |  74.1%   |  0.393   |     25.0%      |  0.740  |
| SVM             |  76.5%   |  0.436   |     25.0%      |  0.734  |

The Random Forest model excels at detecting the most dangerous X-class flares while maintaining strong overall performance, making it ideal for operational space weather forecasting.

## 6. Installation & Usage

### Quick Start (Recommended)
**Google Colab** - No installation required:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sinahuss/solar-flare-prediction/blob/main/notebooks/solar_flare_analysis.ipynb)

- Click the badge above to run the notebook in your browser
- Runtime: ~8 minutes on Colab CPU
- Interactive prediction system included

### Local Installation
**Prerequisites:**
- Conda (Anaconda or Miniconda)
- Git

**Setup:**
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sinahuss/solar-flare-prediction.git
    cd solar-flare-prediction
    ```
2.  **Create and activate a conda environment:**
    ```bash
    conda create --name solar-flare-env --file requirements.txt
    conda activate solar-flare-env
    ```
3.  **Verify installation:**
    ```bash
    python -c "import sklearn, xgboost, shap, imblearn; print('Environment ready!')"
    ```
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  Open `notebooks/solar_flare_analysis.ipynb` and run all cells.

## 7. Repository Structure

```
solar-flare-prediction/
├── data/
│   └── data.csv                     # UCI Solar Flare dataset (1,391 observations)
├── notebooks/
│   └── solar_flare_analysis.ipynb   # Main analysis notebook with complete pipeline
├── .gitignore                       # Git ignore patterns
├── requirements.txt                 # Conda environment specification
└── README.md                       # Project documentation
```

### Directory Contents

* **`data/`**: Contains the UCI Solar Flare dataset with 13 features (10 input + 3 target variables)
* **`notebooks/`**: Jupyter notebook implementing the complete CRISP-DM methodology
* **`requirements.txt`**: Conda environment file generated with `conda list --export` for exact reproducibility
* **`.gitignore`**: Excludes development artifacts and documentation drafts

### Data Files

* **`data.csv`** (35KB): Clean, preprocessed solar flare dataset from UCI Machine Learning Repository
  - **Format**: CSV with headers
  - **Encoding**: UTF-8
  - **Size**: 1,391 rows × 13 columns
  - **Missing Values**: None (cleaned dataset)

## 8. Data Documentation

### 8.1. Data Source & Provenance

* **Immediate Source**: [Kaggle - Solar Flares Dataset](https://www.kaggle.com/datasets/stealthtechnologies/solar-flares-dataset)
* **Original Source**: [UCI Machine Learning Repository - Solar Flare Dataset](https://archive.ics.uci.edu/dataset/89/solar+flare)
* **Dataset ID**: UCI ML Repository #89
* **Original Donation**: February 28, 1989
* **Data Provider**: Solar-Geophysical Data reports from solar monitoring stations
* **Collection Period**: Historical sunspot observations spanning multiple solar cycles
* **Data Type**: Observational data from solar active regions with morphological characteristics
* **Access Method**: Downloaded from Kaggle, hosted on GitHub repository for notebook access
* **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
* **DOI**: [10.24432/C5530G](https://doi.org/10.24432/C5530G)

### 8.2. Data Quality Assessment

#### Completeness
* **Missing Values**: 0% - Dataset is complete with no missing values
* **Coverage**: 1,391 sunspot region observations (1,389 from UCI + header row + potential preprocessing)
* **Target Completeness**: All flare classifications present (C, M, X classes)
* **Data Integrity**: Verified against UCI ML Repository specifications

#### Data Validation
* **Feature Consistency**: All categorical values match expected domain ranges
* **Target Variable Validation**: Non-negative integer counts for flare occurrences
* **Outlier Analysis**: No invalid values detected in feature inspection
* **Data Types**: Proper encoding verified for categorical and numerical features

#### Known Limitations
* **Class Imbalance**: Extreme skew toward no-flare cases (85% of observations)
* **Rare Events**: Only 14 X-class flare observations (1% of dataset)
* **Sample Size**: Limited to 1,391 observations, constraining model training
* **Temporal Coverage**: Historical dataset may not reflect current solar cycle characteristics
* **Feature Scope**: Limited to 10 morphological characteristics - additional solar parameters could enhance predictions

### 8.3. Feature Schema

| Feature | Type | Domain | Description |
|---------|------|---------|-------------|
| `modified Zurich class` | Categorical | A,B,C,D,E,F,H | Magnetic complexity classification |
| `largest spot size` | Ordinal | X,R,S,A,H,K | Size of largest spot (X=smallest, K=largest) |
| `spot distribution` | Ordinal | X,O,I,C | Compactness (X=dispersed, C=compact) |
| `activity` | Binary | 1,2 | Region activity (1=reduced, 2=unchanged) |
| `evolution` | Ordinal | 1,2,3 | 24h evolution (1=decay, 2=no growth, 3=growth) |
| `previous 24 hour flare activity` | Ordinal | 1,2,3 | Prior activity (1=nothing as big as M1, 2=one M1, 3=more than one M1) |
| `historically-complex` | Binary | 1,2 | Ever complex (1=Yes, 2=No) |
| `became complex on this pass` | Binary | 1,2 | Current complexity (1=Yes, 2=No) |
| `area` | Binary | 1,2 | Total area (1=small, 2=large) |
| `area of largest spot` | Binary | 1,2 | Largest spot area (1=≤5, 2=>5) |
| `common flares` | Count | 0+ | Number of C-class flares in next 24h |
| `moderate flares` | Count | 0+ | Number of M-class flares in next 24h |
| `severe flares` | Count | 0+ | Number of X-class flares in next 24h |

## 9. Citation

If you use this project or dataset, please cite:

### Dataset Citation
```
Solar Flare [Dataset]. (1989). UCI Machine Learning Repository. https://doi.org/10.24432/C5530G.
```

### Kaggle Source
```
Stealth Technologies. (2024). Solar Flares Dataset. Kaggle. https://www.kaggle.com/datasets/stealthtechnologies/solar-flares-dataset
```

### BibTeX
```bibtex
@misc{uci_solar_flare_1989,
  author       = {UCI Machine Learning Repository},
  title        = {Solar Flare Dataset},
  year         = {1989},
  howpublished = {\url{https://archive.ics.uci.edu/dataset/89/solar+flare}},
  doi          = {10.24432/C5530G}
}
```