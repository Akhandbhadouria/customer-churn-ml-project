# Bank Churn Analysis Project

This repository contains several Jupyter Notebooks for analyzing and modeling customer churn in banking data. The main dataset used is `Bank_Churn.csv`.

## Notebooks Overview

### 1. EDAA.ipynb
- **Purpose:** Exploratory Data Analysis (EDA) of the bank churn dataset.
- **Key Steps:**
  - Data loading and inspection
  - Missing value and duplicate check
  - Target variable analysis (churn rate)
  - Visualizations: count plots, correlation heatmap, boxplots for outlier detection
  - Distribution analysis of key features

### 2. GRADIENT_BOOST_REGRESSOR.ipynb
- **Purpose:** Predicting customer balance using Gradient Boosting Regression.
- **Key Steps:**
  - Data cleaning and encoding
  - Feature engineering and selection
  - Feature scaling and train-test split
  - Model training and evaluation (R2, MSE, RMSE)
  - User prediction function for balance estimation

### 3. K_MEAN.ipynb
- **Purpose:** Clustering customers using K-Means.
- **Key Steps:**
  - Feature selection and scaling
  - Elbow method for optimal cluster count
  - Cluster assignment and visualization (PCA)
  - Cluster profiling
  - User prediction function for cluster assignment

### 4. KNN.ipynb
- **Purpose:** Classification of customer churn using K-Nearest Neighbors (KNN).
- **Key Steps:**
  - Data cleaning and encoding
  - Feature selection, scaling, and splitting
  - Model training and evaluation (accuracy, confusion matrix, classification report)
  - User prediction function for churn classification
  - Summary statistics for churned and non-churned customers

### 5. decision_treee.ipynb
- **Purpose:** Classification of customer churn using Decision Tree.
- **Key Steps:**
  - Data cleaning and encoding
  - Feature selection, scaling, and splitting
  - Model training and evaluation (accuracy, confusion matrix, classification report)
  - User prediction function for churn classification

## Dataset
- **Bank_Churn.csv:** Main dataset containing customer information and churn status.

## Usage
Open any notebook in VS Code or Jupyter and run the cells sequentially. Each notebook is self-contained and demonstrates a specific analysis or modeling technique.

## Requirements
- Python 3.x
- pandas, numpy, matplotlib, seaborn, scikit-learn

Install dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```


