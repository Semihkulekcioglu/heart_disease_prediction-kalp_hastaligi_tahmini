# 🫀 Heart Disease Prediction - ML Classification Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Türkçe](README_TR.md) | **English**

A comprehensive machine learning project comparing 6 classification algorithms for heart disease prediction.

## 📊 Project Overview

This project implements and compares **6 different machine learning algorithms** to predict heart disease:

- ✅ Logistic Regression
- ✅ k-Nearest Neighbors (kNN)
- ✅ Decision Tree
- ✅ Random Forest
- ✅ LightGBM
- ✅ XGBoost

## 🎯 Features

- **Exploratory Data Analysis (EDA)** with comprehensive visualizations
- **6 ML algorithms** with hyperparameter optimization
- **Performance comparison** across all metrics
- **Reusable code** in modular structure
- **Professional documentation** and clean code

<p align="center">
  <img src="https://github.com/user-attachments/assets/1a92fbcf-5dc2-4170-83f1-45089fa98ae1" width="200" />
  <img src="https://github.com/user-attachments/assets/11b2a060-de8d-4385-a0d5-67047547c93e" width="200" />
  <img src="https://github.com/user-attachments/assets/7667bf29-cbcf-4e19-b2d1-ec50b63132da" width="200" />
</p>


## 📁 Project Structure

```
├── data/                      # Dataset
│   └── heart_disease.csv     # Heart disease data (303 patients)
├── src/                       # Python modules
│   ├── preprocessing.py      # Data preprocessing functions
│   └── model_utils.py        # Model utility functions
├── notebooks/                 # Jupyter notebooks
│   ├── 01_veri_analizi.ipynb           # EDA
│   ├── 02_logistic_regression.ipynb    # Logistic Regression
│   └── 08_model_karsilastirma.ipynb   # Model Comparison ⭐
├── models/                    # Saved models
├── results/                   # Results and visualizations
└── requirements.txt          # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Semihkulekcioglu/heart_disease_prediction-kalp_hastaligi_tahmini.git
cd heart_disease_prediction-kalp_hastaligi_tahmini

# Install dependencies
pip install -r requirements.txt

# Start Jupyter Notebook
jupyter notebook
```

### Usage

**Recommended:** Run `notebooks/08_model_karsilastirma.ipynb` to train and compare all 6 models at once!

## 📈 Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.82     | 0.83      | 0.80   | 0.81     | 0.88    |
| k-NN                | 0.85     | 0.84      | 0.86   | 0.85     | 0.90    |
| Decision Tree       | 0.78     | 0.75      | 0.82   | 0.78     | 0.80    |
| Random Forest       | 0.88     | 0.89      | 0.87   | 0.88     | 0.93    |
| LightGBM            | 0.90     | 0.91      | 0.89   | 0.90     | 0.95    |
| XGBoost             | 0.89     | 0.90      | 0.88   | 0.89     | 0.94    |

🏆 **Best Model:** LightGBM with 90% accuracy and 0.95 ROC-AUC

## 📊 Dataset

**Heart Disease Dataset** contains 303 patient records with 14 attributes:

- Age, sex, chest pain type
- Blood pressure, cholesterol
- ECG results
- Maximum heart rate
- Exercise angina, ST depression
- Target: Disease presence (0=healthy, 1=disease)

## 🛠️ Technologies

- **Python 3.8+**
- **Scikit-learn** - ML algorithms
- **Pandas & NumPy** - Data processing
- **Matplotlib & Seaborn** - Visualization
- **XGBoost & LightGBM** - Gradient boosting
- **Jupyter Notebook** - Interactive development

## 📝 Key Learnings

- ✅ Exploratory Data Analysis (EDA)
- ✅ Data preprocessing and feature scaling
- ✅ 6 classification algorithms
- ✅ Hyperparameter tuning (GridSearchCV)
- ✅ Model evaluation metrics
- ✅ Model comparison techniques

## 🎓 Educational Value

Perfect for:
- Machine learning beginners
- Data science students
- Portfolio projects
- Kaggle competitions practice

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

