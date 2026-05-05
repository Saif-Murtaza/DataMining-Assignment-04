# Heart Disease Classification: A Machine Learning Approach

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

---

## 📌 Project Overview
This project leverages the **UCI Cleveland Heart Disease dataset** to develop a predictive system for cardiovascular risk assessment. By integrating traditional machine learning (Random Forest, XGBoost) with Deep Learning (Neural Networks), the project identifies high-risk patients based on clinical features like cholesterol levels, maximum heart rate, and ST depression.

### Key Objectives:
*   **Data Integrity:** Clean and preprocess clinical data, handling missing values and categorical encoding.
*   **Predictive Modeling:** Compare ensemble methods against neural networks for classification accuracy.
*   **Interpretability:** Use SHAP values to explain which clinical features most heavily influence a "high-risk" diagnosis.
*   **Deployment:** Provide a user-friendly Streamlit interface for real-time risk prediction.

---

## 📊 Dataset Description
The dataset contains **303 initial observations** (reduced to 297 after cleaning) with 14 clinical attributes:

| Feature | Description |
| :--- | :--- |
| `age` | Age in years |
| `cp` | Chest pain type (4 values) |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol in mg/dl |
| `thalach` | Maximum heart rate achieved |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `ca` | Number of major vessels (0-3) colored by fluoroscopy |
| `target` | Diagnosis of heart disease (0 = Absence, 1 = Presence) |

---

## 🛠️ Tech Stack & Methodology

### 1. Preprocessing Pipeline
*   **Cleaning:** Imputation and removal of rows with missing `ca` and `thal` values.
*   **Normalization:** Applied `StandardScaler` to continuous features to ensure model convergence.
*   **Oversampling:** Implemented **SMOTE** to address class imbalance, ensuring the model is sensitive to positive heart disease cases.

### 2. Model Architecture
*   **Random Forest & XGBoost:** Hyperparameter-tuned ensemble models for robust non-linear classification.
*   **Deep Learning:** A 3-layer Dense Neural Network built with **Keras/TensorFlow**, utilizing **Dropout layers** to mitigate overfitting.

---

## 📈 Performance Summary

| Model | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **88.3%** | **0.87** | **0.92** |
| XGBoost | 86.7% | 0.85 | 0.90 |
| Neural Network | 85.0% | 0.84 | 0.89 |

> **Key Finding:** The Random Forest model emerged as the most reliable predictor, particularly effective at identifying the influence of `ca` (vessels colored) and `thalach` (max heart rate) on patient outcomes.

---

## 🚀 How to Use

### Prerequisites
```bash
pip install pandas numpy scikit-learn tensorflow xgboost matplotlib seaborn shap streamlit

Running the Notebook
Clone the repository: git clone https://github.com/yourusername/heart-disease-ml.git

Open the .ipynb file in Jupyter or Google Colab.

Run all cells to reproduce the analysis and model training.

Running the Streamlit App
To launch the interactive dashboard:

streamlit run app.py

📜 Conclusion
This project demonstrates that machine learning can serve as a powerful clinical decision-support tool. By achieving an 88% accuracy rate, the model provides a significant baseline for identifying cardiovascular risks early, potentially assisting healthcare providers in prioritizing patient care.
