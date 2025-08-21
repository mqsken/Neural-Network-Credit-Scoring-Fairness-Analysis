# Neural-Network-Credit-Scoring-Fairness-Analysis

An analysis of algorithmic bias in AI-driven credit scoring models using neural networks and XAI techniques

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)

## 1. Project Overview

This project systematically evaluates the issue of algorithmic bias in various neural network models for credit scoring tasks and explores the effectiveness of eXplainable AI (XAI) techniques in identifying proxy discrimination.

With the widespread application of artificial intelligence in the financial sector, the fairness of its decision-making processes has become a significant social and regulatory concern. This study finds that although machine learning models demonstrate excellent predictive accuracy (generally **exceeding 97%**), they all exhibit significant and **systemic adverse bias** against minority groups.

The research confirms that simply removing sensitive features such as race and gender is insufficient to eliminate bias. Instead, XAI tools (like SHAP and QII) can serve as effective fairness auditing instruments to reveal the underlying mechanisms of bias. These findings have important theoretical and practical implications for building more responsible and equitable AI financial systems.

## 2. Key Findings & Visualization

The following image summarizes the performance comparison, fairness gaps, and feature importance analysis based on SHAP and QII for the five models developed in this project.

![Comprehensive Analysis Visualization](https://github.com/user-attachments/assets/a7a56d52-99bf-4576-8df5-265255d5e454)

## 3. Dataset

The data used in this study is sourced from the public **2023 Home Mortgage Disclosure Act (HMDA)** dataset. A balanced subset of 100,000 loan application records (50% approved, 50% denied) was extracted and constructed for model training and evaluation.

## 4. Methodology

### 4.1 Model Architectures

This project compares five different machine learning models to assess their performance and fairness across various architectures:

- **Multi-Layer Perceptron (MLP)**
- **Deep Neural Network (DNN)**
- **Mixture of Experts (MoE)**
- **Enhanced Radial Basis Function Network (Enhanced RBF)**
- **Random Forest** (as a baseline model)

### 4.2 Analysis Techniques

- **Fairness Metrics:** Bias is quantified using metrics such as Demographic Parity, Equal Opportunity, and the "80% Rule" (Disparate Impact Ratio).
- **Explainability Analysis:**
  - **SHAP (SHapley Additive exPlanations):** Used to quantify the contribution of each feature to the model's predictions.
  - **QII (Quantitative Input Influence):** Used to identify high-risk proxy variables that may lead to indirect discrimination.

## 5. Dependencies

Please ensure the following libraries are installed in your Python environment. You can install them using pip:

```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn jupyter shap
```

## 6. How to Run

### Step 1: Data Preprocessing

1. Place the original `2023_public_lar_csv.csv` dataset in the project's root directory.
2. Run the `Data_preprocessing.ipynb` notebook. This will perform data cleaning, feature engineering, and sampling, generating the `hmda_large_final_*.csv` and `hmda_labels_final_*.csv` files required for model training.

### Step 2: Model Training and Analysis

1. Ensure the data files generated in the previous step are in the same directory as `Models.ipynb`.
2. Run the `Models.ipynb` notebook. This script will sequentially train all models, conduct performance evaluation, fairness analysis, XAI analysis, and generate all result files (`.csv`, `.json`) and the final visualization chart (`.png`).

## 7. Results Summary

### Performance and Bias Metrics

| Model | AUC | Accuracy | Racial Bias Gap |
|-------|-----|----------|-----------------|
| Random Forest | 0.9979 | 98.92% | 56.84% |
| MLP | 0.9951 | 97.67% | 55.94% |
| DNN | 0.9949 | 97.58% | 55.65% |
| MoE | 0.9947 | 97.38% | 57.03% |
| Enhanced RBF | 0.9938 | 97.25% | 55.75% |

**Note:** The "Racial Bias Gap" refers to the absolute difference in predicted approval rates between White and African American applicants.

## 8. License

**MIT License**

This project is licensed under the MIT License.
