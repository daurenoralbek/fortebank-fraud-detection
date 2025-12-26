# ForteBank Fraud Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/daurenoralbek/fortebank-fraud-detection)
[![Jupyter](https://img.shields.io/badge/Jupyter-100%-orange.svg)](https://jupyter.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

> An advanced machine learning-based anti-fraud detection system designed to identify and classify fraudulent banking transactions in real-time, with comprehensive model evaluation and explainability analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Interactive Web App](#interactive-web-app)
- [Model Performance](#model-performance)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project implements a comprehensive fraud detection system for banking transactions using machine learning techniques. The system leverages multiple algorithms and evaluation metrics specifically designed for imbalanced classification problems, ensuring effective fraud identification while minimizing false positives that could negatively impact legitimate customers.

The research was conducted as part of the **Fortebank Hackathon** competition, addressing real-world challenges in financial fraud detection within the Kazakhstani banking sector.

### Problem Statement

Banking fraud poses a significant threat to financial institutions and customers alike. Traditional rule-based systems often miss sophisticated fraud patterns while generating excessive false positives. This project develops a machine learning solution that:

- Detects fraudulent transactions with high accuracy (85% recall at optimized threshold)
- Maintains reasonable false positive rates to minimize customer friction
- Provides interpretable results for risk analysis and decision-making
- Handles severe class imbalance (1.26% fraud rate) typical in real banking data

## ✨ Key Features

### Model Capabilities

- **Multi-Algorithm Ensemble**: Combines XGBoost, LightGBM, and Random Forest with meta-learner (Logistic Regression)
- **Imbalanced Data Handling**: Implements SMOTE (9.09% resampling strategy), class weighting, and threshold optimization
- **Comprehensive Evaluation**: Uses appropriate metrics (Precision, Recall, F1-score, AUC-ROC, AUC-PR) for fraud detection
- **Feature Engineering**: 44 engineered features including transaction velocity, behavioral patterns, and temporal features
- **Model Interpretability**: Implements SHAP values and feature importance analysis for model transparency
- **Interactive Web Interface**: Streamlit application for real-time fraud detection predictions

### Analysis Capabilities

- Exploratory Data Analysis (EDA) with detailed statistical insights
- Feature correlation and distribution analysis
- Confusion matrix analysis and threshold optimization
- ROC and Precision-Recall curve visualization
- SHAP-based feature importance visualization
- Business impact analysis (fraud detection rate vs. false positive rate)
- Customer behavioral pattern analysis
- Mobile internet transaction analysis

## 📊 Dataset

### Overview

The dataset contains **13,126 transactions** from banking operations merging transaction and behavioral data with both transactional and identity information. The project leverages **two main datasets** as specified in the competition criteria.

### Data Splits

| Set | Samples | Fraud Rate | Purpose |
|-----|---------|-----------|---------|
| **Training** | 7,875 | 1.27% | Model training |
| **Validation** | 2,625 | 1.22% | Hyperparameter tuning |
| **Test** | 2,626 | 1.26% | Performance evaluation |
| **After SMOTE** | 8,552 | 9.09% | Balanced training set |

### Feature Categories

| Category | Count | Description |
|----------|-------|-------------|
| **Transaction Features** | ~15 | Amount, timestamp, product code, card details, billing/email address |
| **Engineered Features** | ~300+ | Numerical features (V1-V339) generated from raw data |
| **Count Features** | 14 | C1-C14: Count-based features (cards per address, etc.) |
| **Time Delta Features** | 15 | D1-D15: Temporal features (days between events) |
| **Matching Features** | 9 | M1-M9: Address/cardholder match flags |
| **Behavioral Features** | ~10 | Device type, IP address, browser information |
| **Final Features Used** | **44** | Optimized subset for model training |

### Data Characteristics

- **Target Variable**: Binary classification (0=Legitimate, 1=Fraudulent)
- **Class Imbalance**: 1.26% fraudulent transactions (severe imbalance)
- **Missing Values**: Handled through imputation and feature engineering
- **Temporal Component**: Transactions include timestamp information for behavioral analysis

## 📁 Project Structure

```
fortebank-fraud-detection/
├── .devcontainer/                           # Dev container configuration
├── .gitignore                               # Git ignore rules
│
├── Notebook.ipynb                           # Main analysis and model training notebook
├── app.py                                   # Streamlit web application for predictions
├── models.pkl                               # Trained models (serialized)
├── requirements.txt                         # Python dependencies
│
├── shap_summary_lgb.png                     # SHAP feature importance (LightGBM)
├── shap_summary_xgb.png                     # SHAP feature importance (XGBoost)
│
├── поведенческие_паттерны_клиентов.csv     # Behavioral patterns analysis (Russian)
├── транзакции_в_Мобильном_интернете.csv    # Mobile internet transactions (Russian)
│
├── README.md                                # This file
└── LICENSE                                  # MIT License
```

### File Descriptions

| File | Purpose |
|------|---------|
| **Notebook.ipynb** | Comprehensive Jupyter notebook containing data exploration, feature engineering, model training, evaluation, and SHAP analysis |
| **app.py** | Streamlit web application for interactive fraud detection predictions and model insights |
| **models.pkl** | Pre-trained ensemble model (XGBoost + LightGBM + Random Forest + Logistic Regression meta-learner) |
| **requirements.txt** | Python package dependencies for reproducibility and environment setup |
| **shap_summary_lgb.png** | Feature importance visualization from LightGBM model using SHAP values |
| **shap_summary_xgb.png** | Feature importance visualization from XGBoost model using SHAP values |
| **поведенческие_паттерны_...** | Analysis results of customer behavioral patterns extracted from transactions |
| **транзакции_в_Мобильном_...** | Analysis results of mobile internet channel transactions |
| **.devcontainer/** | Docker configuration for consistent development environment across machines |

## 🔬 Methodology

### 1. Data Processing Pipeline

```
Raw Data (13,126 tx) → Merging (tx + behavior) → Handling Missing Values → 
Feature Engineering (44 features) → Scaling/Normalization → Train-Val-Test Split (60-20-20)
```

**Key Steps**:
- Merge transaction dataset (13,113 rows × 7 cols) with behavior dataset (8,587 rows × 19 cols) on transaction ID
- Handle missing values through median imputation for behavioral features
- Replace sentinel values (-1) with NaN and impute
- Create derived features from raw transaction and identity information
- Apply StandardScaler for numerical features
- Execute time-based train-test split to preserve temporal structure

### 2. Feature Engineering

**Final 44 Features Include**:
- **Temporal Features**: Day of week, hour, time since last transaction, month
- **Aggregation Features**: Customer transaction count (7d, 30d), recipient frequency
- **Behavioral Features**: Device instability, login surge ratio, account dormancy, session anomalies, burstiness
- **Statistical Features**: Average/std deviation of intervals, z-score metrics
- **Interaction Features**: Device instability × amount, inactive × large transaction, anomaly timing
- **One-Hot Encoding**: Day of week, month, amount bins

### 3. Model Development

The project implements and optimizes three base models with ensemble stacking:

| Model | Configuration | Validation PR-AUC | Purpose |
|-------|---|---|---|
| **XGBoost** | 274 estimators, depth 8, lr 0.1 | 0.6341 | Gradient boosting baseline |
| **LightGBM** | 50 leaves, depth 7, lr 0.05 | 0.6526 | Fast, memory-efficient |
| **Random Forest** | 200 estimators, depth 20 | 0.5882 | Robust ensemble |
| **Meta-Learner** | Logistic Regression (C=1.0) | — | Combines predictions |

### 4. Class Imbalance Handling

- **SMOTE**: Applied to training set only, resampling to 9.09% fraud (ratio 0.1)
- **Class Weighting**: scale_pos_weight = 39 for XGBoost
- **Threshold Optimization**: Search across thresholds 0.01-0.99 to minimize business cost
- **Stratified Split**: Maintained 1.26% fraud rate across train/val/test

### 5. Model Evaluation

For fraud detection with imbalanced data, we use:

**Threshold-Based Metrics**:
- **Precision**: TP / (TP + FP) — Reliability of fraud predictions
- **Recall**: TP / (TP + FN) — Fraud detection rate
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall) — Balanced metric
- **Specificity**: TN / (TN + FP) — Legitimate transaction accuracy

**Threshold-Free Metrics**:
- **AUC-ROC**: Overall model discrimination ability (0.9933)
- **AUC-PR**: Precision-Recall curve area (better for imbalanced data)

**Confusion Matrix**:
- **TP = 28**: Fraud correctly detected
- **FP = 28**: Legitimate transactions wrongly flagged
- **FN = 5**: Fraud cases missed (financial loss)
- **TN = 2,565**: Legitimate transactions correctly allowed

## 📈 Results

### Final Model Performance (Test Set, Threshold = 0.03)

| Metric | Value |
|--------|-------|
| **AUC-ROC** | 0.9933 |
| **Precision** | 0.5000 |
| **Recall** | **0.8485** |
| **F1-Score** | 0.6292 |
| **Specificity** | 0.9892 |

### Confusion Matrix (Threshold = 0.03)

```
Predicted:        Fraud    Legitimate
Actual Fraud        28          5       (33 total fraud cases)
Actual Legitimate   28      2,565     (2,593 total legitimate)
```

### Key Insights

1. **High Fraud Detection**: **84.85% recall** detects 28 out of 33 fraud cases in test set
2. **Excellent Specificity**: **98.92% specificity** correctly identifies 2,565 out of 2,593 legitimate transactions
3. **Balanced Trade-off**: At threshold 0.03, FP = FN = 28 (balance between costs)
4. **Optimal Performance**: Threshold 0.03 minimizes business cost function (cost = 0.20×FP + 1.00×FN = 10.60)

### Model Comparison (Validation PR-AUC)

| Model | PR-AUC | Hyperparameters |
|-------|--------|---|
| XGBoost | 0.6341 | 274 est., depth 8, lr 0.1, subsample 0.8 |
| **LightGBM** | **0.6526** | 50 leaves, depth 7, lr 0.05 (BEST) |
| Random Forest | 0.5882 | 200 est., depth 20, sqrt features |
| Ensemble | — | Meta-learner combines all three |

**LightGBM shows superior performance** on PR-AUC metric, particularly important for imbalanced data.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git

### Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/daurenoralbek/fortebank-fraud-detection.git
cd fortebank-fraud-detection
```

2. **Create virtual environment**:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n fraud-detection python=3.8
conda activate fraud-detection
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Required Dependencies

Key packages:
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: ML algorithms & metrics
- `xgboost`, `lightgbm`: Gradient boosting models
- `imbalanced-learn`: SMOTE for handling class imbalance
- `matplotlib`, `seaborn`: Visualization
- `shap`: Model interpretability
- `streamlit`: Interactive web app
- `jupyter`: Notebook environment

## 💻 Usage

### Jupyter Notebook Analysis

1. **Start Jupyter**:
```bash
jupyter notebook
```

2. **Open `Notebook.ipynb`**:
   - Executes all 5 phases: Data Loading, Feature Engineering, Data Splitting, Model Training, Evaluation
   - Run cells sequentially for complete analysis
   - Adjust hyperparameters or thresholds as needed
   - View SHAP visualizations inline

### Load Pre-trained Models

```python
import pickle
import pandas as pd

# Load trained ensemble model
with open('models.pkl', 'rb') as f:
    models = pickle.load(f)

# Make predictions on new data
X_new = pd.read_csv('your_test_data.csv')
predictions = models['ensemble'].predict(X_new)  # Binary: 0 or 1
probabilities = models['ensemble'].predict_proba(X_new)[:, 1]  # Fraud probability

# Custom threshold (e.g., 0.30 instead of 0.50)
fraud_predictions = (probabilities > 0.30).astype(int)
```

## 🌐 Interactive Web App

### Running the Streamlit Application

```bash
streamlit run app.py
```

**Features**:
- Upload transaction data for batch predictions
- Single transaction fraud risk assessment
- Model performance metrics dashboard
- Feature importance visualization (SHAP)
- ROC and Precision-Recall curves
- Custom threshold adjustment slider
- Prediction explanations

## 📊 Model Performance

### Understanding Metrics

**Precision vs. Recall Trade-off**:
- **Precision = 0.50**: Half of flagged transactions are actually fraudulent; half are false alarms
- **Recall = 0.85**: Detects 85% of actual fraud; misses 15%
- **Business Decision**: Lower threshold (0.03) increases recall at cost of more false positives

### Threshold Selection

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| 0.30 | 0.66 | 0.70 | Conservative (lower FP) |
| 0.03 | 0.50 | 0.85 | Aggressive (catch more fraud) |
| 0.50 | — | — | Default scikit-learn |

Threshold 0.03 optimizes cost function: Cost = 0.20×FP + 1.00×FN

### Confusion Matrix Interpretation

**At threshold 0.03 (Test set, n=2,626)**:
- **True Positives (28)**: Fraud correctly caught ✓
- **False Positives (28)**: Legitimate customers wrongly flagged ⚠
- **False Negatives (5)**: Fraud missed ✗
- **True Negatives (2,565)**: Legitimate transactions allowed ✓

**Business Impact**:
- **Detection Rate**: 28/(28+5) = **84.85%** of fraud detected
- **False Positive Rate**: 28/2,593 = **1.08%** of legitimate transactions inconvenienced
- **Precision of Alerts**: 28/(28+28) = **50%** (1 in 2 flags is correct)

## 📊 Data Insights

### Customer Behavioral Patterns
The file `поведенческие_паттерны_клиентов.csv` contains:
- Transaction frequency by customer segment
- Average transaction amounts by product
- Peak transaction hours and days
- Customer lifetime value indicators

### Mobile Internet Transactions
The file `транзакции_в_Мобильном_интернете.csv` analyzes:
- Mobile channel fraud rates
- Device type risk profiles
- Location consistency metrics
- Mobile-specific fraud indicators

## 🔮 Future Enhancements

### Short-term
- [ ] Deploy model as REST API service
- [ ] Implement real-time monitoring dashboard with drift detection
- [ ] Add active learning feedback loop from fraud investigators
- [ ] Create model explainability reports (SHAP + LIME)

### Medium-term
- [ ] Develop hierarchical ensemble models for transaction types
- [ ] Implement concept drift detection for automated retraining
- [ ] Add feature store for feature management and versioning
- [ ] Create customer behavior profiling system

### Long-term
- [ ] Integrate with banking transaction processing system (real-time)
- [ ] Implement explainable AI for regulatory compliance (GDPR, AMLKYC)
- [ ] Develop anomaly detection for new fraud patterns
- [ ] Create automated model retraining pipeline with CI/CD

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. **Fork the repository**: Click "Fork" on GitHub
2. **Create feature branch**: `git checkout -b feature/your-feature`
3. **Make changes**: Implement your improvements
4. **Test thoroughly**: Run notebook cells and validate results
5. **Commit changes**: `git commit -m 'Add your feature'`
6. **Push to branch**: `git push origin feature/your-feature`
7. **Create Pull Request**: Describe your changes and submit PR

### Guidelines
- Follow PEP 8 Python style guide
- Add docstrings to new functions
- Test with sample data before submitting
- Reference issues in commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Permission is hereby granted to use this project for educational and research purposes.

## 👨‍💻 Contact

**Author**: Dauren Oralbek

- **GitHub**: [@daurenoralbek](https://github.com/daurenoralbek)
- **Institution**: Suleyman Demirel University (SDU)
- **Program**: Statistics & Data Science, 3rd Year

### Questions & Support

For questions, issues, or suggestions:
1. Check existing [GitHub Issues](https://github.com/daurenoralbek/fortebank-fraud-detection/issues)
2. Create new issue with detailed description
3. Include dataset characteristics (without sharing sensitive data)
4. For feature requests, explain the business value

## 📚 References & Resources

### Academic Papers & Competitions
- IEEE-CIS Fraud Detection Competition
- "Reproducible Machine Learning for Credit Card Fraud Detection" - Fraud Detection Handbook
- "Financial Fraud Detection Using Explainable AI" - Recent research papers

### External Resources
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Imbalanced Learn Documentation](https://imbalanced-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Related Projects
- [IEEE-CIS Fraud Detection Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)
- [Credit Card Fraud Datasets](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

**Last Updated**: December 2025  
**Project Status**: Active Development  
**Python Version**: 3.8+  
**Language**: Jupyter Notebook (97.8%) + Python (2.2%)

⭐ If you find this project useful, please consider giving it a star on GitHub!
