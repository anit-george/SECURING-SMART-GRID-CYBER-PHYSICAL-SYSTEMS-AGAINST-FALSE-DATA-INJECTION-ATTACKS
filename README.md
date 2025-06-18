# 🔐 Securing Smart Grid Cyber-Physical Systems Against False Data Injection Attacks (FDIAs)

This repository contains a robust machine learning pipeline to detect **False Data Injection Attacks (FDIAs)** in smart grid infrastructure using a custom stacked ensemble model, **XETStack**.

---

## 📌 Problem Statement

False Data Injection Attacks (FDIAs) pose significant threats to modern smart grids. These attacks subtly modify measurement data sent from remote units to mislead control centers, potentially causing blackouts, instability, or equipment damage. Traditional models fail to detect such attacks reliably.

---

## 🎯 Objectives

- ✅ Detect FDIAs using supervised ML techniques
- ✅ Address class imbalance using **SMOTE**
- ✅ Optimize model performance using **GridSearchCV** and **Optuna**
- ✅ Build a custom **stacked ensemble model (XETStack)** for improved detection accuracy

---

## 🔍 Dataset Overview

- **Source:** Mississippi State University & Oak Ridge National Lab  
- **Samples:** 4,966  
- **Features:** 129  
- **Target Variable (`marker`):**
  - `1` = Natural (normal operation)
  - `0` = Attack (FDIA)

### Feature Categories

- Voltage & Current magnitudes and angles  
- Impedance, Frequency  
- Circuit breaker & relay statuses  

---

## 🧠 ML Models Used

- 🌲 Extra Trees Classifier
- 🌳 Random Forest
- ⚡ XGBoost
- 🔁 AdaBoost, Gradient Boosting, Bagging Classifier
- 📉 Logistic Regression, SVM, KNN, Decision Tree
- 🔝 **XETStack (Custom Stacked Ensemble of Extra Trees + XGBoost)**

---

## ⚙️ Tech Stack

- **Python 3.8+**
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn==1.2.2`
- `xgboost==1.7.6`
- `imbalanced-learn==0.10.1`
- `optuna==3.6.1`

---

## 🧪 Methodology

1. **Data Cleaning**
   - Removed `NaN`, `inf`, and redundant columns
   - Encoded target class (`marker`)

2. **Class Balancing**
   - Applied **SMOTE** to increase minority class representation

3. **Feature Analysis**
   - Correlation heatmap (see [`results/correlation_heatmap.jpg`](./results/correlation_heatmap.jpg))
   - Tree-based feature importance

4. **Modeling & Evaluation**
   - Compared 10+ ML classifiers on balanced data
   - Evaluated using **Accuracy**, **Precision**, **Recall**, **F1-score**, and **Confusion Matrix**

5. **XETStack Design**
   - Level-0: Extra Trees + XGBoost  
   - Level-1 Meta-Learner: XGBoost  
   - Hyperparameters optimized using **Optuna**

---

## 📈 Model Performance

| Model         | Accuracy (%) |
|---------------|---------------|
| **XETStack**  | **98.59**     |
| Extra Trees   | 97.40         |
| Random Forest | 96.75         |
| XGBoost       | 95.35         |
| Others        | 70%–94%       |
| SVM           | 23.27         |

📌 SMOTE balanced the classes:  
- Original: 4163 (Attack), 803 (Natural)  
- After SMOTE: 4163 (Attack), **2891 (Natural)**

---

## 📁 Project Structure

```
FDIA-Detection-Project/
├── binaryAllNaturalPlusNormalVsAttacks/   # Raw & balanced datasets
├── notebook/                              # Jupyter notebooks (main: final_take.ipynb)
├── results/                               # Evaluation plots, heatmaps, model outputs
├── src/                                   # ML pipeline scripts (if applicable)
├── requirements.txt                       # Project dependencies
└── README.md                              # Project documentation
```

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/FDIA-detection.git
cd FDIA-detection
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Launch the notebook**
```bash
jupyter notebook notebook/final_take.ipynb
```

---

## 📊 Visual Outputs

- ✅ Correlation Heatmap – [`correlation_heatmap.jpg`](./results/correlation_heatmap.jpg)
- ✅ Confusion Matrices (all models)
- ✅ ROC Curves
- ✅ Feature Importance Plots
- ✅ Model Accuracy Comparison Charts

---

## 🔮 Future Enhancements

- Real-time FDIA detection using streaming data pipelines (Kafka, Spark)
- Deep learning-based detection (LSTM, CNN)
- Integration with SCADA control systems
- Explainability using SHAP/LIME

---

## 👨‍💻 Authors

- **Anit George** – [LinkedIn](https://www.linkedin.com/in/anit-george)
- **Ishita Prakash**
- **K. S. Shanthi Priya**

> Capstone Project – B.Tech CSE  
> Vellore Institute of Technology (VIT)

---

## 📄 License

MIT License – feel free to use, modify, and share with proper credit.
