🔐 Securing Smart Grid Cyber-Physical Systems Against False Data Injection Attacks (FDIAs)

This project aims to build a robust, ML-based detection system for identifying False Data Injection Attacks (FDIAs) in smart grid infrastructures. Using a custom stacked ensemble model (XETStack), it leverages the strengths of Extra Trees and XGBoost classifiers for enhanced cyber-attack detection.

📌 Problem Statement
False Data Injection Attacks are stealthy and dangerous cyber threats targeting smart grids. They manipulate sensor data to mislead control centers, potentially causing widespread disruption. Traditional models lack accuracy and robustness in detecting such threats.

🎯 Objectives
✅ Detect FDIAs using supervised ML techniques.

✅ Address class imbalance using SMOTE.

✅ Optimize models via GridSearchCV and Optuna.

✅ Design a custom stacked ensemble model (XETStack).

🔍 Dataset
Source: Mississippi State University & Oak Ridge National Laboratory

Size: 4966 rows, 129 features

Features: Voltage, Current, Impedance, Phase Angles, Relay Status, etc.

Target Classes: Natural (1), Attack (0)

🧠 Machine Learning Models Used
Extra Trees Classifier

Random Forest

XGBoost

Logistic Regression

Decision Tree

AdaBoost

Gradient Boosting

SVM

KNN

Bagging Classifier

XETStack (Custom Ensemble: Extra Trees + XGBoost with XGBoost Meta-Learner)

⚙️ Technologies & Tools
Python 3.8+

Scikit-learn

XGBoost

Optuna

SMOTE (imbalanced-learn)

Pandas, NumPy, Seaborn, Matplotlib

📈 Model Performance
Model	Accuracy
XETStack	98.59%
Extra Trees	97.40%
Random Forest	96.75%
XGBoost	95.35%
Others	70–94%

SMOTE balanced the dataset effectively, improving detection of the minority class (Natural). Optuna helped achieve optimal hyperparameters for ensemble performance.

🧪 Key Features
Data cleaning (handling NaN and inf values)

Class balancing via SMOTE

Feature importance visualization

Confusion matrices and ROC-AUC curves

Hyperparameter tuning with Optuna & GridSearchCV
