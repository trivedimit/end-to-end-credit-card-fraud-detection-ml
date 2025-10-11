
# 💳 CREDIT CARD FRAUD DETECTION

_End-to-end machine learning project for credit card fraud detection using LightGBM and XGBoost._

---
## 🎯 Objective
Detect fraudulent credit card transactions using machine learning models with a focus on interpretability, precision-recall balance, and real-world deployability.
---

## 📌 Table of Contents
- <a href="#overview">Overview</a>
- <a href="#Problem-Statement">Problem Statement</a>
- <a href="#dataset">Dataset</a>
- <a href="#tools--technologies">Tools & Technologies</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#methodology">Methodology</a>
- <a href="#research-questions--key-findings">Results and Conclusion</a>
- <a href="#author--contact">Author & Contact</a>

---
<h2><a class="anchor" id="overview"></a>Overview</h2>

This project implements a complete **Credit Card Fraud Detection System**, covering every step from data preprocessing and feature engineering to advanced model training and interpretability using **SHAP**.  
It aims to **identify fraudulent transactions** while maintaining a healthy balance between **precision and recall**, crucial in highly imbalanced financial datasets.

---
<h2><a class="anchor" id="Problem Statement"></a>Problem Statement</h2>

Credit card fraud detection is a major challenge due to:
- **Extreme class imbalance** (fraud cases are <1%).
- **Evolving fraud patterns** over time.
- **High cost of false positives/negatives**.

The goal is to build a model that can **accurately detect frauds** while minimizing false alerts, achieving an optimal trade-off between **precision** and **recall**.

---
<h2><a class="anchor" id="dataset"></a>Dataset</h2>

- Couple of CSV files located in `/data/` folder (train_identity,train_transaction)
- **Records:** 590,000+ transactions  
- **Features:**  
  - `Time`, `Amount` — Transaction-level details  
  - `V1...V28` — PCA-transformed numerical features  
  - `Class` — Target variable (1 = Fraud, 0 = Legitimate)  
- **Imbalance ratio:** ~3.50% fraud cases  
- **Shape:** `(590540, 434)`

---

<h2><a class="anchor" id="tools--technologies"></a>Tools & Technologies</h2>

| Category            |  Tools / Libraries                                     |
|---------------------|--------------------------------------------------------|
| **Language**        | Python                                                 |
| **Data Handling**   | pandas, numpy                                          |
| **Visualization**   | matplotlib, seaborn                                    |
| **Modeling**        | scikit-learn, LightGBM, XGBoost                        |
| **Metrics**         | ROC AUC, Precision, Recall, F1-score, Confusion Matrix |
| **Explainability**  | SHAP, PDP, ICE                                         |
| **Development**     | Jupyter Notebook, Google Colab                         |
| **Version Control** | Git & GitHub                                           |

---
<h2><a class="anchor" id="project-structure"></a>Project Structure</h2>

```
CREDIT_CARD_FRAUD_DETECTION/
│
├── README.md
├── requirements.txt
│
├── dataset/
│   └── data files/              
│
├── notebook/
│   ├── credit_card_fraud_detection.ipynp
│
├── PDF/
│   └── Project_Report.pdf         
│
└── visuals/
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── shap_summary_plot.png
    └── other visuals
```

---

<h2><a class="anchor" id="methodology"></a>Methodology</h2>


### 1️⃣ Exploratory Data Analysis (EDA)
- Analyzed transaction amount distribution and fraud occurrence over time.
- Checked missing values, outliers, and correlations.
- Visualized relationships between anonymized PCA features and fraud class.

---

### 2️⃣ Data Cleaning
- Removed duplicates and invalid entries.
- Handled missing values (none significant in dataset).
- Log-transformed the `Amount` feature to reduce skewness.
- Scaled numerical features where appropriate.

---

### 3️⃣ Feature Engineering
- Created derived features like:
  - `Amount_log` = log(1 + Amount)
  - `hour` = transaction time (mod 24)
- Performed correlation filtering and variance thresholding.
- Applied encoding and transformation for consistent model input.
- Handled the class imbalance by using Random Oversampling.

---

### 4️⃣ Model Building
Trained multiple models and compared performance:

| Model                    | Accuracy   | ROC AUC    | Recall        | Notes                        |
|--------------------------|------------|------------|---------------|------------------------------|
| Logistic Regression      |  0.79      | 0.81       | 0.31          | Baseline                     | 
| XGBoost                  |  0.980     | 0.931      | 0.48          | Tuned with early stopping    |
| **LightGBM (base)**      | **0.9774** | **0.9239** | **0.4168**    | Very Fast                    |
| RandomForest Regressiom  |  0.9792    | 0.9295     | 0.4327        | Tuned with early stopping    |
| **LightGBM (Calibrated)**| **0.9804** | **0.945**  | **0.5041**    | ✅ Best overall performance  |

---

### 5️⃣ Model Evaluation
- **Metrics Used:**
  - Accuracy = 0.9804  
  - ROC AUC = 0.945  
  - Recall = 0.5041  
  - Precision = high due to strong imbalance handling  
- **Additional Metrics:**
  - Confusion Matrix  
  - True Positive Rate (TPR), False Positive Rate (FPR), True Negative Rate (TNR), False Negative Rate (FNR)

---

### 6️⃣ Explainability (SHAP)
- Used SHAP TreeExplainer for interpretability.
- Generated:
  - **SHAP Summary Plot** — global feature importance  
  - **SHAP Waterfall Plot** — local instance explanations  
- Top important features (example):
  - `card1`, `card2`,`card3`, `V12`, `Amount_log`

---
<h2><a class="anchor" id="research-questions--key-findings"></a>Results and Conclusion</h2>

- The LightGBM model delivered the highest AUC (0.945) and strong recall, making it ideal for fraud detection systems.
- The model generalizes well and remains interpretable using SHAP values.
- Balancing precision-recall ensured minimal customer disruption while catching majority of fraudulent cases.

Key Takeaways:

- Feature encoding and transformation improved stability.
- Stratified train-test split avoided bias from class imbalance.
- SHAP improved trust and interpretability of the model.

Future Improvements:

- Implement cost-sensitive learning to reflect real-world fraud costs.
- Deploy model via Flask / FastAPI with real-time scoring.

---
<h2><a class="anchor" id="author--contact"></a>Author & Contact</h2>

**MIT TRIVEDI**  
Data science Professional  
📧 Email: trivedimit04@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/mit-trivedi-8714a5344/)  
🔗 [Portfolio](https://github.com/trivedimit/)