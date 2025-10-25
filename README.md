# 🏦 Machine Learning Project – Credit Approval

## 📘 Project Overview

A ficticious bank called Mobil Bank aims to grant vehicle financing to clients.  
To minimize financial risk, it requires a **Machine Learning model** capable of predicting — based on client attributes such as income, profession, and marital status — whether a client will be a **good payer (`1`)** or a **bad payer (`0`)**.

The **Credit Risk Department** defined the following requirements:

* 🎯 **Minimum Accuracy:** 70% (expected ≥ 80%)
* 🔁 **Minimum Recall:** 70% (expected ≥ 75%)
* ⚖️ **Decision Threshold:** 0.5 (adjustable according to business needs)
* 🧠 **Model Explainability:** must be interpretable
* 🌐 **Model served via API**
* 🖥️ **User Interface (UI)**

---
## 📦 Project Structure

```
📂 credit-approval-ml
├── screenshots/                     # Feature importance, metrics, and Streamlit webapp screenshots
│
├── src/                             # Source code
│   ├── API.py                       # Flask API endpoints
│   ├── Best_Model.py                # Model training and evaluation pipeline
│   ├── Const.py                     # SQL queries and constants
│   ├── Models.py                    # Scripts for testing and comparing ML models
│   ├── webapp/                      # Streamlit frontend application
│   └── utils.py                     # Reusable helper functions
│
├── tests/                           # Automated testing suite
│   ├── unit/                        # Unit tests (isolated modules)
│   │   ├── test_utils.py            # Tests for helper functions
│   │   ├── test_data_preprocessing.py # Tests for data cleaning & feature engineering
│   │   ├── test_feature_selection.py # Tests for RFE and feature selection
│   │   ├── test_model_training.py   # Tests for model training and tuning
│   │   ├── test_model_evaluation.py # Tests for model metrics and scoring
│   │   └── test_API.py              # Unit tests for API routes
│   │
│   ├── integration/                 # Integration tests (end-to-end)
│   │   ├── test_ModelWithAPI.py     # Tests integration between model and API
│   │   ├── test_database_connection.py # Tests PostgreSQL connection and data retrieval
│   │   └── test_end_to_end_pipeline.py # Full flow: data → model → API
│   │
│   ├── conftest.py                  # Shared fixtures (Flask client, fake model, DB mocks)
│
├── config.yaml                      # Database configuration (ignored by .gitignore)
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # Project overview, setup, and documentation

```

---

## 📊 Exploratory Data Analysis (EDA)

During the exploratory analysis, the following issues were identified:

* Outliers  
* Input errors  
* Missing values  

All issues were addressed during **data preprocessing**, ensuring a clean and reliable training dataset.

---

## 🧹 Data Cleaning, Preprocessing, and Feature Engineering

The preprocessing pipeline included:

1. **Handling Missing Values**
   * Mode imputation for categorical variables  
   * Median imputation for numerical variables  

2. **Correction of Input Errors**
   * Removal or correction of invalid or out-of-domain entries  

3. **Feature Engineering**
   * Creation of derived variables to enhance predictive power  

4. **Train/Test Split**

5. **Data Scaling**
   * Standardization or normalization  

6. **Categorical Encoding**
   * Label Encoding for categorical features  

7. **Feature Selection**
   * Random Forest feature importance for variable selection  

> 🧩 All preprocessing steps were serialized using **Joblib**, ensuring reproducibility and consistent transformations during inference.

---

## 🤖 Models and Evaluation

Given the small dataset, three candidate models were tested:

* **K-Nearest Neighbors (KNN)**
* **Support Vector Machine (SVM)**
* **Random Forest**

Validation was performed using **Leave-One-Out Cross Validation (LOO-CV)** with **Randomized Search CV** for hyperparameter optimization.  
All experiment tracking was done using **MLflow**.

---

## 🏁 Final Model: Random Forest

The final selected model was a **Random Forest Classifier**, chosen for its superior balance between accuracy, interpretability, and risk control.

### 🔢 Confusion Matrix

|                     | **Pred 0 (Bad)** | **Pred 1 (Good)** |
|---------------------|------------------|-------------------|
| **Actual 0 (Bad)**  | 57               | 4                 |
| **Actual 1 (Good)** | 17               | 72                |

---

### 📈 Classification Metrics

| Class | Description | Precision | Recall | F1-Score | Support |
|--------|--------------|------------|----------|-----------|----------|
| **0** | Bad Payer | 0.7703 | 0.9344 | 0.8444 | 61 |
| **1** | Good Payer | 0.9474 | 0.8090 | 0.8727 | 89 |
| **Accuracy** | — | — | — | **0.8600** | 150 |
| **Macro Avg** | — | 0.8588 | 0.8717 | 0.8586 | 150 |
| **Weighted Avg** | — | 0.8753 | 0.8600 | 0.8612 | 150 |

---

### 🧮 Overall Performance – Random Forest

| Metric        | Score |
|----------------|--------|
| **Accuracy**   | 0.86 |
| **Precision**  | 0.9474 |
| **Recall**     | 0.8090 |
| **F1-Score**   | 0.8727 |
| **ROC-AUC**    | 0.9239 |

---

## 💡 Business Interpretation

In the context of **credit approval**, the **priority is identifying and avoiding bad payers (class `0`)** to minimize default risk.

| Metric (Bad Payers - Class 0) | Value | Interpretation |
|-------------------------------|--------|----------------|
| **Precision (0.77)** | When the model predicts a bad payer, it is correct 77% of the time. |
| **Recall (0.93)** | The model successfully detects **93% of all real bad payers**, which is critical for risk management. |
| **F1-Score (0.84)** | Strong balance between precision and recall for the negative class. |

📊 **Key takeaway:**
> The model correctly identifies 93% of bad payers *(recall for bad payers (93%)* — reducing the probability of granting credit to high-risk clients — while maintaining solid accuracy (86%) across all predictions.

---

## 🧠 Model Explainability

Feature importance was used to understand how variables influence the probability of being a **good payer (class 1)** or **bad payer (class 0)**.

Top three most important features:

1. `requested_to_total_ratio`  
2. `requested_value`  
3. `income`  

These features provide transparency for business users and help justify model decisions during credit review.

---

## 🚀 API and User Interface

Deployment architecture:

* **Flask API** – Receives JSON input and returns predictions in real time  
* **Streamlit App** – Provides an intuitive interface for analysts and managers to test credit scenarios  

Both components are deployed directly on an AWS EC2 instance, which serves as the virtual machine hosting the API and the Streamlit UI.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c18f8c14-4536-4616-a6c2-f8b3ef15d8ee" />


---

## 🗄️ Database Integration (PostgreSQL)

The system connects directly to a **PostgreSQL** database as its **primary data source**.  
All client records used for training, testing, and inference are **fetched dynamically from the database**, ensuring that the model always operates on **the most up-to-date information**.

Typical configuration:

```text
Database: credit_db
User: admin
Password: <your_password>
Host: localhost
Port: 5432
```
---
## 🧱 Tech Stack

* **Python 3.x**
* **Pandas, NumPy, Scikit-learn, Joblib**
* **Flask** (API)
* **Streamlit** (UI)
* **PostgreSQL** (data source)
* **MLflow** (tracking metrics and experiments)
* **AWS EC2** (deployment)
---

## 🏁 Conclusions, Learnings, and Next Steps

### ✅ Conclusions
* The **Random Forest model** effectively balances accuracy and risk management, achieving **86% accuracy** and **93% recall for bad payers**, aligning with the business goal of minimizing credit defaults.  
* High importance features (`requested_to_total_ratio`, `requested_value`, `income`) highlight key drivers of client creditworthiness, providing **transparency and explainability**.  
* Deployment via **Flask API** and **Streamlit UI** demonstrates that the model can be **operationalized for real-time decision-making**, directly connected to the PostgreSQL database.  

### 🧠 Key Learnings
* **Leave-One-Out Cross Validation** combined with **Randomized Search CV** is effective for **small datasets**, ensuring robust evaluation and hyperparameter tuning.  
* Feature engineering and preprocessing are **critical for improving model performance**, especially in noisy or incomplete datasets.  
* Prioritizing **recall for bad payers** is more aligned with business objectives than simply maximizing overall accuracy.  
* Integrating model explainability into the workflow helps **build trust with business stakeholders** and supports decision justification.  

### 🚀 Next Steps
1. **Model Improvement**
   * 🛠️ Test additional ensemble methods (e.g., Gradient Boosting, XGBoost) with a larger dataset to potentially improve performance.  
   * 💡 Experiment with alternative feature engineering approaches.  

2. **Monitoring & Maintenance**
   * 📊 Implement **automatic model performance monitoring** to detect drift over time.  
   * ⚠️ Set up alerts for degradation in recall for bad payers.  

3. **Scalability**
   * 🐳 Containerize both API and UI using **Docker** for consistent deployment across environments.  
   * ☁️ Explore **cloud-managed databases** and serverless deployment to improve scalability and reliability.  

4. **Business Integration**
   * 🧪 Conduct **A/B testing** in a controlled environment to validate the model's real-world impact.  
   * 📈 Expand the system to include **dynamic decision thresholds** based on portfolio risk strategies.  

> This project demonstrates a complete ML lifecycle, from data ingestion to production-ready deployment, while emphasizing **risk-aware credit decision-making**.

