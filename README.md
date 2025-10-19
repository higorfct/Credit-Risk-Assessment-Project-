# 🏦 Machine Learning Project – Credit Approval

## 📘 Project Overview

A ficticious bank called Mobil Bank aims to provide vehicle financing to clients.  
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
├── screenshots/ # feature importance, metrics and Streamlit webapp
├── src/
│ ├── API.py # Flask API
│ ├── Best_Model.py # Model training and evaluation
│ ├── Const.py # SQL queries
│ ├── Models.py # All models ran
│ ├── test_API.py # API endpoints test
│ ├── webapp/ # Streamlit application (frontend)
│ └── utils.py # Reusable functions
├── .gitignore
├── requirements.txt
└── README.md
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
> The model prioritizes *recall for bad payers (93%)* — reducing the probability of granting credit to high-risk clients — while maintaining solid accuracy (86%) across all predictions.

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

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3d01140d-85f1-44a6-a3d4-ef58f5423e0e" />


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

