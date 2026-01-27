# Bank Marketing Binary Classification – Machine Learning Project

## 1. Problem Statement
Direct marketing campaigns are widely used by banks to promote financial products such as term deposits. However, contacting every client is costly and inefficient. The objective of this project is to build and compare multiple machine learning models that can predict whether a client will subscribe to a term deposit based on demographic, financial, and campaign-related attributes.

This is formulated as a **binary classification problem**, where the target variable `y` indicates whether a client subscribed to a term deposit (`yes` or `no`).

---

## 2. Dataset Description
This project uses the **Bank Marketing dataset**, which is publicly available and widely used for research purposes.

### Dataset Source
- Created by **Paulo Cortez** (University of Minho) and **Sérgio Moro** (ISCTE-IUL)
- Described in: *Moro et al., 2011 – Using Data Mining for Bank Direct Marketing*

### Dataset Characteristics
| Property | Value |
|--------|-------|
| Dataset file | `bank-full.csv` |
| Number of instances | 45,211 |
| Number of input features | 16 |
| Target variable | 1 (binary) |
| Missing values | None |

---

### Input Attributes
| Feature | Description |
|--------|------------|
| age | Age of the client (numeric) |
| job | Type of job of the client (categorical) |
| marital | Marital status of the client (categorical) |
| education | Highest education level attained (categorical) |
| default | Has credit in default (binary) |
| balance | Average yearly balance in euros (numeric) |
| housing | Has a housing loan (binary) |
| loan | Has a personal loan (binary) |
| contact | Contact communication type (categorical) |
| day | Last contact day of the month (numeric) |
| month | Last contact month of the year (categorical) |
| duration | Duration of last contact in seconds (numeric) |
| campaign | Number of contacts during current campaign (numeric) |
| pdays | Days since last contact from previous campaign (numeric) |
| previous | Number of contacts before current campaign (numeric) |
| poutcome | Outcome of the previous campaign (categorical) |

### Target Variable
- **`y`**: Has the client subscribed to a term deposit?
  - `yes` → Subscribed
  - `no` → Not subscribed

---

## 3. Models Used and Performance Comparison
Six machine learning models were trained and evaluated using the same preprocessing pipeline and train–test split.

### Model Performance Comparison
| ML Model                 | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
| ------------------------ | -------- | ----- | --------- | ------ | -------- | ----- |
| Logistic Regression      | 0.846    | 0.908 | 0.419     | 0.814  | 0.553    | 0.509 |
| Decision Tree            | 0.875    | 0.688 | 0.464     | 0.443  | 0.453    | 0.383 |
| kNN                      | 0.896    | 0.839 | 0.591     | 0.360  | 0.447    | 0.408 |
| Naive Bayes              | 0.864    | 0.809 | 0.428     | 0.488  | 0.456    | 0.380 |
| Random Forest (Ensemble) | 0.906    | 0.929 | 0.702     | 0.340  | 0.458    | 0.446 |
| XGBoost (Ensemble)       | 0.874    | 0.926 | 0.477     | 0.802  | 0.598    | 0.554 |


---

## 4. Observations on Model Performance
| Model               | Observation                                                                                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Significant improvement in recall due to class balancing, making it effective at identifying potential subscribers, though precision is moderate.              |
| Decision Tree       | Shows stable but limited performance; lower AUC and MCC indicate weaker class separation compared to other models.                                             |
| kNN                 | Maintains good accuracy but struggles with recall, indicating difficulty capturing minority class patterns.                                                    |
| Naive Bayes         | Provides a balance between precision and recall but does not excel in overall class discrimination.                                                            |
| Random Forest       | Strong precision but lower recall suggests it remains conservative in predicting the positive class.                                                           |
| XGBoost             | Best overall performer with highest F1 score, MCC, and strong AUC, demonstrating superior ability to handle class imbalance and capture complex relationships. |


---

## 5. Conclusion
After applying class balancing techniques, the models became more effective at identifying the minority class (customers likely to subscribe). Ensemble methods—particularly XGBoost—deliver the best overall performance by achieving the strongest balance between recall and precision while maintaining high AUC and MCC. This makes **XGBoost** the most suitable model for the Bank Marketing problem, where minimizing missed potential subscribers is more critical than reducing additional marketing calls.

---

## Project Structure
```text
bank-data-app-2024DC04041/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── bank-full.csv (training)
|   └── bank.csv (validation)
│
├── model/
│   ├── train_models.py
│   └── saved_models_2024DC04041.pkl
│   └── eda.ipynb
```

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train models:
   ```bash
   python model/train_models.py
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

