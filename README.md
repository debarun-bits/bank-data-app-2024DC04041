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
| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.90 | 0.91 | 0.65 | 0.35 | 0.45 | 0.43 |
| Decision Tree | 0.88 | 0.71 | 0.48 | 0.50 | 0.49 | 0.42 |
| kNN | 0.90 | 0.84 | 0.59 | 0.36 | 0.44 | 0.41 |
| Naive Bayes | 0.86 | 0.81 | 0.43 | 0.49 | 0.46 | 0.38 |
| Random Forest (Ensemble) | 0.90 | 0.93 | 0.66 | 0.39 | 0.49 | 0.46 |
| XGBoost (Ensemble) | 0.91 | 0.93 | 0.63 | 0.50 | 0.56 | 0.51 |

---

## 4. Observations on Model Performance
| Model | Observation |
|------|-------------|
| Logistic Regression | Strong accuracy and AUC, but low recall indicating difficulty in identifying all positive subscription cases. |
| Decision Tree | Balanced precision and recall but lower AUC, suggesting limited generalization. |
| kNN | Comparable accuracy to Logistic Regression but weaker recall and F1 score. |
| Naive Bayes | Higher recall but lower precision, resulting in more false positives. |
| Random Forest | Improved robustness and class separation over individual models. |
| XGBoost | Best overall performance with highest AUC, F1 score, and MCC. |

---

## 5. Conclusion
Ensemble methods—particularly **XGBoost**—outperform individual classifiers for the Bank Marketing dataset by effectively capturing non-linear relationships and feature interactions. While Logistic Regression provides a strong baseline, advanced ensemble techniques deliver superior performance when evaluated using AUC, F1 score, and MCC.

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

