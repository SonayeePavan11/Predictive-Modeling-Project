# Predictive Modeling Using Machine Learning

## Project Overview

This project is a Machine Learning-based Loan Approval Prediction System developed using Python and Scikit-Learn. The objective is to predict whether a loan application will be approved or rejected based on applicant details such as gender, marital status, income, loan amount, and education level.

The project demonstrates the complete machine learning workflow, including data preprocessing, model training, evaluation, visualization, model persistence, and deployment through a Streamlit dashboard.

---

## Objectives

* Perform data preprocessing and feature encoding.
* Train and evaluate machine learning models.
* Compare Decision Tree and Random Forest algorithms.
* Visualize model performance using charts and metrics.
* Save the best-performing model for future predictions.
* Develop an interactive Streamlit dashboard for real-time predictions.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

---

## Dataset Features

| Feature     | Description                         |
| ----------- | ----------------------------------- |
| Gender      | Applicant Gender                    |
| Married     | Marital Status                      |
| Income      | Monthly Income                      |
| LoanAmount  | Requested Loan Amount               |
| Education   | Education Level                     |
| Loan_Status | Target Variable (Approved/Rejected) |

---

## Machine Learning Algorithms

### Decision Tree Classifier

A supervised learning algorithm used for classification by creating a tree-like structure of decisions.

### Random Forest Classifier

An ensemble learning technique that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

## Project Workflow

1. Load dataset.
2. Handle missing values.
3. Encode categorical features.
4. Split dataset into training and testing sets.
5. Train Decision Tree and Random Forest models.
6. Evaluate model performance.
7. Generate visualizations.
8. Save the best-performing model.
9. Deploy predictions through Streamlit.

---

## Visualizations Generated

* Confusion Matrix
* ROC Curve
* Feature Importance Chart
* Model Accuracy Comparison

---

## Dashboard Features

* Interactive loan approval prediction.
* User-friendly input interface.
* Dataset preview.
* Model evaluation visualizations.
* Real-time prediction results.

---

## Project Structure

```text
Predictive_Modeling_Project/
│
├── app.py
├── project.py
├── dataset.csv
├── best_model.pkl
├── confusion_matrix.png
├── roc_curve.png
├── feature_importance.png
├── accuracy_comparison.png
└── README.md
```

---

## How to Run the Project

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### Train the Model

```bash
python project.py
```

### Launch Streamlit Dashboard

```bash
streamlit run app.py
```

---

## Expected Outcome

The system predicts whether a loan application is likely to be approved or rejected based on applicant information. It provides hands-on experience with supervised machine learning, model evaluation, visualization techniques, and dashboard deployment.

---

## Learning Outcomes

* Data preprocessing and cleaning
* Feature engineering
* Supervised machine learning
* Model evaluation techniques
* Data visualization
* Streamlit application development
* Machine learning model deployment

---

## Author

**Sonayee Pavan Kumar**

Machine Learning | Data Science | Data Analytics Enthusiast
