# Academic Performance Prediction Pipeline

## Overview

This project implements an end-to-end Machine Learning pipeline to predict whether a student is at academic risk. The system processes raw educational data, trains a classification model, and exposes predictions through an API.

The main objective is to **identify students at risk (low performance) early**, enabling timely intervention.

---

## Problem Statement

Educational institutions often detect academic failure too late.
This project addresses that by predicting:

> **Is a student at risk of failing? (Yes / No)**

---

## Tech Stack

* Python
* pandas
* scikit-learn
* xgboost
* FastAPI
* joblib

---

## Project Structure

```
academic-performance-pipeline/
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── validation/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│
├── data/
│   ├── raw/
│
├── api/
│   └── main.py
│
├── run_pipeline.py
├── model.pkl
├── requirements.txt
```

---

## Pipeline Workflow

1. **Data Ingestion**

   * Load raw student dataset

2. **Data Cleaning**

   * Handle missing values
   * Normalize dataset

3. **Data Validation**

   * Ensure data quality and schema consistency

4. **Feature Engineering**

   * Create new variables (e.g. study efficiency)
   * Encode categorical variables

5. **Model Training**

   * XGBoost classifier
   * Binary classification: at risk vs not at risk

6. **Evaluation**

   * Precision, Recall, F1-score

7. **Deployment**

   * REST API using FastAPI

---

## Data Leakage Prevention

To ensure realistic predictions, the following columns were removed:

* Final_Score
* Total_Score

These variables contain direct information about final performance.

---

## Model

* Algorithm: XGBoost Classifier
* Task: Binary classification
* Target:

```
1 → At Risk (Grade D or F)
0 → Not At Risk
```

---

## Example Prediction

### Request

```json
{
  "Gender": "Male",
  "Age": 22,
  "Department": "Engineering",
  "Attendance (%)": 85,
  "Midterm_Score": 50,
  "Assignments_Avg": 70,
  "Quizzes_Avg": 60,
  "Participation_Score": 65,
  "Projects_Score": 70,
  "Study_Hours_per_Week": 10,
  "Extracurricular_Activities": "Yes",
  "Internet_Access_at_Home": "Yes",
  "Parent_Education_Level": "High School",
  "Family_Income_Level": "Medium",
  "Sleep_Hours_per_Night": 6,
  "Stress_Level (1-10)": 7
}
```

### Response

```json
{
  "at_risk": 1
}
```

---

## Running the Project

### 1. Clone the repository

```
git clone <your-repo-url>
cd academic-performance-pipeline
```

### 2. Create virtual environment

```
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Train the model

```
python run_pipeline.py
```

### 5. Run the API

```
uvicorn api.main:app --reload
```

### 6. Access API docs

```
http://127.0.0.1:8000/docs
```

---

## Key Insights

* Academic risk can be predicted using behavioral and performance features.
* Variables such as attendance, study hours, and participation strongly influence outcomes.
* Early prediction allows proactive intervention.

---

## Limitations

* Model performance depends on data quality.
* Some assumptions were made during preprocessing (e.g., missing values).
* Real-world deployment would require continuous retraining.

---

## Future Improvements

* Model explainability (SHAP)
* MLflow for experiment tracking
* Airflow for pipeline orchestration
* Docker for deployment
* Feature store integration

---

## Author

Sidhartha Manríquez

Data Engineer / ML Enthusiast


