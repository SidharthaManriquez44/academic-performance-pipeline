from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("src/pipeline/model.pkl")

EXPECTED_COLUMNS = [
    "Gender",
    "Age",
    "Department",
    "Attendance (%)",
    "Midterm_Score",
    "Assignments_Avg",
    "Quizzes_Avg",
    "Participation_Score",
    "Projects_Score",
    "Study_Hours_per_Week",
    "Extracurricular_Activities",
    "Internet_Access_at_Home",
    "Parent_Education_Level",
    "Family_Income_Level",
    "Sleep_Hours_per_Night",
    "Stress_Level (1-10)"
]
@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = 0  # default value

    if "study_efficiency" not in df.columns:
        df["study_efficiency"] = df["Study_Hours_per_Week"] / (df["Attendance (%)"] + 1)

    prediction = model.predict(df)[0]

    return {"at_risk": int(prediction)}
