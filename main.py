from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="Loan Approval Prediction API")

# Load model
model = joblib.load("loan_model.pkl")

# Define Pydantic schema
class LoanRequest(BaseModel):
    Gender: int
    Married: int
    Education: int
    Self_Employed: int
    Dependents: float
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float

@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}

@app.post("/predict")
def predict_loan(data: LoanRequest):
    # Convert Pydantic object to dict
    data_dict = data.dict()

    total_income = data_dict["ApplicantIncome"] + data_dict["CoapplicantIncome"]
    loan_amount_log = np.log(data_dict["LoanAmount"])
    total_income_log = np.log(total_income + 1)

    features = np.array([[
        data_dict["Gender"],
        data_dict["Married"],
        data_dict["Education"],
        data_dict["Self_Employed"],
        data_dict["Dependents"],
        data_dict["ApplicantIncome"],
        data_dict["CoapplicantIncome"],
        data_dict["LoanAmount"],
        data_dict["Loan_Amount_Term"],
        data_dict["Credit_History"],
        total_income,
        loan_amount_log,
        total_income_log
    ]])

    prediction = model.predict(features)[0]

    return {
        "loan_status": "Approved" if prediction == 1 else "Rejected"
    }
