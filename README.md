# Loan Default Prediction

End-to-end machine learning project to predict loan approval status using historical data.

## Overview
This project uses applicant details such as income, credit history, education, and employment status to predict whether a loan will be approved.

## Workflow
- Data cleaning and preprocessing
- Feature engineering
- Exploratory Data Analysis (EDA)
- Model training using Random Forest
- Model evaluation
- Model saving using joblib
- Dockerized for deployment

## Model
- Algorithm: Random Forest Classifier
- Accuracy: ~78%

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Docker

## Docker
This project is dockerized.

Build image:
docker build -t loan-default .

Run container:
docker run -p 8000:8000 loan-default

## Author
Vishal Kumar Singh  
IIT Madras
