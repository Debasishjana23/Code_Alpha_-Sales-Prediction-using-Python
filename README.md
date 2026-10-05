# Sales Prediction Using Python

A machine learning project that predicts product sales from advertising expenditure on TV, Radio, and Newspaper.

## Objective
Build a Linear Regression model using historical advertising data and deploy it as an interactive Streamlit application.

## Model
- Algorithm: Linear Regression
- Problem: Supervised Regression
- Features: TV, Radio, Newspaper
- Target: Sales

## Workflow
1. Load dataset
2. Check data quality
3. Remove unnecessary columns
4. Remove duplicates and missing rows
5. Split into training and testing data
6. Train Linear Regression
7. Evaluate with MAE, MSE, RMSE and R2
8. Save model with Joblib
9. Run Streamlit app for prediction

## Project Structure
```text
Sales-Prediction-Using-Python/
├── app.py
├── train_model.py
├── lr_model.pkl
├── sales_predictions.csv
├── requirements.txt
├── README.md
├── PROJECT_GUIDE.md
└── .gitignore
```

## Installation
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model
```bash
python train_model.py
```

## Run the application
```bash
streamlit run app.py
```

## Technologies
Python, Pandas, NumPy, Scikit-learn, Joblib, Streamlit

## Author
Debasish Jana
