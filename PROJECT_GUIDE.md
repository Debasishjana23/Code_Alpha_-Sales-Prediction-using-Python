# Project Guide

`train_model.py` cleans the dataset, trains Linear Regression, evaluates it, and saves `lr_model.pkl`.

`app.py` loads `lr_model.pkl` and provides a Streamlit interface for sales prediction.

`sales_predictions.csv` is the project dataset.

`requirements.txt` contains the required Python packages.

`README.md` documents the project.

`.gitignore` prevents unnecessary Python/editor files from being uploaded.

Run:
```bash
python train_model.py
streamlit run app.py
```
