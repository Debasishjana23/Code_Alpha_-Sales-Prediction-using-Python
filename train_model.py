import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("sales_predictions.csv")

print("Original Data:")
print(data.head(10))

print("\nDataset Shape:")
print(data.shape)


# ==========================================
# 2. CHECK DATA
# ==========================================

print("\nColumn Names:")
print(data.columns)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:")
print(data.duplicated().sum())


# ==========================================
# 3. REMOVE UNNECESSARY COLUMN
# ==========================================

if "Unnamed: 0" in data.columns:
    data.drop("Unnamed: 0", axis=1, inplace=True)


# ==========================================
# 4. REMOVE DUPLICATE ROWS
# ==========================================

data.drop_duplicates(inplace=True)


# ==========================================
# 5. HANDLE MISSING VALUES
# ==========================================

data.dropna(inplace=True)


# ==========================================
# 6. DISPLAY CLEAN DATA
# ==========================================

print("\nClean Data:")
print(data.head(10))

print("\nClean Data Shape:")
print(data.shape)


# ==========================================
# 7. DEFINE INPUT AND OUTPUT
# ==========================================

X = data[["TV", "Radio", "Newspaper"]]

y = data["Sales"]


# ==========================================
# 8. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==========================================
# 9. CREATE LINEAR REGRESSION MODEL
# ==========================================

model = LinearRegression()


# ==========================================
# 10. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 11. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 12. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n========== MODEL PERFORMANCE ==========")

print("Mean Absolute Error:", mae)

print("Mean Squared Error:", mse)

print("Root Mean Squared Error:", rmse)

print("R2 Score:", r2)


# ==========================================
# 13. MODEL COEFFICIENTS
# ==========================================

print("\n========== MODEL DETAILS ==========")

print("TV Coefficient:", model.coef_[0])

print("Radio Coefficient:", model.coef_[1])

print("Newspaper Coefficient:", model.coef_[2])

print("Intercept:", model.intercept_)


# ==========================================
# 14. DISPLAY LINEAR REGRESSION EQUATION
# ==========================================

print("\n========== REGRESSION EQUATION ==========")

print(
    "Sales =",
    model.intercept_,
    "+",
    model.coef_[0], "* TV",
    "+",
    model.coef_[1], "* Radio",
    "+",
    model.coef_[2], "* Newspaper"
)


# ==========================================
# 15. SAVE TRAINED MODEL
# ==========================================

joblib.dump(model, "lr_model.pkl")

print("\nModel saved successfully as lr_model.pkl")


# ==========================================
# 16. LOAD SAVED MODEL
# ==========================================

loaded_model = joblib.load("lr_model.pkl")

print("\nSaved model loaded successfully.")


# ==========================================
# 17. TEST WITH NEW DATA
# ==========================================

new_data = pd.DataFrame(
    [[200, 40, 30]],
    columns=["TV", "Radio", "Newspaper"]
)

prediction = loaded_model.predict(new_data)


print("\n========== SALES PREDICTION ==========")

print("TV Advertising:", 200)

print("Radio Advertising:", 40)

print("Newspaper Advertising:", 30)

print("Predicted Sales:", prediction[0])