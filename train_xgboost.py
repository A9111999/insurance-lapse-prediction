import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Name of the CSV file inside your Downloads folder
CSV_FILENAME = "Ex6GroupBChurnData.csv"
# Column you want to predict
TARGET_COLUMN = "Churn"
# Identifier columns that aren't predictive features
ID_COLUMNS = ["customerID"]

downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
csv_path = os.path.join(downloads_folder, CSV_FILENAME)

df = pd.read_csv(csv_path)

X = df.drop(columns=[TARGET_COLUMN] + ID_COLUMNS)
y = df[TARGET_COLUMN]

# TotalCharges is numeric but stored as text with some blank entries
X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
X["TotalCharges"] = X["TotalCharges"].fillna(X["TotalCharges"].median())

# Encode remaining non-numeric feature columns
for col in X.select_dtypes(exclude="number").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Encode target if it's not already numeric
if not pd.api.types.is_numeric_dtype(y):
    y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
