import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle

# Path to training data (relative to this script's location)
DATA_DIR = os.path.join("..", "training_data_pH_T")
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*forTraining_oneHot_pH_T.csv")))
csv_files = [f for f in csv_files if os.path.isfile(f)]

# Shuffle and split
csv_files = shuffle(csv_files, random_state=42)
train_files = csv_files[:30]
test_files = csv_files[30:]

print("Training files:")
for f in train_files:
    print("  ", os.path.basename(f))

print("\nTest files:")
for f in test_files:
    print("  ", os.path.basename(f))

# Load and combine CSVs
def load_data(file_list):
    X, y = [], []
    for file_path in file_list:
        #print(f"Reading: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f" - Skipping empty file")
                continue
            target = df.iloc[1:, 0].astype(float).values
            features = df.iloc[1:, 1:].astype(float).values
            X.append(features)
            y.append(target)
        except Exception as e:
            print(f" - Error reading {file_path}: {e}")
    if not X:
        raise ValueError("No valid data found in provided files.")
    return np.vstack(X), np.concatenate(y)

# Normalize / de-normalize helpers
def normalize_target(y):
    return (y - 6) / 4

def denormalize_target(y_norm):
    return y_norm * 4 + 6

# Load data
X_train, y_train = load_data(train_files)
X_test, y_test = load_data(test_files)

# Normalize targets
y_train_norm = normalize_target(y_train)
y_test_norm = normalize_target(y_test)

# Train XGBoost regressor
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=20, max_depth=4)
model.fit(X_train, y_train_norm)

# Predict (normalized)
train_preds_norm = model.predict(X_train)
test_preds_norm = model.predict(X_test)

# De-normalize predictions and targets
train_preds = denormalize_target(train_preds_norm)
test_preds = denormalize_target(test_preds_norm)
y_train_actual = denormalize_target(y_train_norm)
y_test_actual = denormalize_target(y_test_norm)

# Save predictions
pd.DataFrame({
    "actual": y_train_actual,
    "predicted": train_preds
}).to_csv("train_predictions.csv", index=False)

pd.DataFrame({
    "actual": y_test_actual,
    "predicted": test_preds
}).to_csv("test_predictions.csv", index=False)

# Report metrics
print("\n📊 Evaluation Metrics (Test Set)")
print(f" - Mean Squared Error:  {mean_squared_error(y_test_actual, test_preds):.4f}")
print(f" - Mean Absolute Error: {mean_absolute_error(y_test_actual, test_preds):.4f}")
print(f" - R² Score:             {r2_score(y_test_actual, test_preds):.4f}")

import matplotlib.pyplot as plt

# Plot: Predicted vs Actual (Training Set)
plt.figure()
plt.scatter(y_train_actual, train_preds, alpha=0.5)
plt.xlabel("Actual H Shift")
plt.ylabel("Predicted H Shift")
plt.title("Training Set: Predicted vs Actual")
plt.plot([6, 10], [6, 10], 'r--')  # Reference line y = x
plt.grid(True)
plt.savefig("train_scatter.png", dpi=300)
plt.close()
print("Saved training scatter plot to train_scatter.png")

# Plot: Predicted vs Actual (Test Set)
plt.figure()
plt.scatter(y_test_actual, test_preds, alpha=0.5)
plt.xlabel("Actual H Shift")
plt.ylabel("Predicted H Shift")
plt.title("Test Set: Predicted vs Actual")
plt.plot([6, 10], [6, 10], 'r--')  # Reference line y = x
plt.grid(True)
plt.savefig("test_scatter.png", dpi=300)
plt.close()
print("Saved test scatter plot to test_scatter.png")

