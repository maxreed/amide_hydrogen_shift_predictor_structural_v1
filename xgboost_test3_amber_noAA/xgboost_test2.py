import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle
import re

# Path to training data (relative to this script's location)
DATA_DIR = os.path.join("..", "training_data_withAmber_noAA")
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*forTraining_oneHot_cleaned.csv")))
csv_files = [f for f in csv_files if os.path.isfile(f)]

# Shuffle and split
csv_files = shuffle(csv_files, random_state=21)
train_files = csv_files[:220]
test_files = csv_files[220:]

print("Training files:")
for f in train_files:
    print("  ", os.path.basename(f))

print("\nTest files:")
for f in test_files:
    print("  ", os.path.basename(f))

def load_data(file_list):
    X, y, ids = [], [], []
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            target = df.iloc[1:, 0].astype(float).values
            

            features = df.iloc[1:, 2:].astype(float).values

            # # remove some unnecessary features
            # # Start with all candidate columns
            # columns = list(range(5, df.shape[1] - 200))
            # # Drop known bad blocks
            # drop_ranges = list(range(58, 97))
            # columns = [c for c in columns if c not in drop_ranges]
            # # Extract only the selected features
            # features = df.iloc[1:, columns].astype(float).values
            # # Print the corresponding column names
            # # print("✅ Keeping the following columns:")
            # # for i in columns:
            # #     print(f"{i}: {df.columns[i]}")
            # # exit()

            X.append(features)
            y.append(target)
            # Extract numeric ID from file name, repeat for all rows
            match = re.search(r"bmr(\d+)", os.path.basename(file_path))
            file_id = match.group(1) if match else "unknown"
            ids.extend([file_id] * features.shape[0])
        except Exception as e:
            print(f" - Error reading {file_path}: {e}")
    if not X:
        raise ValueError("No valid data found in provided files.")
    return np.vstack(X), np.concatenate(y), ids

# Normalize / de-normalize helpers
def normalize_target(y):
    return (y - 6) / 4

def denormalize_target(y_norm):
    return y_norm * 4 + 6

# Load data
X_train, y_train, train_ids = load_data(train_files)
print(X_train.shape)
X_test, y_test, test_ids = load_data(test_files)

# Normalize targets
y_train_norm = normalize_target(y_train)
y_test_norm = normalize_target(y_test)

# Train XGBoost regressor
num_estimators = 50
print(f"Number of estimators: {num_estimators}")
maximum_depth = 6
print(f"Maximum depth: {maximum_depth}")
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=num_estimators, max_depth=maximum_depth)
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
    "source_id": train_ids,
    "actual": y_train_actual,
    "predicted": train_preds
}).to_csv("train_predictions.csv", index=False)

pd.DataFrame({
    "source_id": test_ids,
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

from xgboost import plot_importance

importance_types = ["gain", "cover", "weight"]

for imp_type in importance_types:
    plt.figure(figsize=(10, 6))
    plot_importance(model, importance_type=imp_type, max_num_features=30, show_values=False)
    plt.title(f"Feature Importance by {imp_type.capitalize()}")
    plt.tight_layout()
    out_path = f"feature_importance_{imp_type}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {imp_type} plot to {out_path}")

importances_dict = {
    t: model.get_booster().get_score(importance_type=t)
    for t in importance_types
}

# Merge all importances into one DataFrame
importance_df = pd.DataFrame.from_dict(importances_dict, orient='index').T.fillna(0)
importance_df.index.name = "feature"
importance_df.to_csv("feature_importance_summary.csv")
print("Saved raw importances to feature_importance_summary.csv")
