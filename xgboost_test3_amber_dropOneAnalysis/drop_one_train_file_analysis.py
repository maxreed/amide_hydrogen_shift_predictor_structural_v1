import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle

DATA_DIR = os.path.join("..", "training_data_withAmber")
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
csv_files = [f for f in csv_files if os.path.isfile(f)]
csv_files = shuffle(csv_files, random_state=42)
train_files = csv_files[:225]
test_files = csv_files[225:]

# Normalize / de-normalize
def normalize_target(y): return (y - 6) / 4
def denormalize_target(y): return y * 4 + 6

# Load files into X, y
def load_data(file_list):
    X, y = [], []
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path, header=None)
            if df.empty:
                continue
            target = df.iloc[1:, 0].astype(float).values
            features = df.iloc[1:, 1:].astype(float).values
            X.append(features)
            y.append(target)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return np.vstack(X), np.concatenate(y)

# Fixed test set
X_test, y_test = load_data(test_files)
y_test_norm = normalize_target(y_test)

# Baseline with all training data
X_train_full, y_train_full = load_data(train_files)
y_train_norm_full = normalize_target(y_train_full)

num_estimators = 50
maximum_depth = 4
print(f"Number of estimators: {num_estimators}")
print(f"Maximum depth: {maximum_depth}")
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=num_estimators, max_depth=maximum_depth)
model.fit(X_train_full, y_train_norm_full)
preds = model.predict(X_test)
y_preds = denormalize_target(preds)
y_actual = denormalize_target(y_test_norm)

baseline_metrics = {
    "mse": mean_squared_error(y_actual, y_preds),
    "mae": mean_absolute_error(y_actual, y_preds),
    "r2": r2_score(y_actual, y_preds)
}

# Output table
results = []

print("✅ Baseline Evaluation:")
for k, v in baseline_metrics.items():
    print(f" - {k.upper()}: {v:.4f}")

# Loop over training files
for i, file_to_drop in enumerate(train_files):
    reduced_train = train_files[:i] + train_files[i+1:]
    X_train, y_train = load_data(reduced_train)
    y_train_norm = normalize_target(y_train)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=num_estimators, max_depth=maximum_depth)
    model.fit(X_train, y_train_norm)
    preds = model.predict(X_test)
    y_preds = denormalize_target(preds)

    mse = mean_squared_error(y_actual, y_preds)
    mae = mean_absolute_error(y_actual, y_preds)
    r2 = r2_score(y_actual, y_preds)

    results.append({
        "dropped_file": os.path.basename(file_to_drop),
        "mse": mse,
        "mae": mae,
        "r2": r2
    })
    print(f"Dropped {os.path.basename(file_to_drop)} → MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# Save CSV with baseline at top
baseline_row = {"dropped_file": "ALL_TRAINING_FILES", **baseline_metrics}
df_results = pd.DataFrame([baseline_row] + results)
df_results.to_csv("drop_one_train_file_analysis.csv", index=False)
print("📁 Results saved to drop_one_train_file_analysis.csv")
