import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def analyze_offsets(file_path, output_prefix, num_plots=5):
    df = pd.read_csv(file_path)
    df['abs_error'] = (df['predicted'] - df['actual']).abs()
    source_ids = df['source_id'].unique()

    x_vals = np.round(np.arange(-1, 1.01, 0.04), 2)
    results = {}

    for source_id in source_ids:
        group = df[df['source_id'] == source_id]
        errors = []
        for x in x_vals:
            shifted = group['actual'] + x
            mae = np.mean(np.abs(group['predicted'] - shifted))
            errors.append(mae)
        best_x = x_vals[np.argmin(errors)]
        results[source_id] = errors + [best_x]

    # Create DataFrame
    columns = [f"x={x:.2f}" for x in x_vals] + ["best_x"]
    error_df = pd.DataFrame.from_dict(results, orient='index', columns=columns)
    error_df.index.name = "source_id"

    # Save to CSV
    output_csv = f"{output_prefix}_offset_analysis.csv"
    error_df.to_csv(output_csv)
    print(f"Saved offset analysis to {output_csv}")

    # Find top N source_ids with highest |best_x|
    top_ids = (
        error_df["best_x"]
        .astype(float)
        .abs()
        .sort_values(ascending=False)
        .head(num_plots)
        .index
    )

    for source_id in top_ids:
        errors = error_df.loc[source_id][:-1].astype(float).values
        plt.figure()
        plt.plot(x_vals, errors, marker='o')
        plt.xlabel("Offset x (added to actual values)")
        plt.ylabel("Mean Absolute Error")
        plt.title(f"{output_prefix.capitalize()} — source_id {source_id} (best_x = {error_df.loc[source_id]['best_x']})")
        plt.grid(True)
        plt.savefig(f"{output_prefix}_offset_curve_{source_id}.png", dpi=300)
        plt.close()
        print(f"Saved plot for source_id {source_id}")

    # Histogram of best_x across all structures
    best_x_values = error_df["best_x"].astype(float)
    plt.figure()
    plt.hist(best_x_values, bins=np.arange(-1, 1.05, 0.05), edgecolor='black')
    plt.xlabel("Ideal Offset x")
    plt.ylabel("Number of Structures")
    plt.title(f"Distribution of Best x — {output_prefix.capitalize()} Set")
    plt.grid(True)
    plt.savefig(f"{output_prefix}_best_x_histogram.png", dpi=300)
    plt.close()
    print(f"Saved best_x histogram to {output_prefix}_best_x_histogram.png")


# Run on both datasets
analyze_offsets("train_predictions.csv", "train")
analyze_offsets("test_predictions.csv", "test")
