import pandas as pd
import matplotlib.pyplot as plt

def summarize_deviation(file_path, output_prefix):
    df = pd.read_csv(file_path)
    if 'source_id' not in df.columns:
        raise ValueError(f"{file_path} must contain a 'source_id' column.")
    
    df['abs_error'] = (df['predicted'] - df['actual']).abs()
    grouped = df.groupby('source_id')['abs_error'].mean().reset_index()
    grouped = grouped.sort_values(by='abs_error', ascending=False)

    print(f"\n🔎 File: {file_path}")
    print("Top files with highest average abs(predicted - actual):")
    print(grouped.head(10).to_string(index=False))

    # Save summary
    summary_path = f"{output_prefix}_mae_by_file.csv"
    grouped.to_csv(summary_path, index=False)
    print(f"Saved MAE summary to {summary_path}")

    # Plot histogram
    bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    plt.figure()
    plt.hist(grouped['abs_error'], bins=bins, edgecolor='black')
    plt.xlabel("Average Absolute Error")
    plt.ylabel("Number of Structures")
    plt.title(f"{output_prefix.capitalize()} MAE by Source ID")
    plt.grid(True)
    plt.savefig(f"{output_prefix}_mae_histogram.png", dpi=300)
    plt.close()
    print(f"Saved histogram to {output_prefix}_mae_histogram.png")

    return grouped

train_summary = summarize_deviation("train_predictions.csv", "train")
test_summary = summarize_deviation("test_predictions.csv", "test")
