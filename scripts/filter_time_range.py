"""Filter CSV rows by time range and save to a new file."""
import pandas as pd
import os

INPUT_PATH = "data/ZH_dataset/0105/data.csv"
OUTPUT_PATH = "data/ZH_dataset/0105/data_filtered.csv"
START_TIME = "2026-01-05 04:30:00"
END_TIME = "2026-01-05 08:00:00"

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(project_root, INPUT_PATH)
output_path = os.path.join(project_root, OUTPUT_PATH)

df = pd.read_csv(input_path)
print(f"Original: {df.shape[0]} rows, time range: {df['time'].min()} — {df['time'].max()}")

# Filter by time range
mask = (df["time"] >= START_TIME) & (df["time"] <= END_TIME)
filtered = df[mask]

filtered.to_csv(output_path, index=False)
print(f"Filtered: {filtered.shape[0]} rows ({START_TIME} — {END_TIME})")
print(f"Saved to: {OUTPUT_PATH}")
