import json
import numpy as np
import pandas as pd
import os

RAW_DATA_DIR = "Raw_data"
OUTPUT_FILE = "iri_features_all_videos.csv"

rows = []

for filename in os.listdir(RAW_DATA_DIR):
    # Process only JSON sensor files
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(RAW_DATA_DIR, filename)

    with open(file_path, "r") as f:
        data = json.load(f)


    video_id = filename.split("video_")[-1].replace(".json", "")

    for entry in data:
        accel = entry.get("accelerometer", [])
        if not accel:
            continue

        # Extract Z-axis values
        z = np.array([a["z"] for a in accel])

        # Compute features
        z_std = np.std(z)
        z_rms = np.sqrt(np.mean(z ** 2))
        z_peak_to_peak = np.max(z) - np.min(z)

        speed = float(entry.get("speed", 0))
        MT = entry.get("MT")

        rows.append({
            "video_id": video_id,
            "MT": MT,
            "z_std": z_std,
            "z_rms": z_rms,
            "z_peak_to_peak": z_peak_to_peak,
            "speed": speed,


        })

# Create DataFrame
df_features = pd.DataFrame(rows)

# Save combined feature file
df_features.to_csv(OUTPUT_FILE, index=False)

print("Feature extraction completed successfully")
print("Total rows extracted:", len(df_features))
print(df_features.head())

# ---- SANITY CHECKS ----
print("\n Missing values check:")
print(df_features.isnull().sum())

print("\n Feature statistics:")
print(df_features.describe())

