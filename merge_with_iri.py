import os
import json
import pandas as pd
import numpy as np



# VIDEO ID MAPPING

VIDEO_ID_MAP = {
    "3300056911": 339,
    "3619085850": 340,
    "4120779505": 341,
    "4686999252": 344,
    "5361729629": 342,
    "5665902180": 343,
    "5886386918": 345,
    "6966757704": 330,
    "7160157877": 331,
    "7337141204": 332,
    "7638016176": 333
}


# Load feature file

features = pd.read_csv("iri_features_all_videos.csv")
features.columns = features.columns.str.strip().str.lower()

# IMPORTANT: ensure string type for mapping
features["video_id"] = features["video_id"].astype(str)


# Load IRI DB

iri_db = pd.read_csv("Raw_data/Waimakirir_iri.csv")
iri_db.columns = iri_db.columns.str.strip().str.lower()

# Rename columns
iri_db = iri_db.rename(columns={
    "videoid": "iri_videoid",
    "framenumber": "frame"
})


# Apply video ID mapping

features["iri_videoid"] = features["video_id"].map(VIDEO_ID_MAP)

# Drop rows where mapping not found
features = features.dropna(subset=["iri_videoid"])
features["iri_videoid"] = features["iri_videoid"].astype(int)


# Ensure correct types

features["mt"] = features["mt"].astype(int)
iri_db["frame"] = iri_db["frame"].astype(int)


# Sort for merge_asof

features = features.sort_values(["iri_videoid", "mt"])
iri_db = iri_db.sort_values(["iri_videoid", "frame"])

merged_rows = []


# Merge per video

for vid in features["iri_videoid"].unique():
    f_vid = features[features["iri_videoid"] == vid]
    i_vid = iri_db[iri_db["iri_videoid"] == vid][
        ["iri_videoid", "frame", "iri_est", "speed"]
    ]

    if i_vid.empty:
        print(f" No IRI rows for video {vid}")
        continue

    aligned = pd.merge_asof(
        f_vid,
        i_vid,
        left_on="mt",
        right_on="frame",
        direction="nearest",
        tolerance=2
    )

    merged_rows.append(aligned)


# Combine all videos

if not merged_rows:
    raise RuntimeError(" No objects to concatenate. Check VIDEO_ID_MAP and frame alignment.")

training_df = pd.concat(merged_rows, ignore_index=True)


# Keep only required columns

training_df = training_df[
    ["video_id", "mt", "z_std", "z_rms", "z_peak_to_peak", "speed_y", "iri_est"]
].rename(columns={"speed_y": "speed"})


# Drop rows without IRI
training_df = training_df.dropna(subset=["iri_est"])

training_df = training_df[training_df["iri_est"] >= 0]

# Save final training dataset

training_df.to_csv("iri_training_data.csv", index=False)

print(" Feature–IRI alignment completed")
print("Final rows:", len(training_df))
print(training_df.head())
