import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel("ERROR")


PROJECT_ROOT = Path(
    r"C:\Users\nanag_ltzlj6d\esp\EchoEdge_WakeNet_Min_STREAMING"
)

DATASET_ROOT = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8_windows"
)

MODEL_PATH = PROJECT_ROOT / "models" / "echoedge_v8_best.keras"

FEATURES_PATH = (
    DATASET_ROOT / "features" / "validation_features.npz"
)

MANIFEST_PATH = (
    DATASET_ROOT / "validation_v8.csv"
)

THRESHOLD = 0.46


print("=" * 75)
print("ECHOEDGE V8 - ANALYZE MISSED POSITIVE RECORDINGS")
print("=" * 75)

model = tf.keras.models.load_model(MODEL_PATH)

data = np.load(FEATURES_PATH)

X = data["X"]

manifest = pd.read_csv(MANIFEST_PATH)

probabilities = model.predict(
    X,
    batch_size=32,
    verbose=1
).reshape(-1)

manifest = manifest.copy()

manifest["probability"] = probabilities


# ------------------------------------------------------------
# Find positive recordings
# ------------------------------------------------------------

print("\nAnalyzing positive recordings...")

missed = []

for source_file, group in manifest.groupby(
    "source_file",
    sort=False
):

    # Positive recording = contains at least one target=1
    if not np.any(group["target"].to_numpy() == 1):
        continue

    max_probability = group["probability"].max()

    category = group["category"].iloc[0]

    if max_probability < THRESHOLD:

        best_row = group.loc[
            group["probability"].idxmax()
        ]

        missed.append({
            "category": category,
            "source_file": source_file,
            "max_probability": max_probability,
            "best_window_start": best_row["window_start"],
            "best_window_end": best_row["window_end"],
            "best_window_target": best_row["target"],
            "best_window_label": best_row["label"],
            "best_window_coverage": best_row["coverage"],
            "position": best_row["position"],
        })


# ------------------------------------------------------------
# Print missed recordings
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print(f"MISSED POSITIVE RECORDINGS AT THRESHOLD {THRESHOLD:.2f}")
print("=" * 75)

if not missed:

    print("\nNo missed positive recordings.")

else:

    for item in sorted(
        missed,
        key=lambda x: x["max_probability"]
    ):

        print("\n----------------------------------------")

        print(
            f"Category           : {item['category']}"
        )

        print(
            f"File               : {item['source_file']}"
        )

        print(
            f"Maximum score      : "
            f"{item['max_probability']:.6f}"
        )

        print(
            f"Best window        : "
            f"{item['best_window_start']:.3f}s - "
            f"{item['best_window_end']:.3f}s"
        )

        print(
            f"Window target      : "
            f"{item['best_window_target']}"
        )

        print(
            f"Window label       : "
            f"{item['best_window_label']}"
        )

        print(
            f"Coverage           : "
            f"{item['best_window_coverage']}"
        )

        print(
            f"Position            : "
            f"{item['position']}"
        )


# ------------------------------------------------------------
# Category summary
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("MISSED POSITIVES BY CATEGORY")
print("=" * 75)

if missed:

    counts = {}

    for item in missed:

        category = item["category"]

        counts[category] = (
            counts.get(category, 0) + 1
        )

    for category, count in sorted(
        counts.items(),
        key=lambda x: (-x[1], x[0])
    ):

        print(
            f"{category:<28} {count}"
        )

else:

    print("None")


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_path = (
    PROJECT_ROOT
    / "models"
    / "echoedge_v8_missed_positive_recordings.txt"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"ECHOEDGE V8 MISSED POSITIVES "
        f"AT THRESHOLD {THRESHOLD:.2f}\n"
    )

    f.write("=" * 75 + "\n\n")

    for item in sorted(
        missed,
        key=lambda x: x["max_probability"]
    ):

        f.write(
            f"Category: {item['category']}\n"
        )

        f.write(
            f"File: {item['source_file']}\n"
        )

        f.write(
            f"Maximum score: "
            f"{item['max_probability']:.6f}\n"
        )

        f.write(
            f"Best window: "
            f"{item['best_window_start']:.3f} - "
            f"{item['best_window_end']:.3f}s\n"
        )

        f.write(
            f"Target: "
            f"{item['best_window_target']}\n"
        )

        f.write(
            f"Label: "
            f"{item['best_window_label']}\n"
        )

        f.write(
            f"Coverage: "
            f"{item['best_window_coverage']}\n"
        )

        f.write(
            f"Position: "
            f"{item['position']}\n\n"
        )

print("\n")
print("=" * 75)
print("DONE")
print("=" * 75)

print("\nSaved to:")
print(output_path)

print("\nTEST SET WAS NOT USED.")