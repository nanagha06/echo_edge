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


# ============================================================
# PATHS
# ============================================================

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


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("ECHOEDGE V8 - FINE RECORDING-LEVEL THRESHOLD SWEEP")
print("=" * 75)

model = tf.keras.models.load_model(MODEL_PATH)

data = np.load(FEATURES_PATH)

X = data["X"]
y = data["y"]

manifest = pd.read_csv(MANIFEST_PATH)

if len(X) != len(manifest):
    raise ValueError(
        f"Feature/manifest mismatch: {len(X)} vs {len(manifest)}"
    )

print(f"\nFeatures : {X.shape}")
print(f"Manifest : {manifest.shape}")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

probabilities = model.predict(
    X,
    batch_size=32,
    verbose=1
).reshape(-1)

manifest = manifest.copy()
manifest["_probability"] = probabilities


# ============================================================
# RECORDING-LEVEL EVALUATION
# ============================================================

def evaluate_recordings(manifest, threshold):

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    positive_recordings = []
    negative_recordings = []

    for source_file, group in manifest.groupby(
        "source_file",
        sort=False
    ):

        probabilities = group["_probability"].to_numpy()
        targets = group["target"].to_numpy()

        is_positive_recording = np.any(
            targets == 1
        )

        max_probability = float(
            np.max(probabilities)
        )

        triggered = (
            max_probability >= threshold
        )

        if is_positive_recording:

            positive_recordings.append(
                (
                    source_file,
                    max_probability,
                    triggered
                )
            )

            if triggered:
                tp += 1
            else:
                fn += 1

        else:

            negative_recordings.append(
                (
                    source_file,
                    max_probability,
                    triggered
                )
            )

            if triggered:
                fp += 1
            else:
                tn += 1

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "positive_recordings": positive_recordings,
        "negative_recordings": negative_recordings,
    }


# ============================================================
# FINE SWEEP
# ============================================================

thresholds = np.arange(
    0.20,
    0.701,
    0.01
)

results = []

print("\n")
print("=" * 105)
print("FINE RECORDING-LEVEL RESULTS")
print("=" * 105)

print(
    f"{'THR':>6}"
    f"{'TP':>6}"
    f"{'FP':>6}"
    f"{'TN':>6}"
    f"{'FN':>6}"
    f"{'PREC':>10}"
    f"{'RECALL':>10}"
    f"{'FPR':>10}"
)

print("-" * 105)

for threshold in thresholds:

    result = evaluate_recordings(
        manifest,
        threshold
    )

    results.append(result)

    print(
        f"{threshold:6.2f}"
        f"{result['tp']:6d}"
        f"{result['fp']:6d}"
        f"{result['tn']:6d}"
        f"{result['fn']:6d}"
        f"{result['precision']:10.3f}"
        f"{result['recall']:10.3f}"
        f"{result['fpr']:10.3f}"
    )


# ============================================================
# ZERO-FP THRESHOLDS
# ============================================================

zero_fp = [
    r for r in results
    if r["fp"] == 0
]

print("\n")
print("=" * 75)
print("ZERO-FALSE-POSITIVE THRESHOLDS")
print("=" * 75)

if zero_fp:

    best = max(
        zero_fp,
        key=lambda r: r["threshold"]
    )

    print(
        f"\nHighest threshold with zero false-positive "
        f"recordings: {best['threshold']:.2f}"
    )

    print(
        f"TP        : {best['tp']}"
    )

    print(
        f"FP        : {best['fp']}"
    )

    print(
        f"TN        : {best['tn']}"
    )

    print(
        f"FN        : {best['fn']}"
    )

    print(
        f"Precision : {best['precision']:.4f}"
    )

    print(
        f"Recall    : {best['recall']:.4f}"
    )

else:

    print(
        "No zero-FP threshold found."
    )


# ============================================================
# NEGATIVE RECORDING MARGIN
# ============================================================

print("\n")
print("=" * 75)
print("NEGATIVE RECORDING MAXIMUM SCORES")
print("=" * 75)

negative_scores = sorted(
    [
        (name, score)
        for name, score, triggered
        in results[0]["negative_recordings"]
    ],
    key=lambda x: x[1],
    reverse=True
)

for name, score in negative_scores:

    print(
        f"{score:.6f}  {name}"
    )


# ============================================================
# POSITIVE RECORDING SCORES
# ============================================================

print("\n")
print("=" * 75)
print("POSITIVE RECORDING MAXIMUM SCORES")
print("=" * 75)

positive_scores = sorted(
    [
        (name, score)
        for name, score, triggered
        in results[0]["positive_recordings"]
    ],
    key=lambda x: x[1],
    reverse=True
)

for name, score in positive_scores:

    print(
        f"{score:.6f}  {name}"
    )


# ============================================================
# SAVE
# ============================================================

output_path = (
    PROJECT_ROOT
    / "models"
    / "echoedge_v8_fine_threshold_results.txt"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ECHOEDGE V8 FINE RECORDING-LEVEL THRESHOLD RESULTS\n"
    )

    f.write("=" * 75 + "\n\n")

    for r in results:

        f.write(
            f"{r['threshold']:.2f},"
            f"{r['tp']},"
            f"{r['fp']},"
            f"{r['tn']},"
            f"{r['fn']},"
            f"{r['precision']:.6f},"
            f"{r['recall']:.6f},"
            f"{r['fpr']:.6f}\n"
        )

    f.write("\n\nZERO-FP THRESHOLDS\n")

    for r in zero_fp:

        f.write(
            f"threshold={r['threshold']:.2f}, "
            f"TP={r['tp']}, "
            f"FP={r['fp']}, "
            f"TN={r['tn']}, "
            f"FN={r['fn']}, "
            f"precision={r['precision']:.6f}, "
            f"recall={r['recall']:.6f}\n"
        )

print("\n")
print("=" * 75)
print("DONE")
print("=" * 75)

print("\nResults saved to:")
print(output_path)

print("\nTEST SET WAS NOT USED.")
print("TFLITE EXPORT WAS NOT PERFORMED.")