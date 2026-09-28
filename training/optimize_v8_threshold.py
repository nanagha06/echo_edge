from pathlib import Path
import numpy as np
import os

# Suppress TensorFlow startup messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from pathlib import Path
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
import tensorflow as tf

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\nanag_ltzlj6d\esp\EchoEdge_WakeNet_Min_STREAMING"
)

MODEL_PATH = PROJECT_ROOT / "models" / "echoedge_v8_best.keras"

DATASET_PATH = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8_windows"
)

VALIDATION_FEATURES = DATASET_PATH / "features" / "validation_features.npz"

OUTPUT_FILE = PROJECT_ROOT / "models" / "echoedge_v8_threshold_results.txt"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("ECHOEDGE V8 - THRESHOLD OPTIMIZATION")
print("=" * 70)

print("\nLoading model:")
print(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data:")
print(VALIDATION_FEATURES)

data = np.load(VALIDATION_FEATURES)

X_val = data["X"]
y_val = data["y"]

print(f"X shape: {X_val.shape}")
print(f"y shape: {y_val.shape}")

print(f"Positive samples: {np.sum(y_val == 1)}")
print(f"Negative samples: {np.sum(y_val == 0)}")


# ============================================================
# MODEL PREDICTIONS
# ============================================================

print("\nGenerating validation probabilities...")

probabilities = model.predict(
    X_val,
    batch_size=32,
    verbose=1
).reshape(-1)

print("\nPrediction range:")
print(f"Minimum probability : {probabilities.min():.6f}")
print(f"Maximum probability : {probabilities.max():.6f}")
print(f"Mean probability    : {probabilities.mean():.6f}")


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(np.int32)

    tp = np.sum((y_true == 1) & (predictions == 1))
    tn = np.sum((y_true == 0) & (predictions == 0))
    fp = np.sum((y_true == 0) & (predictions == 1))
    fn = np.sum((y_true == 1) & (predictions == 0))

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / len(y_true)
        if len(y_true) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "fpr": false_positive_rate,
        "specificity": specificity,
        "accuracy": accuracy,
    }


# ============================================================
# THRESHOLD SWEEP
# ============================================================

thresholds = np.arange(
    0.05,
    1.00,
    0.01
)

results = []

for threshold in thresholds:

    metrics = calculate_metrics(
        y_val,
        probabilities,
        threshold
    )

    results.append(metrics)


# ============================================================
# PRINT COMPLETE TABLE
# ============================================================

print("\n")
print("=" * 105)
print("THRESHOLD RESULTS")
print("=" * 105)

print(
    f"{'THR':>6} "
    f"{'TP':>5} "
    f"{'FP':>5} "
    f"{'TN':>5} "
    f"{'FN':>5} "
    f"{'PREC':>8} "
    f"{'RECALL':>8} "
    f"{'FPR':>8} "
    f"{'SPEC':>8} "
    f"{'ACC':>8}"
)

print("-" * 105)

for r in results:

    print(
        f"{r['threshold']:6.2f} "
        f"{r['tp']:5d} "
        f"{r['fp']:5d} "
        f"{r['tn']:5d} "
        f"{r['fn']:5d} "
        f"{r['precision']:8.3f} "
        f"{r['recall']:8.3f} "
        f"{r['fpr']:8.3f} "
        f"{r['specificity']:8.3f} "
        f"{r['accuracy']:8.3f}"
    )


# ============================================================
# FIND THRESHOLDS WITH ZERO FALSE POSITIVES
# ============================================================

zero_fp = [
    r for r in results
    if r["fp"] == 0
]

print("\n")
print("=" * 70)
print("ZERO-FALSE-POSITIVE THRESHOLDS")
print("=" * 70)

if zero_fp:

    print(
        f"{'THR':>6} "
        f"{'TP':>5} "
        f"{'TN':>5} "
        f"{'FN':>5} "
        f"{'RECALL':>10}"
    )

    print("-" * 50)

    for r in zero_fp:

        print(
            f"{r['threshold']:6.2f} "
            f"{r['tp']:5d} "
            f"{r['tn']:5d} "
            f"{r['fn']:5d} "
            f"{r['recall']:10.3f}"
        )

else:

    print("No threshold produced zero false positives.")


# ============================================================
# FIND LOW-FP OPERATING POINTS
# ============================================================

print("\n")
print("=" * 70)
print("LOW-FALSE-POSITIVE OPERATING POINTS")
print("=" * 70)

for max_fp in [1, 2, 3, 5, 10]:

    candidates = [
        r for r in results
        if r["fp"] <= max_fp
    ]

    if candidates:

        best = max(
            candidates,
            key=lambda r: r["recall"]
        )

        print(
            f"\nFP <= {max_fp}: "
            f"threshold={best['threshold']:.2f}, "
            f"TP={best['tp']}, "
            f"FP={best['fp']}, "
            f"TN={best['tn']}, "
            f"FN={best['fn']}, "
            f"precision={best['precision']:.3f}, "
            f"recall={best['recall']:.3f}"
        )

    else:

        print(
            f"\nFP <= {max_fp}: no operating point"
        )


# ============================================================
# FIND HIGHEST RECALL AT ZERO FP
# ============================================================

if zero_fp:

    best_zero_fp = max(
        zero_fp,
        key=lambda r: r["recall"]
    )

    print("\n")
    print("=" * 70)
    print("BEST ZERO-FP OPERATING POINT")
    print("=" * 70)

    print(
        f"Threshold   : {best_zero_fp['threshold']:.2f}"
    )
    print(
        f"TP          : {best_zero_fp['tp']}"
    )
    print(
        f"FP          : {best_zero_fp['fp']}"
    )
    print(
        f"TN          : {best_zero_fp['tn']}"
    )
    print(
        f"FN          : {best_zero_fp['fn']}"
    )
    print(
        f"Precision   : {best_zero_fp['precision']:.4f}"
    )
    print(
        f"Recall      : {best_zero_fp['recall']:.4f}"
    )
    print(
        f"Specificity : {best_zero_fp['specificity']:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    f.write("ECHOEDGE V8 THRESHOLD OPTIMIZATION\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        f"Model: {MODEL_PATH}\n"
    )

    f.write(
        f"Validation samples: {len(y_val)}\n"
    )

    f.write(
        f"Positive samples: {np.sum(y_val == 1)}\n"
    )

    f.write(
        f"Negative samples: {np.sum(y_val == 0)}\n\n"
    )

    f.write(
        "Threshold,TP,FP,TN,FN,Precision,Recall,FPR,Specificity,Accuracy\n"
    )

    for r in results:

        f.write(
            f"{r['threshold']:.2f},"
            f"{r['tp']},"
            f"{r['fp']},"
            f"{r['tn']},"
            f"{r['fn']},"
            f"{r['precision']:.6f},"
            f"{r['recall']:.6f},"
            f"{r['fpr']:.6f},"
            f"{r['specificity']:.6f},"
            f"{r['accuracy']:.6f}\n"
        )

    f.write("\n")

    if zero_fp:

        f.write("ZERO FALSE POSITIVE THRESHOLDS\n")
        f.write("-" * 50 + "\n")

        for r in zero_fp:

            f.write(
                f"Threshold={r['threshold']:.2f}, "
                f"TP={r['tp']}, "
                f"FP={r['fp']}, "
                f"TN={r['tn']}, "
                f"FN={r['fn']}, "
                f"Recall={r['recall']:.6f}\n"
            )

        f.write("\n")

        f.write("BEST ZERO-FP OPERATING POINT\n")
        f.write("-" * 50 + "\n")

        f.write(
            f"Threshold={best_zero_fp['threshold']:.2f}\n"
        )
        f.write(
            f"TP={best_zero_fp['tp']}\n"
        )
        f.write(
            f"FP={best_zero_fp['fp']}\n"
        )
        f.write(
            f"TN={best_zero_fp['tn']}\n"
        )
        f.write(
            f"FN={best_zero_fp['fn']}\n"
        )
        f.write(
            f"Precision={best_zero_fp['precision']:.6f}\n"
        )
        f.write(
            f"Recall={best_zero_fp['recall']:.6f}\n"
        )


print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)
print(f"Results saved to:")
print(OUTPUT_FILE)
print()
print("TEST SET WAS NOT USED.")
print("TFLITE EXPORT WAS NOT PERFORMED.")
