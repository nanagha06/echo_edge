import os

# Keep TensorFlow output clean
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

OUTPUT_PATH = (
    PROJECT_ROOT / "models" / "echoedge_v8_temporal_results.txt"
)


# ============================================================
# SETTINGS
# ============================================================

# Validation windows were generated every 250 ms.
HOP_SECONDS = 0.25

# Thresholds to investigate.
THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
]

# Temporal rules:
#
# consecutive_2:
#   2 neighboring windows above threshold
#
# consecutive_3:
#   3 neighboring windows above threshold
#
# two_of_three:
#   at least 2 of the last 3 windows above threshold
#
# two_of_four:
#   at least 2 of the last 4 windows above threshold
#
RULES = {
    "single": 1,
    "consecutive_2": 2,
    "consecutive_3": 3,
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 75)
print("ECHOEDGE V8 - TEMPORAL VALIDATION")
print("=" * 75)

print("\nLoading model:")
print(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading validation features:")
print(FEATURES_PATH)

data = np.load(FEATURES_PATH)

X_val = data["X"]
y_val = data["y"]

print(f"X shape: {X_val.shape}")
print(f"y shape: {y_val.shape}")


# ============================================================
# LOAD VALIDATION MANIFEST
# ============================================================

print("\nLoading validation manifest:")
print(MANIFEST_PATH)

if not MANIFEST_PATH.exists():

    raise FileNotFoundError(
        f"\nValidation manifest not found:\n{MANIFEST_PATH}\n\n"
        "Check that validation_v8.csv exists in dataset_v8_windows."
    )

manifest = pd.read_csv(MANIFEST_PATH)

print(f"Manifest rows: {len(manifest)}")
print("\nManifest columns:")
print(list(manifest.columns))


# ============================================================
# CHECK FEATURE / MANIFEST ALIGNMENT
# ============================================================

if len(manifest) != len(X_val):

    raise ValueError(
        "\nFeature/manifest length mismatch!\n"
        f"Features : {len(X_val)}\n"
        f"Manifest : {len(manifest)}"
    )

print("\nFeature/manifest row count matches.")


# ============================================================
# FIND IMPORTANT COLUMNS
# ============================================================

def find_column(columns, candidates):

    for candidate in candidates:

        if candidate in columns:
            return candidate

    return None


source_col = find_column(
    manifest.columns,
    [
        "source_file",
        "filename",
        "file",
        "source",
    ]
)

start_col = find_column(
    manifest.columns,
    [
        "window_start",
        "start",
        "start_seconds",
        "start_time",
    ]
)

target_col = find_column(
    manifest.columns,
    [
        "target",
        "label",
        "y",
    ]
)


if source_col is None:

    raise ValueError(
        "\nCould not find source-file column.\n"
        f"Available columns: {list(manifest.columns)}"
    )

if target_col is None:

    raise ValueError(
        "\nCould not find target/label column.\n"
        f"Available columns: {list(manifest.columns)}"
    )


print("\nUsing columns:")
print(f"Source : {source_col}")
print(f"Target : {target_col}")

if start_col:
    print(f"Start  : {start_col}")
else:
    print(
        "Start  : not found; manifest row order will be used."
    )


# ============================================================
# PREDICT
# ============================================================

print("\nGenerating validation probabilities...")

probabilities = model.predict(
    X_val,
    batch_size=32,
    verbose=1
).reshape(-1)

print("\nPrediction statistics:")
print(f"Minimum : {probabilities.min():.6f}")
print(f"Maximum : {probabilities.max():.6f}")
print(f"Mean    : {probabilities.mean():.6f}")


# ============================================================
# ADD PREDICTIONS TO MANIFEST
# ============================================================

manifest = manifest.copy()

manifest["_probability"] = probabilities
manifest["_target"] = manifest[target_col].astype(int)

if start_col:

    manifest["_start"] = pd.to_numeric(
        manifest[start_col],
        errors="coerce"
    )

else:

    manifest["_start"] = np.arange(len(manifest)) * HOP_SECONDS


# ============================================================
# TEMPORAL RULE
# ============================================================

def find_triggers(probabilities, threshold, rule):

    above = probabilities >= threshold

    triggers = []

    if rule == "single":

        for i in range(len(above)):

            if above[i]:
                triggers.append(i)

    elif rule == "consecutive_2":

        for i in range(1, len(above)):

            if above[i] and above[i - 1]:

                triggers.append(i)

    elif rule == "consecutive_3":

        for i in range(2, len(above)):

            if (
                above[i]
                and above[i - 1]
                and above[i - 2]
            ):

                triggers.append(i)

    return triggers


# ============================================================
# RECORDING-LEVEL EVALUATION
# ============================================================

def evaluate_rule(manifest, threshold, rule):

    true_positive_recordings = 0
    false_positive_recordings = 0
    true_negative_recordings = 0
    false_negative_recordings = 0

    total_positive_recordings = 0
    total_negative_recordings = 0

    positive_detection_scores = []
    negative_trigger_scores = []

    grouped = manifest.groupby(
        source_col,
        sort=False
    )

    for source_file, group in grouped:

        group = group.copy()

        if start_col:

            group = group.sort_values(
                "_start"
            )

        probs = group["_probability"].to_numpy()
        targets = group["_target"].to_numpy()

        # A recording is considered a positive recording
        # if it contains at least one genuine positive window.
        recording_is_positive = np.any(
            targets == 1
        )

        triggers = find_triggers(
            probs,
            threshold,
            rule
        )

        # No trigger.
        if len(triggers) == 0:

            if recording_is_positive:

                total_positive_recordings += 1
                false_negative_recordings += 1

            else:

                total_negative_recordings += 1
                true_negative_recordings += 1

            continue

        # There was at least one trigger.
        #
        # A trigger is treated as a genuine detection if
        # the trigger occurs on a positive target window.
        #
        # For temporal rules, also allow the trigger's
        # preceding confirmation windows to contain a
        # positive target.
        detection = False

        best_trigger_score = 0.0

        for trigger_index in triggers:

            best_trigger_score = max(
                best_trigger_score,
                float(probs[trigger_index])
            )

            if rule == "single":

                confirmation_start = trigger_index

            elif rule == "consecutive_2":

                confirmation_start = max(
                    0,
                    trigger_index - 1
                )

            elif rule == "consecutive_3":

                confirmation_start = max(
                    0,
                    trigger_index - 2
                )

            confirmation_targets = targets[
                confirmation_start:
                trigger_index + 1
            ]

            if np.any(
                confirmation_targets == 1
            ):

                detection = True
                break

        if recording_is_positive:

            total_positive_recordings += 1

            if detection:

                true_positive_recordings += 1
                positive_detection_scores.append(
                    best_trigger_score
                )

            else:

                false_negative_recordings += 1

        else:

            total_negative_recordings += 1

            if detection:

                false_positive_recordings += 1
                negative_trigger_scores.append(
                    best_trigger_score
                )

            else:

                true_negative_recordings += 1

    precision = (
        true_positive_recordings /
        (
            true_positive_recordings +
            false_positive_recordings
        )
        if (
            true_positive_recordings +
            false_positive_recordings
        ) > 0
        else 0.0
    )

    recall = (
        true_positive_recordings /
        total_positive_recordings
        if total_positive_recordings > 0
        else 0.0
    )

    fpr = (
        false_positive_recordings /
        total_negative_recordings
        if total_negative_recordings > 0
        else 0.0
    )

    specificity = (
        true_negative_recordings /
        total_negative_recordings
        if total_negative_recordings > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "rule": rule,
        "tp": true_positive_recordings,
        "fp": false_positive_recordings,
        "tn": true_negative_recordings,
        "fn": false_negative_recordings,
        "positive_recordings": total_positive_recordings,
        "negative_recordings": total_negative_recordings,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "specificity": specificity,
    }


# ============================================================
# RUN TEMPORAL TESTS
# ============================================================

all_results = []

print("\n")
print("=" * 110)
print("RECORDING-LEVEL TEMPORAL RESULTS")
print("=" * 110)

print(
    f"{'RULE':<16}"
    f"{'THR':>6}"
    f"{'TP':>6}"
    f"{'FP':>6}"
    f"{'TN':>6}"
    f"{'FN':>6}"
    f"{'PREC':>9}"
    f"{'RECALL':>9}"
    f"{'FPR':>9}"
    f"{'SPEC':>9}"
)

print("-" * 110)


for rule in RULES:

    for threshold in THRESHOLDS:

        result = evaluate_rule(
            manifest,
            threshold,
            rule
        )

        all_results.append(result)

        print(
            f"{rule:<16}"
            f"{threshold:6.2f}"
            f"{result['tp']:6d}"
            f"{result['fp']:6d}"
            f"{result['tn']:6d}"
            f"{result['fn']:6d}"
            f"{result['precision']:9.3f}"
            f"{result['recall']:9.3f}"
            f"{result['fpr']:9.3f}"
            f"{result['specificity']:9.3f}"
        )


# ============================================================
# BEST LOW-FP OPERATING POINTS
# ============================================================

print("\n")
print("=" * 75)
print("BEST LOW-FALSE-POSITIVE TEMPORAL OPERATING POINTS")
print("=" * 75)

for max_fp in [0, 1, 2, 3, 5]:

    print(f"\nFP <= {max_fp}")

    for rule in RULES:

        candidates = [
            r for r in all_results
            if r["rule"] == rule
            and r["fp"] <= max_fp
        ]

        if not candidates:

            print(
                f"  {rule:<16}: no operating point"
            )

            continue

        best = max(
            candidates,
            key=lambda r: (
                r["recall"],
                r["precision"]
            )
        )

        print(
            f"  {rule:<16}: "
            f"threshold={best['threshold']:.2f}, "
            f"TP={best['tp']}, "
            f"FP={best['fp']}, "
            f"TN={best['tn']}, "
            f"FN={best['fn']}, "
            f"precision={best['precision']:.3f}, "
            f"recall={best['recall']:.3f}, "
            f"FPR={best['fpr']:.3f}"
        )


# ============================================================
# BEST ZERO-FP POINTS
# ============================================================

print("\n")
print("=" * 75)
print("ZERO-FALSE-POSITIVE TEMPORAL OPERATING POINTS")
print("=" * 75)

for rule in RULES:

    candidates = [
        r for r in all_results
        if r["rule"] == rule
        and r["fp"] == 0
    ]

    if not candidates:

        print(
            f"{rule:<16}: none"
        )

        continue

    best = max(
        candidates,
        key=lambda r: r["recall"]
    )

    print(
        f"{rule:<16}: "
        f"threshold={best['threshold']:.2f}, "
        f"TP={best['tp']}, "
        f"FP={best['fp']}, "
        f"TN={best['tn']}, "
        f"FN={best['fn']}, "
        f"precision={best['precision']:.3f}, "
        f"recall={best['recall']:.3f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ECHOEDGE V8 TEMPORAL VALIDATION\n"
    )

    f.write("=" * 75 + "\n\n")

    f.write(
        f"Model: {MODEL_PATH}\n"
    )

    f.write(
        f"Features: {FEATURES_PATH}\n"
    )

    f.write(
        f"Manifest: {MANIFEST_PATH}\n\n"
    )

    f.write(
        "RULE,THRESHOLD,TP,FP,TN,FN,"
        "PRECISION,RECALL,FPR,SPECIFICITY\n"
    )

    for r in all_results:

        f.write(
            f"{r['rule']},"
            f"{r['threshold']:.2f},"
            f"{r['tp']},"
            f"{r['fp']},"
            f"{r['tn']},"
            f"{r['fn']},"
            f"{r['precision']:.6f},"
            f"{r['recall']:.6f},"
            f"{r['fpr']:.6f},"
            f"{r['specificity']:.6f}\n"
        )

    f.write("\n")
    f.write(
        "TEST SET WAS NOT USED.\n"
    )

    f.write(
        "TFLITE EXPORT WAS NOT PERFORMED.\n"
    )


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 75)
print("TEMPORAL VALIDATION COMPLETE")
print("=" * 75)

print("\nResults saved to:")
print(OUTPUT_PATH)

print("\nTEST SET WAS NOT USED.")
print("TFLITE EXPORT WAS NOT PERFORMED.")