import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path(
    r"C:\Users\nanag_ltzlj6d\esp\EchoEdge_WakeNet_Min_STREAMING"
)

WINDOW_DATASET = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8_windows"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "echoedge_v8_best.keras"
)


def evaluate_split(name, manifest_path, feature_path):

    manifest = pd.read_csv(manifest_path)

    data = np.load(feature_path)

    X = data["X"]

    if len(X) != len(manifest):
        raise RuntimeError(
            f"{name}: feature/manifest mismatch: "
            f"{len(X)} vs {len(manifest)}"
        )

    probabilities = model.predict(
        X,
        verbose=0
    ).reshape(-1)

    manifest = manifest.copy()

    manifest["probability"] = probabilities

    print()
    print("=" * 80)
    print(name.upper())
    print("=" * 80)

    # ---------------------------------------------------------
    # Category-level window statistics
    # ---------------------------------------------------------

    for category, group in manifest.groupby(
        "category"
    ):

        positives = group[
            group["target"] == 1
        ]["probability"]

        negatives = group[
            group["target"] == 0
        ]["probability"]

        print()
        print(
            f"{category:<28}"
            f" windows={len(group):4d}"
        )

        if len(positives) > 0:

            print(
                f"  POS: "
                f"n={len(positives):4d} "
                f"mean={positives.mean():.3f} "
                f"median={positives.median():.3f} "
                f"max={positives.max():.3f}"
            )

        if len(negatives) > 0:

            print(
                f"  NEG: "
                f"n={len(negatives):4d} "
                f"mean={negatives.mean():.3f} "
                f"median={negatives.median():.3f} "
                f"max={negatives.max():.3f}"
            )


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("=" * 80)
print("ECHOEDGE V8 - CATEGORY ANALYSIS")
print("=" * 80)

evaluate_split(
    "TRAIN",
    WINDOW_DATASET / "train_v8.csv",
    WINDOW_DATASET / "features" / "train_features.npz"
)

evaluate_split(
    "VALIDATION",
    WINDOW_DATASET / "validation_v8.csv",
    WINDOW_DATASET / "features" / "validation_features.npz"
)

print()
print("=" * 80)
print("DONE")
print("=" * 80)

print()
print("TEST SET WAS NOT USED.")