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

DATASET = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8"
)

WINDOW_DATASET = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8_windows"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "echoedge_v8_best.keras"
)

MANIFEST_PATH = (
    WINDOW_DATASET
    / "validation_v8.csv"
)

ANNOTATIONS_PATH = (
    DATASET
    / "annotations.csv"
)

THRESHOLD = 0.46


def main():

    print("=" * 80)
    print("ECHOEDGE V8 - WAKE_CLEAN VALIDATION ANALYSIS")
    print("=" * 80)

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded.")

    # ---------------------------------------------------------
    # Load validation data
    # ---------------------------------------------------------

    manifest = pd.read_csv(MANIFEST_PATH)

    features_path = (
        WINDOW_DATASET
        / "features"
        / "validation_features.npz"
    )

    data = np.load(features_path)

    X = data["X"]

    print(f"\nValidation windows: {len(X)}")

    if len(X) != len(manifest):
        raise RuntimeError(
            f"Feature/manifest mismatch: "
            f"{len(X)} vs {len(manifest)}"
        )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    probabilities = model.predict(
        X,
        verbose=0
    ).reshape(-1)

    manifest = manifest.copy()

    manifest["probability"] = probabilities

    # ---------------------------------------------------------
    # Only wake_clean
    # ---------------------------------------------------------

    wake = manifest[
        manifest["category"] == "wake_clean"
    ].copy()

    print(
        f"\nWake_clean validation windows: {len(wake)}"
    )

    if len(wake) == 0:

        print("No wake_clean validation data found.")
        return

    # ---------------------------------------------------------
    # Annotations
    # ---------------------------------------------------------

    annotations = pd.read_csv(
        ANNOTATIONS_PATH
    )

    # ---------------------------------------------------------
    # Analyze each recording
    # ---------------------------------------------------------

    results = []

    for source_file, group in wake.groupby(
        "source_file"
    ):

        group = group.sort_values(
            "probability",
            ascending=False
        )

        best = group.iloc[0]

        # Find annotation
        matches = annotations[
            annotations["file"]
            .astype(str)
            .apply(
                lambda x:
                str(x).replace("\\", "/")
                == str(source_file).replace("\\", "/")
            )
        ]

        if len(matches) == 0:

            matches = annotations[
                annotations["file"]
                .astype(str)
                .apply(
                    lambda x:
                    Path(x).name
                    == Path(source_file).name
                )
            ]

        if len(matches) > 0:

            annotation = matches.iloc[0]

            annotation_start = float(
                annotation["start_time"]
            )

            annotation_end = float(
                annotation["end_time"]
            )

        else:

            annotation_start = np.nan
            annotation_end = np.nan

        # Audio duration from annotation / windows
        audio_path = DATASET / source_file

        if audio_path.exists():

            import soundfile as sf

            info = sf.info(audio_path)

            duration = (
                info.frames / info.samplerate
            )

        else:

            duration = np.nan

        results.append({

            "file": source_file,

            "duration": duration,

            "annotation_duration":
                annotation_end - annotation_start
                if not np.isnan(annotation_start)
                else np.nan,

            "max_probability":
                float(best["probability"]),

            "threshold_result":
                "DETECT"
                if best["probability"] >= THRESHOLD
                else "MISS",

            "window_start":
                float(best["window_start"]),

            "window_end":
                float(best["window_end"]),

            "coverage":
                float(best["coverage"]),

            "label":
                best["label"],

            "position":
                best["position"],

            "target":
                int(best["target"]),
        })

    results = pd.DataFrame(results)

    results = results.sort_values(
        "max_probability"
    )

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("ALL WAKE_CLEAN VALIDATION RECORDINGS")
    print("=" * 80)

    for _, row in results.iterrows():

        status = (
            "OK"
            if row["threshold_result"] == "DETECT"
            else "MISS"
        )

        print(
            f"{status:4s}  "
            f"{row['max_probability']:.6f}  "
            f"duration={row['duration']:.3f}s  "
            f"coverage={row['coverage']:.3f}  "
            f"{row['file']}"
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    detected = (
        results["max_probability"]
        >= THRESHOLD
    )

    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Total wake_clean validation recordings : "
        f"{len(results)}"
    )

    print(
        f"Detected at threshold {THRESHOLD:.2f} : "
        f"{detected.sum()}"
    )

    print(
        f"Missed : "
        f"{(~detected).sum()}"
    )

    print(
        f"Recall : "
        f"{detected.mean():.3f}"
    )

    print("\nScore statistics:")

    print(
        f"Minimum : "
        f"{results['max_probability'].min():.6f}"
    )

    print(
        f"Maximum : "
        f"{results['max_probability'].max():.6f}"
    )

    print(
        f"Mean    : "
        f"{results['max_probability'].mean():.6f}"
    )

    print(
        f"Median  : "
        f"{results['max_probability'].median():.6f}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output = (
        PROJECT_ROOT
        / "models"
        / "echoedge_v8_wake_clean_analysis.csv"
    )

    results.to_csv(
        output,
        index=False
    )

    print("\nSaved to:")
    print(output)

    print("\nTEST SET WAS NOT USED.")


if __name__ == "__main__":
    main()