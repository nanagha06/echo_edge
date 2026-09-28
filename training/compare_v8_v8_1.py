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

V8_MODEL = (
    PROJECT_ROOT
    / "models"
    / "echoedge_v8_best.keras"
)

V81_MODEL = (
    PROJECT_ROOT
    / "models"
    / "echoedge_v8_1_best.keras"
)

MANIFEST = (
    WINDOW_DATASET
    / "validation_v8.csv"
)

FEATURES = (
    WINDOW_DATASET
    / "features"
    / "validation_features.npz"
)


def load_predictions(model_path, X):

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model.predict(
        X,
        verbose=0
    ).reshape(-1)


def recording_scores(
    manifest,
    probabilities
):

    df = manifest.copy()

    df["probability"] = probabilities

    results = []

    for source_file, group in df.groupby(
        "source_file"
    ):

        # A recording is positive if it contains
        # at least one genuine positive target window.
        is_positive = (
            group["target"] == 1
        ).any()

        max_probability = (
            group["probability"].max()
        )

        best = group.loc[
            group["probability"].idxmax()
        ]

        results.append({

            "source_file":
                source_file,

            "category":
                best["category"],

            "is_positive":
                is_positive,

            "max_probability":
                float(max_probability),

            "best_label":
                best["label"],

            "best_target":
                int(best["target"]),

            "coverage":
                float(best["coverage"])
        })

    return pd.DataFrame(results)


def print_summary(
    name,
    results,
    threshold
):

    positive = results[
        results["is_positive"]
    ]

    negative = results[
        ~results["is_positive"]
    ]

    tp = (
        positive["max_probability"]
        >= threshold
    ).sum()

    fn = (
        positive["max_probability"]
        < threshold
    ).sum()

    fp = (
        negative["max_probability"]
        >= threshold
    ).sum()

    tn = (
        negative["max_probability"]
        < threshold
    ).sum()

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    print()
    print("=" * 75)
    print(name)
    print("=" * 75)

    print(
        f"TP        : {tp}"
    )

    print(
        f"FN        : {fn}"
    )

    print(
        f"FP        : {fp}"
    )

    print(
        f"TN        : {tn}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"FPR       : {fpr:.4f}"
    )


def main():

    print("=" * 75)
    print("ECHOEDGE V8 vs V8.1")
    print("VALIDATION COMPARISON")
    print("=" * 75)

    data = np.load(FEATURES)

    X = data["X"]

    manifest = pd.read_csv(
        MANIFEST
    )

    if len(X) != len(manifest):

        raise RuntimeError(
            "Feature/manifest mismatch"
        )

    print(
        f"\nValidation windows: {len(X)}"
    )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    print("\nRunning V8...")

    p_v8 = load_predictions(
        V8_MODEL,
        X
    )

    print("Running V8.1...")

    p_v81 = load_predictions(
        V81_MODEL,
        X
    )

    # ---------------------------------------------------------
    # Recording-level results
    # ---------------------------------------------------------

    r_v8 = recording_scores(
        manifest,
        p_v8
    )

    r_v81 = recording_scores(
        manifest,
        p_v81
    )

    # ---------------------------------------------------------
    # Threshold comparison
    # ---------------------------------------------------------

    for threshold in [
        0.40,
        0.42,
        0.44,
        0.46,
        0.48,
        0.50,
        0.55,
        0.60
    ]:

        print(
            "\n\nTHRESHOLD:",
            threshold
        )

        print_summary(
            "V8",
            r_v8,
            threshold
        )

        print_summary(
            "V8.1",
            r_v81,
            threshold
        )

    # ---------------------------------------------------------
    # Difficult wake_clean files
    # ---------------------------------------------------------

    difficult_files = [
        "wake_clean/wake.71d9tcfc.s1.wav",
        "wake_clean/wave20.wav",
        "wake_clean/wave33.wav",
        "wake_clean/wake.71d9tcfc.s7.wav",
        "wake_clean/wave35.wav",
        "wake_clean/wave12.wav",
        "wake_clean/wave34.wav"
    ]

    print()
    print("=" * 75)
    print("WAKE_CLEAN COMPARISON")
    print("=" * 75)

    for filename in difficult_files:

        a = r_v8[
            r_v8["source_file"] == filename
        ]

        b = r_v81[
            r_v81["source_file"] == filename
        ]

        if len(a) == 0 or len(b) == 0:
            continue

        print(
            f"\n{filename}"
        )

        print(
            f"  V8   : "
            f"{a.iloc[0]['max_probability']:.6f}"
        )

        print(
            f"  V8.1 : "
            f"{b.iloc[0]['max_probability']:.6f}"
        )

    # ---------------------------------------------------------
    # Hard negatives
    # ---------------------------------------------------------

    hard_categories = [
        "similar_words",
        "similar_phrases",
        "normal_commands",
        "background_speech"
    ]

    print()
    print("=" * 75)
    print("HARDEST HARD-NEGATIVE RECORDINGS")
    print("=" * 75)

    hard = r_v8[
        r_v8["category"].isin(
            hard_categories
        )
    ].copy()

    hard["v81_probability"] = (
        hard["source_file"]
        .map(
            r_v81.set_index(
                "source_file"
            )["max_probability"]
        )
    )

    hard = hard.sort_values(
        "max_probability",
        ascending=False
    )

    for _, row in hard.head(15).iterrows():

        print(
            f"{row['source_file']}"
        )

        print(
            f"  V8   : "
            f"{row['max_probability']:.6f}"
        )

        print(
            f"  V8.1 : "
            f"{row['v81_probability']:.6f}"
        )

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)

    print()
    print("TEST SET WAS NOT USED.")


if __name__ == "__main__":
    main()