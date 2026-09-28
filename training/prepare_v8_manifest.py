import csv
from pathlib import Path

from config_v8 import WINDOW_DATASET_PATH


INPUT_MANIFEST = (
    WINDOW_DATASET_PATH / "windows_v8.csv"
)

OUTPUT_MANIFEST = (
    WINDOW_DATASET_PATH / "windows_v8_final.csv"
)


# ============================================================
# FINAL V8 LABEL / WEIGHT STRATEGY
# ============================================================

LABEL_CONFIG = {
    "positive_strong": {
        "label": 1,
        "weight": 1.0,
    },

    "positive_near_complete": {
        "label": 1,
        "weight": 0.8,
    },

    "short_positive": {
        "label": 1,
        "weight": 1.0,
    },

    "partial_wake": {
        "label": 0,
        "weight": 0.25,
    },

    "negative_fragment": {
        "label": 0,
        "weight": 0.5,
    },

    "negative": {
        "label": 0,
        "weight": 1.0,
    },
}


def prepare_manifest():

    if not INPUT_MANIFEST.exists():

        raise FileNotFoundError(
            f"Input manifest not found:\n"
            f"{INPUT_MANIFEST}"
        )

    rows = []

    label_counts = {
        0: 0,
        1: 0,
    }

    weight_counts = {}

    with open(
        INPUT_MANIFEST,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            original_label = row["label"]

            if original_label not in LABEL_CONFIG:

                raise ValueError(
                    f"Unknown label: "
                    f"{original_label}"
                )

            config = LABEL_CONFIG[
                original_label
            ]

            row["target"] = str(
                config["label"]
            )

            row["sample_weight"] = str(
                config["weight"]
            )

            rows.append(row)

            label_counts[
                config["label"]
            ] += 1

            weight_counts[
                original_label
            ] = weight_counts.get(
                original_label,
                0
            ) + 1

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fieldnames = list(rows[0].keys())

    with open(
        OUTPUT_MANIFEST,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("=" * 70)
    print("EchoEdge V8 Final Manifest")
    print("=" * 70)

    print(
        f"Input windows:  {len(rows)}"
    )

    print()

    print("Original categories:")

    for category, count in weight_counts.items():

        config = LABEL_CONFIG[category]

        print(
            f"  {category:24s} "
            f"-> target={config['label']} "
            f"weight={config['weight']} "
            f"count={count}"
        )

    print()

    print("Final targets:")

    print(
        f"  NEGATIVE (0): {label_counts[0]}"
    )

    print(
        f"  POSITIVE (1): {label_counts[1]}"
    )

    print()

    print(
        "Output:"
    )

    print(
        OUTPUT_MANIFEST
    )

    print()

    print(
        "All windows have been retained."
    )


if __name__ == "__main__":
    prepare_manifest()