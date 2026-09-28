from pathlib import Path
import pandas as pd
import numpy as np


BASE_PATH = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8_windows"
)

FILES = {
    "TRAIN": BASE_PATH / "train_v8.csv",
    "VALIDATION": BASE_PATH / "validation_v8.csv",
    "TEST": BASE_PATH / "test_v8.csv",
}


def inspect_split(name, path):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    df = pd.read_csv(path)

    print(f"Rows: {len(df)}")

    print("\nNaN counts:")
    print(
        df.isna()
        .sum()
        .to_string()
    )

    print("\nwindow_start statistics:")

    valid_start = df["window_start"].dropna()

    if len(valid_start) > 0:
        print(
            valid_start.describe()
            .to_string()
        )

    nan_rows = df[
        df["window_start"].isna()
    ]

    print()
    print(
        f"Rows with NaN window_start: {len(nan_rows)}"
    )

    if len(nan_rows) > 0:

        print("\nNaN rows by category:")

        print(
            nan_rows["category"]
            .value_counts()
            .to_string()
        )

        print("\nNaN rows by label:")

        print(
            nan_rows["label"]
            .value_counts(dropna=False)
            .to_string()
        )

        print("\nNaN rows by target:")

        print(
            nan_rows["target"]
            .value_counts(dropna=False)
            .to_string()
        )

        print("\nFirst 20 NaN rows:")

        columns = [
            "category",
            "source_file",
            "window_start",
            "window_end",
            "label",
            "coverage",
            "occurrence_index",
            "position",
            "target",
            "sample_weight",
        ]

        print(
            nan_rows[columns]
            .head(20)
            .to_string(index=False)
        )


def main():

    print("=" * 70)
    print("EchoEdge V8 MANIFEST NaN INSPECTION")
    print("=" * 70)

    for name, path in FILES.items():

        if not path.exists():
            print(
                f"\nERROR: Missing file:\n{path}"
            )
            continue

        inspect_split(
            name,
            path,
        )

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()