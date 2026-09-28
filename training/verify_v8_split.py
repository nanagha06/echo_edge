from pathlib import Path
import pandas as pd
from collections import Counter

# ============================================================
# EchoEdge V8 Dataset Split Verification
# ============================================================

BASE_PATH = Path(r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8_windows")

FILES = {
    "TRAIN": BASE_PATH / "train_v8.csv",
    "VALIDATION": BASE_PATH / "validation_v8.csv",
    "TEST": BASE_PATH / "test_v8.csv",
}

EXPECTED_CATEGORIES = {
    "wake_clean",
    "wake_sentence_start",
    "wake_sentence_middle",
    "wake_sentence_end",
    "wake_sentence_natural",
    "wake_repeated",
    "wake_sentence_repeated",
    "wake_variation",
    "similar_words",
    "similar_phrases",
    "normal_commands",
    "background_speech",
}

EXPECTED_TARGET_WEIGHTS = {
    0: {1.0, 0.5, 0.25},
    1: {1.0, 0.8},
}


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")

    df = pd.read_csv(path)

    print(f"\nLoaded: {path.name}")
    print(f"Rows:   {len(df)}")
    print(f"Columns: {list(df.columns)}")

    return df


def verify_columns(df):
    required = {
        "source_file",
        "category",
        "target",
        "sample_weight",
    }

    missing = required - set(df.columns)

    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return False

    print("Column check: PASS")
    return True


def verify_targets(df, name):
    errors = []

    for _, row in df.iterrows():
        target = int(row["target"])
        weight = float(row["sample_weight"])

        if target not in EXPECTED_TARGET_WEIGHTS:
            errors.append(
                f"Invalid target {target} in {row['source_file']}"
            )
            continue

        if weight not in EXPECTED_TARGET_WEIGHTS[target]:
            errors.append(
                f"Invalid weight {weight} for target {target} "
                f"in {row['source_file']}"
            )

    if errors:
        print(f"{name} target/weight check: FAIL")
        for e in errors[:20]:
            print("  ", e)

        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

        return False

    print(f"{name} target/weight check: PASS")
    return True


def verify_categories(df, name):
    categories = set(df["category"].unique())

    unexpected = categories - EXPECTED_CATEGORIES

    if unexpected:
        print(
            f"{name} category check: FAIL\n"
            f"Unexpected categories: {unexpected}"
        )
        return False

    print(f"{name} category check: PASS")
    return True


def verify_duplicate_windows(df, name):
    possible_columns = [
        "source_file",
        "window_start",
        "window_end",
    ]

    if not all(c in df.columns for c in possible_columns):
        print(
            f"{name} duplicate-window check: SKIPPED "
            f"(window position columns unavailable)"
        )
        return True

    duplicates = df.duplicated(
        subset=possible_columns,
        keep=False
    )

    duplicate_count = int(duplicates.sum())

    if duplicate_count:
        print(
            f"{name} duplicate-window check: FAIL "
            f"({duplicate_count} duplicate rows)"
        )

        print(
            df.loc[
                duplicates,
                possible_columns
            ].head(20).to_string(index=False)
        )

        return False

    print(f"{name} duplicate-window check: PASS")
    return True


def print_split_summary(df, name):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    recordings = df["source_file"].nunique()
    windows = len(df)

    positive = int((df["target"] == 1).sum())
    negative = int((df["target"] == 0).sum())

    print(f"Recordings : {recordings}")
    print(f"Windows    : {windows}")
    print(f"Positive   : {positive}")
    print(f"Negative   : {negative}")

    print("\nTarget distribution:")
    print(df["target"].value_counts().sort_index().to_string())

    print("\nSample-weight distribution:")
    print(
        df["sample_weight"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nCategory distribution:")

    category_summary = (
        df.groupby("category")
        .agg(
            recordings=("source_file", "nunique"),
            windows=("source_file", "size"),
            positive=("target", "sum"),
        )
        .sort_index()
    )

    category_summary["negative"] = (
        category_summary["windows"]
        - category_summary["positive"]
    )

    print(category_summary.to_string())

    return recordings


def verify_cross_split_leakage(dataframes):
    print("\n" + "=" * 70)
    print("CROSS-SPLIT RECORDING LEAKAGE CHECK")
    print("=" * 70)

    recording_sets = {
        name: set(df["source_file"].astype(str))
        for name, df in dataframes.items()
    }

    train_val = recording_sets["TRAIN"] & recording_sets["VALIDATION"]
    train_test = recording_sets["TRAIN"] & recording_sets["TEST"]
    val_test = recording_sets["VALIDATION"] & recording_sets["TEST"]

    passed = True

    if train_val:
        print(
            f"FAIL: TRAIN ↔ VALIDATION overlap: "
            f"{len(train_val)} recordings"
        )
        passed = False
    else:
        print("TRAIN ↔ VALIDATION: PASS")

    if train_test:
        print(
            f"FAIL: TRAIN ↔ TEST overlap: "
            f"{len(train_test)} recordings"
        )
        passed = False
    else:
        print("TRAIN ↔ TEST: PASS")

    if val_test:
        print(
            f"FAIL: VALIDATION ↔ TEST overlap: "
            f"{len(val_test)} recordings"
        )
        passed = False
    else:
        print("VALIDATION ↔ TEST: PASS")

    return passed


def verify_total_recordings(dataframes):
    print("\n" + "=" * 70)
    print("TOTAL RECORDING ACCOUNTING")
    print("=" * 70)

    all_recordings = set()

    for df in dataframes.values():
        all_recordings.update(
            df["source_file"].astype(str)
        )

    print(f"Unique recordings across splits: {len(all_recordings)}")

    expected = 379

    if len(all_recordings) != expected:
        print(
            f"WARNING: Expected {expected}, "
            f"found {len(all_recordings)}"
        )
        return False

    print(f"Expected recordings ({expected}): PASS")

    return True


def verify_total_windows(dataframes):
    print("\n" + "=" * 70)
    print("TOTAL WINDOW ACCOUNTING")
    print("=" * 70)

    total = sum(len(df) for df in dataframes.values())

    expected = 3536

    print(f"Windows across splits: {total}")
    print(f"Expected windows:      {expected}")

    if total != expected:
        print("Window accounting: FAIL")
        return False

    print("Window accounting: PASS")
    return True


def verify_recording_window_integrity(dataframes):
    print("\n" + "=" * 70)
    print("RECORDING WINDOW INTEGRITY")
    print("=" * 70)

    passed = True

    for name, df in dataframes.items():

        counts = (
            df.groupby("source_file")
            .size()
        )

        if (counts <= 0).any():
            print(f"{name}: FAIL — recording with zero windows")
            passed = False
        else:
            print(
                f"{name}: PASS — "
                f"{len(counts)} recordings all have windows"
            )

    return passed


def main():

    print("=" * 70)
    print("EchoEdge V8 DATASET SPLIT VERIFICATION")
    print("=" * 70)

    dataframes = {}

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    for name, path in FILES.items():
        dataframes[name] = load_csv(path)

    # --------------------------------------------------------
    # Individual checks
    # --------------------------------------------------------

    all_passed = True

    for name, df in dataframes.items():

        print("\n" + "-" * 70)
        print(f"CHECKING {name}")
        print("-" * 70)

        if not verify_columns(df):
            all_passed = False

        if not verify_targets(df, name):
            all_passed = False

        if not verify_categories(df, name):
            all_passed = False

        if not verify_duplicate_windows(df, name):
            all_passed = False

        print_split_summary(df, name)

    # --------------------------------------------------------
    # Cross split checks
    # --------------------------------------------------------

    if not verify_cross_split_leakage(dataframes):
        all_passed = False

    if not verify_total_recordings(dataframes):
        all_passed = False

    if not verify_total_windows(dataframes):
        all_passed = False

    if not verify_recording_window_integrity(dataframes):
        all_passed = False

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    if all_passed:
        print("ALL VERIFICATION CHECKS PASSED")
        print()
        print("Dataset is ready for the next stage.")
        print("Feature extraction has NOT been performed.")
    else:
        print("VERIFICATION FOUND PROBLEMS")
        print()
        print("DO NOT start feature extraction yet.")
        print("Fix the reported issue first.")

    print("=" * 70)


if __name__ == "__main__":
    main()