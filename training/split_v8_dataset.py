import csv
import random
from collections import defaultdict, Counter
from pathlib import Path

from config_v8 import (
    WINDOW_DATASET_PATH,
    RANDOM_SEED,
)


# ============================================================
# V8 RECORDING-LEVEL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

INPUT_MANIFEST = (
    WINDOW_DATASET_PATH / "windows_v8_final.csv"
)

TRAIN_MANIFEST = (
    WINDOW_DATASET_PATH / "train_v8.csv"
)

VAL_MANIFEST = (
    WINDOW_DATASET_PATH / "validation_v8.csv"
)

TEST_MANIFEST = (
    WINDOW_DATASET_PATH / "test_v8.csv"
)


# Split ratios.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# LOAD MANIFEST
# ============================================================

def load_manifest():

    if not INPUT_MANIFEST.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n{INPUT_MANIFEST}"
        )

    with open(
        INPUT_MANIFEST,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(
            "The manifest is empty."
        )

    return rows


# ============================================================
# GROUP WINDOWS BY SOURCE RECORDING
# ============================================================

def group_by_recording(rows):

    recordings = defaultdict(list)

    for row in rows:

        source = row["source_file"]

        recordings[source].append(row)

    return recordings


# ============================================================
# DETERMINE RECORDING TYPE
# ============================================================

def recording_group(rows):

    """
    Determine the dominant target/category of a recording.

    A recording belongs to one original source file, so all
    windows from it must remain together.
    """

    targets = [
        int(row["target"])
        for row in rows
    ]

    # A positive recording has at least one positive window.
    if any(target == 1 for target in targets):
        target = 1
    else:
        target = 0

    category = rows[0]["category"]

    return target, category


# ============================================================
# STRATIFIED RECORDING SPLIT
# ============================================================

def split_recordings(recordings):

    random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # Group recordings by category.
    #
    # This helps ensure that wake_clean, repeated speech,
    # hard negatives, etc. are represented across splits.
    # --------------------------------------------------------

    by_category = defaultdict(list)

    for source_file, rows in recordings.items():

        target, category = recording_group(rows)

        by_category[
            category
        ].append(
            source_file
        )

    train_files = []
    val_files = []
    test_files = []

    # --------------------------------------------------------
    # Split each recording category separately.
    # --------------------------------------------------------

    for category in sorted(by_category):

        files = list(
            by_category[category]
        )

        random.shuffle(files)

        n = len(files)

        n_test = max(
            1,
            round(n * TEST_RATIO)
        )

        n_val = max(
            1,
            round(n * VAL_RATIO)
        )

        # Make sure at least one recording remains
        # for training.
        if n_test + n_val >= n:

            n_val = 1
            n_test = 1

        test_part = files[
            :n_test
        ]

        val_part = files[
            n_test:n_test + n_val
        ]

        train_part = files[
            n_test + n_val:
        ]

        train_files.extend(
            train_part
        )

        val_files.extend(
            val_part
        )

        test_files.extend(
            test_part
        )

    return (
        set(train_files),
        set(val_files),
        set(test_files)
    )


# ============================================================
# SAVE MANIFEST
# ============================================================

def save_manifest(
    path,
    rows
):

    if not rows:
        raise RuntimeError(
            f"No rows to save for {path}"
        )

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        path,
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


# ============================================================
# SUMMARY
# ============================================================

def summarize(name, rows):

    recordings = sorted(
        set(
            row["source_file"]
            for row in rows
        )
    )

    targets = Counter(
        int(row["target"])
        for row in rows
    )

    categories = Counter(
        row["category"]
        for row in rows
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Recordings: {len(recordings)}"
    )

    print(
        f"Windows:    {len(rows)}"
    )

    print(
        f"Positive:   {targets[1]}"
    )

    print(
        f"Negative:   {targets[0]}"
    )

    print()
    print("Recordings by category:")

    for category in sorted(categories):

        category_recordings = len(
            set(
                row["source_file"]
                for row in rows
                if row["category"] == category
            )
        )

        print(
            f"  {category:24s}: "
            f"{category_recordings:3d} recordings, "
            f"{categories[category]:4d} windows"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EchoEdge V8 Recording-Level Dataset Split")
    print("=" * 70)

    rows = load_manifest()

    recordings = group_by_recording(
        rows
    )

    print(
        f"Total recordings: {len(recordings)}"
    )

    print(
        f"Total windows:    {len(rows)}"
    )

    (
        train_files,
        val_files,
        test_files
    ) = split_recordings(
        recordings
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if (
        train_files
        & val_files
        or train_files
        & test_files
        or val_files
        & test_files
    ):

        raise RuntimeError(
            "DATA LEAKAGE: a recording appears "
            "in more than one split."
        )

    all_files = (
        train_files
        | val_files
        | test_files
    )

    if all_files != set(
        recordings.keys()
    ):

        raise RuntimeError(
            "Some recordings were lost during splitting."
        )

    # --------------------------------------------------------
    # Convert recording sets back into rows.
    # --------------------------------------------------------

    train_rows = []
    val_rows = []
    test_rows = []

    for source_file, source_rows in recordings.items():

        if source_file in train_files:

            train_rows.extend(
                source_rows
            )

        elif source_file in val_files:

            val_rows.extend(
                source_rows
            )

        elif source_file in test_files:

            test_rows.extend(
                source_rows
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_manifest(
        TRAIN_MANIFEST,
        train_rows
    )

    save_manifest(
        VAL_MANIFEST,
        val_rows
    )

    save_manifest(
        TEST_MANIFEST,
        test_rows
    )

    # --------------------------------------------------------
    # Print summaries
    # --------------------------------------------------------

    summarize(
        "TRAIN",
        train_rows
    )

    summarize(
        "VALIDATION",
        val_rows
    )

    summarize(
        "TEST",
        test_rows
    )

    # --------------------------------------------------------
    # Final safety information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)

    print(
        "No source audio files were modified."
    )

    print(
        "All windows from one recording stay in the same split."
    )

    print(
        "Train / validation / test recording overlap: NONE"
    )

    print()
    print("Files created:")

    print(TRAIN_MANIFEST)
    print(VAL_MANIFEST)
    print(TEST_MANIFEST)


if __name__ == "__main__":
    main()