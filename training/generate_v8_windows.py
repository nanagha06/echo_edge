import csv
import wave
from collections import defaultdict, Counter

from config_v8 import (
    DATASET_PATH,
    WINDOW_DATASET_PATH,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    WINDOW_HOP_SAMPLES,
    POSITIVE_CATEGORIES,
    HARD_NEGATIVE_CATEGORIES,
)


# ============================================================
# V8 WINDOW GENERATOR
# ============================================================

FULL_POSITIVE_COVERAGE = 0.95
NEAR_COMPLETE_POSITIVE_COVERAGE = 0.85
PARTIAL_MIN_COVERAGE = 0.30


# ------------------------------------------------------------
# PATH HELPERS
# ------------------------------------------------------------

def normalize_path(path_string):
    return str(path_string).replace("\\", "/").strip("/")


# ------------------------------------------------------------
# LOAD ANNOTATIONS
# ------------------------------------------------------------

def load_annotations(annotation_file):
    """
    Load all ECHOEDGE annotations.

    Multiple occurrences in the same recording are supported.
    """

    annotations = defaultdict(list)

    with open(
        annotation_file,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "category",
            "file",
            "start_time",
            "end_time"
        }

        if not required.issubset(reader.fieldnames):
            raise ValueError(
                "annotations.csv must contain: "
                "category,file,start_time,end_time"
            )

        for row in reader:

            category = row["category"].strip()
            file_path = normalize_path(row["file"])

            start = float(row["start_time"])
            end = float(row["end_time"])

            if end <= start:
                print(
                    "WARNING: invalid annotation ignored:",
                    category,
                    file_path,
                    start,
                    end
                )
                continue

            annotations[
                (category, file_path)
            ].append(
                {
                    "start": start,
                    "end": end
                }
            )

    # Sort repeated occurrences by time.
    for key in annotations:
        annotations[key].sort(
            key=lambda x: x["start"]
        )

    return annotations


# ------------------------------------------------------------
# AUDIO INFORMATION
# ------------------------------------------------------------

def get_audio_duration(audio_path):

    with wave.open(str(audio_path), "rb") as wf:

        frames = wf.getnframes()
        rate = wf.getframerate()

    return frames / rate


# ------------------------------------------------------------
# COLLECT RECORDINGS
# ------------------------------------------------------------

def collect_audio_files(annotations):

    files = []

    # --------------------------------------------------------
    # POSITIVE RECORDINGS
    #
    # Only annotated positive recordings are included.
    # This means unclear/unannotated wake_clean recordings
    # are completely excluded.
    # --------------------------------------------------------

    for category in POSITIVE_CATEGORIES:

        folder = DATASET_PATH / category

        if not folder.exists():

            print(
                f"WARNING: missing folder: {folder}"
            )

            continue

        for path in sorted(folder.glob("*.wav")):

            relative_file = normalize_path(
                path.relative_to(DATASET_PATH)
            )

            key = (
                category,
                relative_file
            )

            if key in annotations:

                files.append(
                    (category, path)
                )

    # --------------------------------------------------------
    # HARD NEGATIVES
    # --------------------------------------------------------

    hard_negative_root = (
        DATASET_PATH / "hard_negatives"
    )

    for category in HARD_NEGATIVE_CATEGORIES:

        folder = (
            hard_negative_root / category
        )

        if not folder.exists():

            print(
                f"WARNING: missing folder: {folder}"
            )

            continue

        for path in sorted(
            folder.glob("*.wav")
        ):

            files.append(
                (category, path)
            )

    return files


# ------------------------------------------------------------
# OVERLAP
# ------------------------------------------------------------

def overlap_seconds(
    window_start,
    window_end,
    wake_start,
    wake_end
):

    return max(
        0.0,
        min(window_end, wake_end)
        - max(window_start, wake_start)
    )


# ------------------------------------------------------------
# CLASSIFY WINDOW
# ------------------------------------------------------------

def classify_window(
    window_start,
    window_end,
    occurrences
):

    if not occurrences:

        return (
            "negative",
            0.0,
            -1,
            "none"
        )

    best = None

    for index, occurrence in enumerate(
        occurrences
    ):

        wake_start = occurrence["start"]
        wake_end = occurrence["end"]

        wake_duration = (
            wake_end - wake_start
        )

        overlap = overlap_seconds(
            window_start,
            window_end,
            wake_start,
            wake_end
        )

        if overlap <= 0:
            continue

        coverage = (
            overlap / wake_duration
        )

        # Determine what part of ECHOEDGE
        # the window contains.
        if (
            window_start <= wake_start
            and window_end >= wake_end
        ):

            position = "complete"

        elif (
            window_start > wake_start
            and window_end < wake_end
        ):

            position = "middle_partial"

        elif window_start > wake_start:

            position = "ending_partial"

        elif window_end < wake_end:

            position = "beginning_partial"

        else:

            position = "partial"

        candidate = {
            "coverage": coverage,
            "index": index,
            "position": position
        }

        if (
            best is None
            or coverage > best["coverage"]
        ):

            best = candidate

    if best is None:

        return (
            "negative",
            0.0,
            -1,
            "none"
        )

    coverage = best["coverage"]
    index = best["index"]
    position = best["position"]

    # --------------------------------------------------------
    # COMPLETE / NEAR COMPLETE
    # --------------------------------------------------------

    if coverage >= FULL_POSITIVE_COVERAGE:

        return (
            "positive_strong",
            coverage,
            index,
            position
        )

    if coverage >= NEAR_COMPLETE_POSITIVE_COVERAGE:

        return (
            "positive_near_complete",
            coverage,
            index,
            position
        )

    # --------------------------------------------------------
    # USEFUL PARTIAL EXAMPLES
    # --------------------------------------------------------

    if coverage >= PARTIAL_MIN_COVERAGE:

        return (
            "partial_wake",
            coverage,
            index,
            position
        )

    # --------------------------------------------------------
    # ONLY A SMALL FRAGMENT OF ECHOEDGE
    # --------------------------------------------------------

    return (
        "negative_fragment",
        coverage,
        index,
        position
    )


# ------------------------------------------------------------
# MAIN GENERATOR
# ------------------------------------------------------------

def generate_windows():

    annotation_file = (
        DATASET_PATH / "annotations.csv"
    )

    if not annotation_file.exists():

        raise FileNotFoundError(
            f"Annotation file not found:\n"
            f"{annotation_file}"
        )

    annotations = load_annotations(
        annotation_file
    )

    audio_files = collect_audio_files(
        annotations
    )

    if not audio_files:

        raise RuntimeError(
            "No usable WAV files found."
        )

    WINDOW_DATASET_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_path = (
        WINDOW_DATASET_PATH
        / "windows_v8.csv"
    )

    rows = []

    counters = Counter()

    category_counters = defaultdict(
        Counter
    )

    print("=" * 70)
    print("EchoEdge V8 Window Generation")
    print("=" * 70)

    print(
        f"Dataset:       {DATASET_PATH}"
    )

    print(
        f"Window size:   {WINDOW_SAMPLES} samples"
    )

    print(
        f"Window time:   "
        f"{WINDOW_SAMPLES / SAMPLE_RATE:.3f} sec"
    )

    print(
        f"Window hop:    {WINDOW_HOP_SAMPLES} samples"
    )

    print(
        f"Window hop:    "
        f"{WINDOW_HOP_SAMPLES / SAMPLE_RATE:.3f} sec"
    )

    print(
        f"Annotations:   "
        f"{sum(len(v) for v in annotations.values())}"
    )

    print(
        f"Usable recordings: "
        f"{len(audio_files)}"
    )

    print()

    # ========================================================
    # PROCESS RECORDINGS
    # ========================================================

    for file_index, (
        category,
        audio_path
    ) in enumerate(
        audio_files,
        1
    ):

        relative_file = normalize_path(
            audio_path.relative_to(
                DATASET_PATH
            )
        )

        duration = get_audio_duration(
            audio_path
        )

        occurrences = annotations.get(
            (
                category,
                relative_file
            ),
            []
        )

        window_length = (
            WINDOW_SAMPLES / SAMPLE_RATE
        )

        max_start = (
            duration - window_length
        )

        # ====================================================
        # SHORT POSITIVE RECORDINGS
        #
        # Keep annotated recordings shorter than 1 second.
        # They will be zero-padded later during feature
        # extraction.
        # ====================================================

        if max_start < 0:

            if (
                category in POSITIVE_CATEGORIES
                and occurrences
            ):

                first_occurrence = (
                    occurrences[0]
                )

                wake_duration = (
                    first_occurrence["end"]
                    - first_occurrence["start"]
                )

                row = {
                    "category": category,
                    "source_file": relative_file,
                    "window_start": "0.0000",
                    "window_end":
                        f"{duration:.4f}",
                    "label": "short_positive",
                    "coverage":
                        f"{min(wake_duration / duration, 1.0):.4f}",
                    "occurrence_index": "1",
                    "position":
                        "short_recording"
                }

                rows.append(row)

                counters[
                    "short_positive"
                ] += 1

                category_counters[
                    category
                ]["short_positive"] += 1

                counters[
                    "recordings_processed"
                ] += 1

            else:

                counters[
                    "ignored_short_recording"
                ] += 1

            continue

        # ====================================================
        # NORMAL 1-SECOND WINDOWS
        # ====================================================

        start_time = 0.0

        while (
            start_time
            <= max_start + 1e-9
        ):

            end_time = (
                start_time
                + window_length
            )

            (
                label,
                coverage,
                occurrence_index,
                position
            ) = classify_window(
                start_time,
                end_time,
                occurrences
            )

            row = {
                "category": category,
                "source_file": relative_file,
                "window_start":
                    f"{start_time:.4f}",
                "window_end":
                    f"{end_time:.4f}",
                "label": label,
                "coverage":
                    f"{coverage:.4f}",
                "occurrence_index":
                    (
                        occurrence_index + 1
                        if occurrence_index >= 0
                        else ""
                    ),
                "position": position
            }

            rows.append(row)

            counters[label] += 1

            category_counters[
                category
            ][label] += 1

            start_time += (
                WINDOW_HOP_SAMPLES
                / SAMPLE_RATE
            )

        counters[
            "recordings_processed"
        ] += 1

        if (
            file_index % 25 == 0
            or file_index == len(audio_files)
        ):

            print(
                f"Processed "
                f"{file_index}/{len(audio_files)} "
                f"recordings..."
            )

    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    fieldnames = [
        "category",
        "source_file",
        "window_start",
        "window_end",
        "label",
        "coverage",
        "occurrence_index",
        "position"
    ]

    with open(
        manifest_path,
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

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("WINDOW GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Total windows: "
        f"{len(rows)}"
    )

    print()

    print("Overall:")

    for label in sorted(counters):

        print(
            f"  {label:24s}: "
            f"{counters[label]}"
        )

    print()

    print("By category:")

    for category in sorted(
        category_counters
    ):

        print()
        print(
            f"  {category}"
        )

        for label in sorted(
            category_counters[category]
        ):

            print(
                f"    {label:22s}: "
                f"{category_counters[category][label]}"
            )

    print()
    print(
        "Manifest saved to:"
    )
    print(
        manifest_path
    )

    print()
    print(
        "Original audio files were NOT modified."
    )
    print(
        "Unannotated positive recordings were excluded."
    )
    print(
        "Short annotated positives were retained."
    )
    print(
        "Hard negatives were retained."
    )
    print()
    print(
        "Do NOT start feature extraction yet."
    )


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    generate_windows()