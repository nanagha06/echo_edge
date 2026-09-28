from pathlib import Path
import wave
from collections import defaultdict, Counter

# ============================================================
# ECHOEDGE V8 DATASET AUDIT
# ============================================================

DATASET = Path(r"C:\Users\nanag_ltzlj6d\Desktop\dataset_refined")

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"
}

POSITIVE_FOLDERS = [
    "wake_clean",
    "wake_sentence_start",
    "wake_sentence_middle",
    "wake_sentence_end",
    "wake_sentence_natural",
    "wake_repeated",
    "wake_sentence_repeated",
    "wake_variation",
]

HARD_NEGATIVE_FOLDERS = [
    "similar_words",
    "similar_phrases",
    "normal_commands",
    "background_speech",
]

def get_wav_info(path):
    """Read basic WAV properties."""
    try:
        with wave.open(str(path), "rb") as w:
            channels = w.getnchannels()
            sample_rate = w.getframerate()
            sample_width = w.getsampwidth()
            frames = w.getnframes()
            duration = frames / sample_rate if sample_rate else 0
            return {
                "ok": True,
                "channels": channels,
                "sample_rate": sample_rate,
                "sample_width": sample_width,
                "duration": duration,
            }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }

def classify_file(path):
    """Return dataset category from folder structure."""
    parts = [p.lower() for p in path.parts]

    for folder in POSITIVE_FOLDERS:
        if folder.lower() in parts:
            return folder

    for folder in HARD_NEGATIVE_FOLDERS:
        if folder.lower() in parts:
            return folder

    # If a file is directly inside hard_negatives, keep that information.
    if "hard_negatives" in parts:
        return "hard_negatives_unclassified"

    return "other"

def main():
    print("=" * 70)
    print("ECHOEDGE V8 DATASET AUDIT")
    print("=" * 70)

    if not DATASET.exists():
        print(f"\nERROR: Dataset folder not found:")
        print(DATASET)
        print("\nChange DATASET at the top of this script if required.")
        return

    files = [
        p for p in DATASET.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]

    print(f"\nDataset: {DATASET}")
    print(f"Audio files found: {len(files)}")

    if not files:
        print("\nNo supported audio files found.")
        return

    counts = Counter()
    wav_stats = defaultdict(Counter)
    bad_files = []
    durations = defaultdict(list)

    for path in sorted(files):
        category = classify_file(path)
        counts[category] += 1

        if path.suffix.lower() == ".wav":
            info = get_wav_info(path)

            if not info["ok"]:
                bad_files.append((path, info["error"]))
                continue

            wav_stats[category][
                (info["sample_rate"], info["channels"], info["sample_width"])
            ] += 1

            durations[category].append(info["duration"])

    # ------------------------------------------------------------
    # CATEGORY COUNTS
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CATEGORY COUNTS")
    print("-" * 70)

    print("\nPOSITIVE:")
    positive_total = 0
    for folder in POSITIVE_FOLDERS:
        n = counts[folder]
        positive_total += n
        print(f"  {folder:<28} {n:>4}")

    print(f"  {'TOTAL POSITIVE':<28} {positive_total:>4}")

    print("\nHARD NEGATIVES:")
    negative_total = 0
    for folder in HARD_NEGATIVE_FOLDERS:
        n = counts[folder]
        negative_total += n
        print(f"  {folder:<28} {n:>4}")

    if counts["hard_negatives_unclassified"]:
        print(
            f"  {'UNCLASSIFIED inside hard_negatives':<28} "
            f"{counts['hard_negatives_unclassified']:>4}"
        )

    print(f"  {'TOTAL HARD NEGATIVE':<28} {negative_total:>4}")

    print(f"\n  {'TOTAL AUDITED':<28} {positive_total + negative_total:>4}")

    # ------------------------------------------------------------
    # WAV FORMAT SUMMARY
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("WAV FORMAT SUMMARY")
    print("-" * 70)

    for category in POSITIVE_FOLDERS + HARD_NEGATIVE_FOLDERS:
        if not wav_stats[category]:
            continue

        print(f"\n{category}:")
        for fmt, n in wav_stats[category].most_common():
            sr, ch, width = fmt
            bits = width * 8
            print(
                f"  {n:>4} files | "
                f"{sr:>6} Hz | "
                f"{ch} channel(s) | "
                f"{bits}-bit"
            )

    # ------------------------------------------------------------
    # DURATION SUMMARY
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("DURATION SUMMARY")
    print("-" * 70)

    for category in POSITIVE_FOLDERS + HARD_NEGATIVE_FOLDERS:
        values = durations[category]
        if not values:
            continue

        print(
            f"{category:<28} "
            f"count={len(values):>3}  "
            f"min={min(values):>5.2f}s  "
            f"max={max(values):>5.2f}s  "
            f"avg={sum(values)/len(values):>5.2f}s"
        )

    # ------------------------------------------------------------
    # NON-WAV FILES
    # ------------------------------------------------------------
    non_wav = [p for p in files if p.suffix.lower() != ".wav"]

    print("\n" + "-" * 70)
    print("NON-WAV FILES")
    print("-" * 70)

    if non_wav:
        ext_counts = Counter(p.suffix.lower() for p in non_wav)
        for ext, n in sorted(ext_counts.items()):
            print(f"  {ext:<10} {n}")
    else:
        print("  None")

    # ------------------------------------------------------------
    # BAD WAV FILES
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CORRUPTED / UNREADABLE WAV FILES")
    print("-" * 70)

    if bad_files:
        for path, error in bad_files:
            print(f"  {path}")
            print(f"    {error}")
    else:
        print("  None detected")

    # ------------------------------------------------------------
    # EXPECTED COUNTS
    # ------------------------------------------------------------
    expected = {
        "wake_clean": 30,
        "wake_sentence_start": 30,
        "wake_sentence_middle": 30,
        "wake_sentence_end": 25,
        "wake_sentence_natural": 30,
        "wake_repeated": 15,
        "wake_sentence_repeated": None,  # user-defined category
        "wake_variation": 30,
        "similar_words": 30,
        "similar_phrases": 30,
        "normal_commands": 30,
        "background_speech": 20,
    }

    print("\n" + "-" * 70)
    print("EXPECTED-COUNT CHECK")
    print("-" * 70)

    for category, target in expected.items():
        actual = counts[category]

        if target is None:
            status = "CHECK MANUALLY"
        elif actual == target:
            status = "OK"
        else:
            status = f"EXPECTED {target}"

        print(f"  {category:<28} {actual:>4}  {status}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    print("\nNothing in the dataset was modified.")
    print("Send the complete output here before we preprocess/train V8.")

if __name__ == "__main__":
    main()
