import os
import subprocess
from pathlib import Path

# ============================================================
# ECHOEDGE V8 - PREPARE CLEAN ML DATASET
# ============================================================

SOURCE = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_refined"
)

DESTINATION = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8"
)

CATEGORIES = [
    "wake_clean",
    "wake_sentence_start",
    "wake_sentence_middle",
    "wake_sentence_end",
    "wake_sentence_natural",
    "wake_repeated",
    "wake_sentence_repeated",
    "wake_variation",
    "hard_negatives",
]

# ============================================================
# FIND FFMPEG AUTOMATICALLY
# ============================================================

winget_root = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "Microsoft"
    / "WinGet"
    / "Packages"
)

ffmpeg_candidates = []

if winget_root.exists():
    ffmpeg_candidates = list(winget_root.rglob("ffmpeg.exe"))

if not ffmpeg_candidates:
    print("ERROR: ffmpeg.exe was not found.")
    raise SystemExit(1)

FFMPEG = ffmpeg_candidates[0]

# ============================================================
# START
# ============================================================

print("=" * 70)
print("ECHOEDGE V8 - CLEAN ML DATASET PREPARATION")
print("=" * 70)

print(f"FFmpeg      : {FFMPEG}")
print(f"Source      : {SOURCE}")
print(f"Destination : {DESTINATION}")

print("\nTarget format:")
print("  WAV")
print("  PCM signed 16-bit")
print("  16 kHz")
print("  Mono")

if not SOURCE.exists():
    print("\nERROR: Source dataset does not exist.")
    raise SystemExit(1)

DESTINATION.mkdir(parents=True, exist_ok=True)

total_success = 0
total_failed = 0

# ============================================================
# PROCESS CATEGORIES
# ============================================================

for category in CATEGORIES:

    source_dir = SOURCE / category
    dest_dir = DESTINATION / category

    print("\n" + "-" * 70)
    print(f"CATEGORY: {category}")
    print("-" * 70)

    if not source_dir.exists():
        print("WARNING: Folder not found. Skipping.")
        continue

    audio_files = [
        p for p in source_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {
            ".wav",
            ".m4a",
            ".mp3",
            ".aac",
            ".mp4",
            ".3gp",
            ".3gpp"
        }
    ]

    print(f"Audio files found: {len(audio_files)}")

    category_success = 0
    category_failed = 0

    for source_file in audio_files:

        relative_path = source_file.relative_to(source_dir)

        output_file = (
            dest_dir / relative_path
        ).with_suffix(".wav")

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        command = [
            str(FFMPEG),
            "-y",
            "-i",
            str(source_file),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_file)
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0 and output_file.exists():

            category_success += 1
            total_success += 1

        else:

            category_failed += 1
            total_failed += 1

            print(f"\nFAILED:")
            print(source_file)
            print(result.stderr[-500:])

    print(f"Converted successfully : {category_success}")
    print(f"Failed                 : {category_failed}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)

print(f"Successful conversions : {total_success}")
print(f"Failed conversions     : {total_failed}")

print("\nOutput dataset:")
print(DESTINATION)

print("\nAll successful files are:")
print("  WAV / PCM 16-bit / 16 kHz / mono")

print("\nOriginal dataset was NOT modified.")

print("=" * 70)