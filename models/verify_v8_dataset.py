import os
import subprocess
from pathlib import Path

# ============================================================
# ECHOEDGE V8 - DATASET VERIFICATION
# ============================================================

DATASET = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8"
)

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_CODEC = "pcm_s16le"

# ============================================================
# FIND FFPROBE
# ============================================================

winget_root = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "Microsoft"
    / "WinGet"
    / "Packages"
)

candidates = []

if winget_root.exists():
    candidates = list(winget_root.rglob("ffprobe.exe"))

if not candidates:
    print("ERROR: ffprobe.exe not found.")
    raise SystemExit(1)

FFPROBE = candidates[0]

# ============================================================
# START
# ============================================================

print("=" * 70)
print("ECHOEDGE V8 - FINAL DATASET VERIFICATION")
print("=" * 70)

print(f"Dataset : {DATASET}")
print(f"FFprobe : {FFPROBE}")

if not DATASET.exists():
    print("\nERROR: dataset_v8 does not exist.")
    raise SystemExit(1)

# ============================================================
# FIND WAV FILES
# ============================================================

files = list(DATASET.rglob("*.wav"))

print(f"\nWAV files found: {len(files)}")
print()

# ============================================================
# VERIFY
# ============================================================

passed = 0
failed = 0

failed_files = []

for index, audio_file in enumerate(files, start=1):

    command = [
        str(FFPROBE),
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_file)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        failed += 1
        failed_files.append(
            (audio_file, "FFprobe error")
        )
        continue

    values = result.stdout.strip().splitlines()

    if len(values) < 3:
        failed += 1
        failed_files.append(
            (audio_file, "Missing audio information")
        )
        continue

    codec = values[0].strip()
    sample_rate = values[1].strip()
    channels = values[2].strip()

    if (
        codec == EXPECTED_CODEC
        and sample_rate == str(EXPECTED_SAMPLE_RATE)
        and channels == str(EXPECTED_CHANNELS)
    ):
        passed += 1

    else:
        failed += 1
        failed_files.append(
            (
                audio_file,
                f"codec={codec}, "
                f"sample_rate={sample_rate}, "
                f"channels={channels}"
            )
        )

# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)

print(f"Total files : {len(files)}")
print(f"PASSED      : {passed}")
print(f"FAILED      : {failed}")

# ============================================================
# FAILED FILES
# ============================================================

if failed_files:

    print("\nFILES THAT FAILED VERIFICATION:")
    print("-" * 70)

    for file, reason in failed_files:
        print(f"\n{file}")
        print(f"  {reason}")

else:

    print("\nALL FILES PASSED.")
    print()
    print("Dataset is ready for V8 ML preprocessing.")
    print()
    print("Verified format:")
    print("  WAV")
    print("  PCM signed 16-bit")
    print("  16 kHz")
    print("  Mono")

print("=" * 70)