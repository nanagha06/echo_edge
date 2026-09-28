import os
import subprocess
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

SOURCE = Path(r"C:\Users\nanag_ltzlj6d\Desktop\dataset_refined\hard_negatives")

DESTINATION = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_refined_converted\hard_negatives"
)

# ============================================================
# FIND FFmpeg AUTOMATICALLY
# ============================================================

ffmpeg_candidates = []

# WinGet installation
winget_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"

if winget_root.exists():
    ffmpeg_candidates = list(winget_root.rglob("ffmpeg.exe"))

if not ffmpeg_candidates:
    print("ERROR: Could not find ffmpeg.exe automatically.")
    print("Make sure FFmpeg is installed.")
    exit(1)

FFMPEG = ffmpeg_candidates[0]

print("=" * 60)
print("ECHOEDGE V8 - HARD NEGATIVE CONVERTER")
print("=" * 60)

print(f"FFmpeg found at:")
print(FFMPEG)

print(f"\nSource:")
print(SOURCE)

print(f"\nDestination:")
print(DESTINATION)

# ============================================================
# CHECK SOURCE
# ============================================================

if not SOURCE.exists():
    print("\nERROR: Source folder does not exist.")
    exit(1)

# ============================================================
# FIND ALL AUDIO FILES
# ============================================================

files = []

for file in SOURCE.rglob("*"):
    if file.is_file():
        files.append(file)

print(f"\nFound {len(files)} files.")

if len(files) == 0:
    print("No files found.")
    exit(0)

# ============================================================
# CONVERT
# ============================================================

success = 0
failed = 0

for index, source_file in enumerate(files, start=1):

    # Preserve folder structure
    relative_path = source_file.relative_to(SOURCE)

    output_file = DESTINATION / relative_path

    # Force .wav extension
    output_file = output_file.with_suffix(".wav")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[{index}/{len(files)}]")
    print(f"Input : {source_file}")
    print(f"Output: {output_file}")

    command = [
        str(FFMPEG),

        "-y",

        "-i",
        str(source_file),

        # Convert to 16 kHz
        "-ar",
        "16000",

        # Mono
        "-ac",
        "1",

        # PCM signed 16-bit WAV
        "-c:a",
        "pcm_s16le",

        str(output_file)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode == 0 and output_file.exists():
        print("  OK")
        success += 1
    else:
        print("  FAILED")
        print(result.stderr[-1000:])
        failed += 1

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)

print(f"Successful : {success}")
print(f"Failed     : {failed}")
print(f"Total      : {len(files)}")

print(f"\nConverted files are here:")
print(DESTINATION)

print("\nOriginal files were NOT modified.")