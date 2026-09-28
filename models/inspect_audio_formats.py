from pathlib import Path
import subprocess
import json
import os

DATASET = Path(r"C:\Users\nanag_ltzlj6d\Desktop\dataset_refined")


# ---------------------------------------------------------
# Find FFprobe automatically
# ---------------------------------------------------------
def find_ffprobe():
    # First try PATH
    try:
        result = subprocess.run(
            ["where", "ffprobe"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            paths = result.stdout.strip().splitlines()
            if paths:
                return Path(paths[0])
    except Exception:
        pass

    # Search common WinGet installation locations
    locations = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
        Path(os.environ.get("ProgramFiles", "")),
        Path(os.environ.get("ProgramFiles(x86)", "")),
    ]

    for location in locations:
        if not location.exists():
            continue

        print(f"Searching for ffprobe.exe in:")
        print(location)

        try:
            matches = list(location.rglob("ffprobe.exe"))

            if matches:
                return matches[0]

        except PermissionError:
            continue

    return None


# ---------------------------------------------------------
# Find audio files
# ---------------------------------------------------------
def find_audio_files():

    extensions = {
        ".wav",
        ".m4a",
        ".mp4",
        ".aac",
        ".mp3",
        ".ogg",
        ".flac"
    }

    files = []

    for p in DATASET.rglob("*"):

        if p.is_file() and p.suffix.lower() in extensions:
            files.append(p)

    return sorted(files)


# ---------------------------------------------------------
# Inspect one file
# ---------------------------------------------------------
def inspect_file(ffprobe, file):

    command = [
        str(ffprobe),

        "-v",
        "error",

        "-show_entries",
        "format=format_name,format_long_name,duration",

        "-show_entries",
        "stream=index,codec_name,codec_long_name,codec_type,"
        "sample_rate,channels,bits_per_sample",

        "-of",
        "json",

        str(file)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        print("  STATUS : COULD NOT DECODE")

        if result.stderr.strip():
            print("  ERROR  :", result.stderr.strip())

        return

    try:
        data = json.loads(result.stdout)

    except json.JSONDecodeError:
        print("  STATUS : INVALID FFPROBE OUTPUT")
        return

    fmt = data.get("format", {})

    print("  Container :", fmt.get("format_name"))
    print("  Type      :", fmt.get("format_long_name"))
    print("  Duration  :", fmt.get("duration"), "sec")

    for stream in data.get("streams", []):

        if stream.get("codec_type") == "audio":

            print("  Codec     :", stream.get("codec_name"))
            print("  SampleRate:", stream.get("sample_rate"), "Hz")
            print("  Channels  :", stream.get("channels"))
            print("  BitDepth  :", stream.get("bits_per_sample"))


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    print("=" * 80)
    print("ECHOEDGE V8 AUDIO FORMAT INSPECTION")
    print("=" * 80)

    print()
    print("Locating FFprobe...")

    ffprobe = find_ffprobe()

    if ffprobe is None:

        print()
        print("ERROR: ffprobe.exe could not be located.")
        print()
        print("The FFmpeg installation exists, but Windows has")
        print("not exposed ffprobe through PATH.")
        print()

        return

    print()
    print("FFprobe found:")
    print(ffprobe)

    print()
    print("Scanning dataset...")

    files = find_audio_files()

    print(f"Files found: {len(files)}")
    print()

    print("=" * 80)

    for i, file in enumerate(files, 1):

        print()
        print(f"[{i}/{len(files)}] {file}")

        inspect_file(ffprobe, file)

    print()
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()