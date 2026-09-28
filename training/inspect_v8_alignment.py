from pathlib import Path
import pandas as pd
import soundfile as sf


DATASET = Path(r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8")
WINDOW_DATASET = Path(r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8_windows")

ANNOTATIONS = DATASET / "annotations.csv"
MANIFEST = WINDOW_DATASET / "windows_v8_final.csv"

FILES = [
    "wake_clean/wave20.wav",
    "wake_clean/wave30.wav",
    "wake_clean/wave33.wav",
]


def normalize_path(path):
    return str(path).replace("\\", "/").strip()


def find_annotation_rows(annotations, filename):

    filename = normalize_path(filename)

    # Try full relative path
    matches = annotations[
        annotations["file"]
        .astype(str)
        .apply(normalize_path)
        == filename
    ]

    if len(matches) > 0:
        return matches

    # Fallback: compare filename only
    target_name = Path(filename).name

    return annotations[
        annotations["file"]
        .astype(str)
        .apply(lambda x: Path(x).name == target_name)
    ]


def find_manifest_rows(manifest, filename):

    filename = normalize_path(filename)

    matches = manifest[
        manifest["source_file"]
        .astype(str)
        .apply(normalize_path)
        == filename
    ]

    if len(matches) > 0:
        return matches

    target_name = Path(filename).name

    return manifest[
        manifest["source_file"]
        .astype(str)
        .apply(lambda x: Path(x).name == target_name)
    ]


def main():

    print("=" * 80)
    print("ECHOEDGE V8 - WAKE WORD ALIGNMENT CHECK")
    print("=" * 80)

    annotations = pd.read_csv(ANNOTATIONS)
    manifest = pd.read_csv(MANIFEST)

    print("\nAnnotation columns:")
    print(list(annotations.columns))

    print("\nManifest columns:")
    print(list(manifest.columns))

    for filename in FILES:

        print("\n" + "=" * 80)
        print("FILE:", filename)
        print("=" * 80)

        # --------------------------------------------------
        # Audio duration
        # --------------------------------------------------

        audio_path = DATASET / filename

        audio, sr = sf.read(audio_path)

        duration = len(audio) / sr

        print(f"Audio duration : {duration:.4f} s")
        print(f"Sample rate    : {sr}")

        # --------------------------------------------------
        # Annotation
        # --------------------------------------------------

        ann = find_annotation_rows(
            annotations,
            filename
        )

        print("\nANNOTATIONS:")

        if len(ann) == 0:

            print("NO ANNOTATION FOUND")

        else:

            for _, row in ann.iterrows():

                start = float(row["start_time"])
                end = float(row["end_time"])

                print(
                    f"  {start:.4f}s - {end:.4f}s"
                    f" | duration = {end - start:.4f}s"
                    f" | category = {row['category']}"
                )

        # --------------------------------------------------
        # Generated windows
        # --------------------------------------------------

        win = find_manifest_rows(
            manifest,
            filename
        )

        print("\nGENERATED WINDOWS:")

        if len(win) == 0:

            print("NO WINDOWS FOUND")

        else:

            # Sort by window start
            win = win.sort_values(
                "window_start"
            )

            for _, row in win.iterrows():

                print(
                    f"  {float(row['window_start']):.4f}s"
                    f" - {float(row['window_end']):.4f}s"
                    f" | target={int(row['target'])}"
                    f" | label={row['label']}"
                    f" | coverage={float(row['coverage']):.4f}"
                    f" | position={row['position']}"
                )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()