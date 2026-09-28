from pathlib import Path
import sys

import numpy as np
import pandas as pd
import librosa


# ============================================================
# EchoEdge V8 Feature Extraction
# ============================================================
#
# Input:
#   train_v8.csv
#   validation_v8.csv
#   test_v8.csv
#
# Audio:
#   16 kHz
#   mono
#   1-second windows
#
# Features:
#   FFT = 400
#   hop = 160
#   Mel bands = 40
#   fmin = 20 Hz
#   fmax = 7600 Hz
#   power spectrogram
#   power_to_db(ref=np.max)
#   per-window standardization
#
# Output:
#   train_features.npz
#   validation_features.npz
#   test_features.npz
#
# IMPORTANT:
#   No source audio is modified.
#   No train/validation/test mixing occurs.
# ============================================================


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Users\nanag_ltzlj6d\esp\EchoEdge_WakeNet_Min_STREAMING"
)

DATASET_PATH = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8"
)

WINDOW_DATASET_PATH = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8_windows"
)

FEATURE_PATH = WINDOW_DATASET_PATH / "features"

SAMPLE_RATE = 16000

WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = 16000

N_FFT = 400
HOP_LENGTH = 160

N_MELS = 40
N_FRAMES = 101

FMIN = 20
FMAX = 7600


# ------------------------------------------------------------
# Split files
# ------------------------------------------------------------

SPLITS = {
    "train": WINDOW_DATASET_PATH / "train_v8.csv",
    "validation": WINDOW_DATASET_PATH / "validation_v8.csv",
    "test": WINDOW_DATASET_PATH / "test_v8.csv",
}


# ------------------------------------------------------------
# Audio loading
# ------------------------------------------------------------

def load_audio_window(audio_path, start_seconds):
    """
    Load exactly one 1-second audio window.

    Audio is converted to:
        16 kHz
        mono
        float32

    Short windows are zero-padded.
    Longer audio is cropped to exactly 1 second.
    """

    start_sample = int(round(start_seconds * SAMPLE_RATE))

    try:
        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True,
            offset=0.0,
            duration=None,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load audio:\n{audio_path}\n{exc}"
        ) from exc

    if start_sample < 0:
        start_sample = 0

    end_sample = start_sample + WINDOW_SAMPLES

    window = audio[start_sample:end_sample]

    # Zero-pad short windows.
    if len(window) < WINDOW_SAMPLES:
        window = np.pad(
            window,
            (0, WINDOW_SAMPLES - len(window)),
            mode="constant",
        )

    # Safety crop.
    elif len(window) > WINDOW_SAMPLES:
        window = window[:WINDOW_SAMPLES]

    return window.astype(np.float32)


# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------

def extract_log_mel_features(audio):
    """
    Extract the exact V6-style 40 x 101 Log-Mel feature.

    Pipeline:

        waveform
            ↓
        STFT
            ↓
        power spectrogram
            ↓
        Mel filterbank
            ↓
        power_to_db(ref=np.max)
            ↓
        per-window standardization
            ↓
        40 x 101
    """

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        center=True,
    )

    # Power spectrogram
    power = np.abs(stft) ** 2

    # --------------------------------------------------------
    # Mel spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        S=power,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    # --------------------------------------------------------
    # Log-Mel
    # --------------------------------------------------------

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    # --------------------------------------------------------
    # Per-window standardization
    # --------------------------------------------------------

    mean = np.mean(log_mel)
    std = np.std(log_mel)

    if std < 1e-8:
        std = 1.0

    log_mel = (log_mel - mean) / std

    # --------------------------------------------------------
    # Shape verification
    # --------------------------------------------------------

    expected_shape = (N_MELS, N_FRAMES)

    if log_mel.shape != expected_shape:

        # This should not happen with the 1-second /
        # 400 FFT / 160 hop configuration.

        raise RuntimeError(
            f"Unexpected feature shape: {log_mel.shape}. "
            f"Expected {expected_shape}."
        )

    return log_mel.astype(np.float32)


# ------------------------------------------------------------
# Resolve audio path
# ------------------------------------------------------------

def resolve_audio_path(source_file):
    """
    Resolve source_file stored in the CSV.

    The manifest may contain either:
        - an absolute path
        - a filename
        - a relative path
    """

    source_file = Path(str(source_file))

    # Absolute path
    if source_file.is_absolute():
        if source_file.exists():
            return source_file

    # Directly inside dataset_v8
    candidate = DATASET_PATH / source_file

    if candidate.exists():
        return candidate

    # Search by filename if necessary
    filename = source_file.name

    matches = list(DATASET_PATH.rglob(filename))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple audio files found for:\n"
            f"{source_file}\n"
            f"Matches:\n"
            + "\n".join(str(m) for m in matches)
        )

    raise FileNotFoundError(
        f"Could not locate source audio:\n"
        f"{source_file}"
    )


# ------------------------------------------------------------
# Process one split
# ------------------------------------------------------------

def process_split(split_name, manifest_path):
    print()
    print("=" * 70)
    print(f"FEATURE EXTRACTION: {split_name.upper()}")
    print("=" * 70)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{manifest_path}"
        )

    df = pd.read_csv(manifest_path)

    print(f"Manifest: {manifest_path}")
    print(f"Windows : {len(df)}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = {
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
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns in {manifest_path.name}: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Output arrays
    # --------------------------------------------------------

    features = []
    targets = []
    sample_weights = []

    categories = []
    source_files = []
    window_starts = []
    window_ends = []
    occurrence_indices = []
    positions = []
    coverages = []

    failures = []

    # --------------------------------------------------------
    # Process each window
    # --------------------------------------------------------

    total = len(df)

    for index, row in df.iterrows():

        source_file = str(row["source_file"])
        start_seconds = float(row["window_start"])

        try:

            audio_path = resolve_audio_path(source_file)

            audio = load_audio_window(
                audio_path,
                start_seconds,
            )

            feature = extract_log_mel_features(audio)

            features.append(feature)

            targets.append(
                int(row["target"])
            )

            sample_weights.append(
                float(row["sample_weight"])
            )

            categories.append(
                str(row["category"])
            )

            source_files.append(
                source_file
            )

            window_starts.append(
                start_seconds
            )

            window_ends.append(
                float(row["window_end"])
            )

            occurrence_value = row["occurrence_index"]

            if pd.isna(occurrence_value):
                occurrence_indices.append(-1)
            else:
                occurrence_indices.append(
                    int(occurrence_value)
                )

            positions.append(
                str(row["position"])
            )

            coverages.append(
                float(row["coverage"])
            )

        except Exception as exc:

            failures.append(
                {
                    "row": index,
                    "source_file": source_file,
                    "error": str(exc),
                }
            )

        # Progress
        completed = index + 1

        if (
            completed % 100 == 0
            or completed == total
        ):
            print(
                f"Progress: {completed}/{total}"
            )

    # --------------------------------------------------------
    # Failure check
    # --------------------------------------------------------

    if failures:

        print()
        print("=" * 70)
        print("FEATURE EXTRACTION FAILURES")
        print("=" * 70)

        for failure in failures[:20]:
            print(
                f"Row {failure['row']} | "
                f"{failure['source_file']}"
            )
            print(
                f"  {failure['error']}"
            )

        if len(failures) > 20:
            print(
                f"... and {len(failures) - 20} more failures"
            )

        raise RuntimeError(
            f"{len(failures)} windows failed feature extraction."
        )

    # --------------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------------

    X = np.stack(features).astype(np.float32)

    y = np.asarray(
        targets,
        dtype=np.int8,
    )

    weights = np.asarray(
        sample_weights,
        dtype=np.float32,
    )

    # Add channel dimension:
    #
    # (N, 40, 101)
    #       ↓
    # (N, 40, 101, 1)

    X = X[..., np.newaxis]

    # --------------------------------------------------------
    # Final shape verification
    # --------------------------------------------------------

    expected_shape = (
        len(df),
        N_MELS,
        N_FRAMES,
        1,
    )

    if X.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected final shape: {X.shape}. "
            f"Expected: {expected_shape}"
        )

    if len(y) != len(df):
        raise RuntimeError(
            "Target count does not match manifest."
        )

    if len(weights) != len(df):
        raise RuntimeError(
            "Sample-weight count does not match manifest."
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    FEATURE_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        FEATURE_PATH /
        f"{split_name}_features.npz"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        sample_weight=weights,
        category=np.asarray(
            categories,
            dtype=str,
        ),
        source_file=np.asarray(
            source_files,
            dtype=str,
        ),
        window_start=np.asarray(
            window_starts,
            dtype=np.float32,
        ),
        window_end=np.asarray(
            window_ends,
            dtype=np.float32,
        ),
        occurrence_index=np.asarray(
            occurrence_indices,
            dtype=np.int32,
        ),
        position=np.asarray(
            positions,
            dtype=str,
        ),
        coverage=np.asarray(
            coverages,
            dtype=np.float32,
        ),
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print(f"{split_name.upper()} COMPLETE")
    print("-" * 70)

    print(f"Features shape : {X.shape}")
    print(f"Targets shape  : {y.shape}")
    print(f"Weights shape  : {weights.shape}")

    print(
        f"Positive       : {np.sum(y == 1)}"
    )

    print(
        f"Negative       : {np.sum(y == 0)}"
    )

    print(
        f"Feature dtype  : {X.dtype}"
    )

    print(
        f"Feature range  : "
        f"{X.min():.4f} to {X.max():.4f}"
    )

    print(
        f"Feature mean   : {X.mean():.6f}"
    )

    print(
        f"Feature std    : {X.std():.6f}"
    )

    print(f"Saved to       : {output_file}")

    return output_file


# ------------------------------------------------------------
# Verify saved feature file
# ------------------------------------------------------------

def verify_saved_file(path, expected_count):
    print()
    print(f"Verifying: {path.name}")

    data = np.load(
        path,
        allow_pickle=False,
    )

    required_arrays = {
        "X",
        "y",
        "sample_weight",
        "category",
        "source_file",
        "window_start",
        "window_end",
        "occurrence_index",
        "position",
        "coverage",
    }

    missing = required_arrays - set(data.files)

    if missing:
        raise RuntimeError(
            f"Missing arrays in {path.name}: {missing}"
        )

    X = data["X"]
    y = data["y"]
    weights = data["sample_weight"]

    if X.shape != (
        expected_count,
        N_MELS,
        N_FRAMES,
        1,
    ):
        raise RuntimeError(
            f"Bad X shape in {path.name}: {X.shape}"
        )

    if y.shape != (expected_count,):
        raise RuntimeError(
            f"Bad y shape in {path.name}: {y.shape}"
        )

    if weights.shape != (expected_count,):
        raise RuntimeError(
            f"Bad weight shape in {path.name}: "
            f"{weights.shape}"
        )

    print("Saved-file verification: PASS")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("EchoEdge V8 FEATURE EXTRACTION")
    print("=" * 70)

    print()
    print("Configuration:")
    print(f"Sample rate      : {SAMPLE_RATE}")
    print(f"Window samples   : {WINDOW_SAMPLES}")
    print(f"FFT              : {N_FFT}")
    print(f"FFT hop          : {HOP_LENGTH}")
    print(f"Mel bands        : {N_MELS}")
    print(f"Frames           : {N_FRAMES}")
    print(f"Frequency range  : {FMIN} - {FMAX} Hz")
    print("Log-Mel          : YES")
    print("Standardization  : per-window")
    print()

    print("IMPORTANT:")
    print("Only feature extraction will be performed.")
    print("No model training will be performed.")
    print("No source audio will be modified.")
    print()

    output_files = {}

    # --------------------------------------------------------
    # Process all three splits
    # --------------------------------------------------------

    for split_name, manifest_path in SPLITS.items():

        output_files[split_name] = process_split(
            split_name,
            manifest_path,
        )

    # --------------------------------------------------------
    # Verify all saved files
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VERIFYING ALL FEATURE FILES")
    print("=" * 70)

    for split_name, output_file in output_files.items():

        manifest = pd.read_csv(
            SPLITS[split_name]
        )

        verify_saved_file(
            output_file,
            len(manifest),
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    for split_name, path in output_files.items():
        print(
            f"{split_name:12s}: {path}"
        )

    print()
    print("No model was trained.")
    print("No TFLite model was generated.")
    print("No source audio was modified.")
    print("=" * 70)


if __name__ == "__main__":
    main()