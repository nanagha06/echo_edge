from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf


DATASET = Path(r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8")
WINDOW_DATASET = Path(r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8_windows")

TRAIN_MANIFEST = WINDOW_DATASET / "train_v8.csv"

FILES = [
    "wake_clean/wave20.wav",
    "wake_clean/wave33.wav",
]


def extract_feature(path):
    audio, sr = librosa.load(
        path,
        sr=16000,
        mono=True
    )

    # Exactly one 1-second window
    if len(audio) < 16000:
        audio = np.pad(
            audio,
            (0, 16000 - len(audio))
        )
    else:
        audio = audio[:16000]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_fft=400,
        hop_length=160,
        n_mels=40,
        fmin=20,
        fmax=7600,
        power=2
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Match V8 preprocessing
    mel_db = (
        mel_db - np.mean(mel_db)
    ) / (
        np.std(mel_db) + 1e-8
    )

    # 101 frames
    if mel_db.shape[1] < 101:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, 101 - mel_db.shape[1]))
        )
    elif mel_db.shape[1] > 101:
        mel_db = mel_db[:, :101]

    return mel_db.astype(np.float32)


def normalize_flat(x):
    x = x.reshape(-1).astype(np.float32)

    norm = np.linalg.norm(x)

    if norm > 0:
        x = x / norm

    return x


def cosine_similarity(a, b):
    return float(
        np.dot(
            normalize_flat(a),
            normalize_flat(b)
        )
    )


def main():

    print("=" * 75)
    print("ECHOEDGE V8 - FIND SIMILAR TRAINING EXAMPLES")
    print("=" * 75)

    # ---------------------------------------------------------
    # Load training manifest
    # ---------------------------------------------------------

    df = pd.read_csv(TRAIN_MANIFEST)

    # Only genuine positive training windows
    positives = df[
        df["target"] == 1
    ].copy()

    print()
    print("Training positive windows:", len(positives))

    # ---------------------------------------------------------
    # Feature cache
    # ---------------------------------------------------------

    feature_file = (
        WINDOW_DATASET
        / "features"
        / "train_features.npz"
    )

    data = np.load(feature_file)

    X = data["X"]

    print("Training features:", X.shape)

    if len(X) != len(df):
        raise RuntimeError(
            f"Manifest/features mismatch: "
            f"{len(df)} vs {len(X)}"
        )

    # ---------------------------------------------------------
    # Compare each missed recording
    # ---------------------------------------------------------

    for filename in FILES:

        path = DATASET / filename

        print()
        print("=" * 75)
        print("TARGET:", filename)
        print("=" * 75)

        target = extract_feature(path)

        similarities = []

        for index in positives.index:

            train_feature = X[index]

            similarity = cosine_similarity(
                target,
                train_feature
            )

            similarities.append(
                (
                    similarity,
                    index
                )
            )

        similarities.sort(
            reverse=True
        )

        print()
        print("Most similar POSITIVE training examples:")
        print()

        for similarity, index in similarities[:10]:

            row = df.iloc[index]

            print(
                f"{similarity:.4f}  "
                f"{row['category']:<28} "
                f"{row['source_file']}"
            )

        print()
        print("Least similar positive training example:")
        print(
            f"{similarities[-1][0]:.4f}"
        )


if __name__ == "__main__":
    main()