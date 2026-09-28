import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf


DATASET = Path(r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8")

MODEL_PATH = Path(
    r"C:\Users\nanag_ltzlj6d\esp\EchoEdge_WakeNet_Min_STREAMING\models\echoedge_v8_best.keras"
)

FILES = [
    "wake_clean/wave20.wav",
    "wake_clean/wave30.wav",
    "wake_clean/wave33.wav",
]


def extract_feature(path):

    audio, sr = librosa.load(
        path,
        sr=16000,
        mono=True
    )

    # One-second window
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

    # V8 preprocessing
    mel_db = (
        mel_db - np.mean(mel_db)
    ) / (
        np.std(mel_db) + 1e-8
    )

    # Ensure 101 frames
    if mel_db.shape[1] < 101:

        mel_db = np.pad(
            mel_db,
            (
                (0, 0),
                (0, 101 - mel_db.shape[1])
            )
        )

    elif mel_db.shape[1] > 101:

        mel_db = mel_db[:, :101]

    return mel_db.astype(np.float32)


def main():

    print("=" * 70)
    print("ECHOEDGE V8 - DIRECT MODEL COMPARISON")
    print("=" * 70)

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded.")

    print()

    for filename in FILES:

        path = DATASET / filename

        feature = extract_feature(path)

        x = feature[np.newaxis, :, :, np.newaxis]

        probability = float(
            model.predict(
                x,
                verbose=0
            )[0][0]
        )

        print("-" * 70)
        print(filename)
        print(f"Probability : {probability:.6f}")

        if probability >= 0.46:
            print("Result      : DETECT")
        else:
            print("Result      : MISS")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()