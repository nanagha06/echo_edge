from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\nanag_ltzlj6d\esp\EchoEdge_WakeNet_Min_STREAMING"
)

DATASET_PATH = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8"
)

MODEL_PATH = PROJECT_ROOT / "models" / "echoedge_v8_best.keras"

MISSED_FILES = [
    "wake_clean/wave20.wav",
    "wake_clean/wave33.wav",
    "wake_sentence_middle/WhatsApp Audio 2026-09-22 at 9.22.21 PM (2).wav",
    "wake_sentence_natural/WhatsApp Audio 2026-09-22 at 9.59.59 PM.wav",
    "wake_sentence_middle/WhatsApp Audio 2026-09-22 at 9.22.20 PM.wav",
]

# ============================================================
# FEATURE PARAMETERS — SAME AS V8
# ============================================================

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000

N_FFT = 400
HOP_LENGTH = 160
N_MELS = 40
FMIN = 20
FMAX = 7600

INPUT_SCALE = 0.035467877984046936
INPUT_ZERO_POINT = -26

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_feature(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
        center=True
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mean = np.mean(log_mel)
    std = np.std(log_mel)

    log_mel = (log_mel - mean) / (std + 1e-8)

    # Ensure exactly 101 frames
    if log_mel.shape[1] < 101:
        pad = 101 - log_mel.shape[1]
        log_mel = np.pad(
            log_mel,
            ((0, 0), (0, pad)),
            mode="constant"
        )

    elif log_mel.shape[1] > 101:
        log_mel = log_mel[:, :101]

    return log_mel[..., np.newaxis].astype(np.float32)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 75)
print("V8 WINDOW-HOP ANALYSIS")
print("=" * 75)

print("\nLoading V8 model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded.")


# ============================================================
# TEST ONE FILE
# ============================================================

def analyze_file(relative_path, hop_seconds):

    path = DATASET_PATH / relative_path

    audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Normalize exactly as deployment preprocessing expects
    power = np.mean(audio ** 2)

    if power > 1e-12:
        audio = audio / np.sqrt(power)

    hop_samples = int(hop_seconds * SAMPLE_RATE)

    scores = []

    start = 0

    while start < len(audio):

        end = start + WINDOW_SAMPLES

        window = audio[start:end]

        # Pad short final window
        if len(window) < WINDOW_SAMPLES:

            window = np.pad(
                window,
                (0, WINDOW_SAMPLES - len(window)),
                mode="constant"
            )

        feature = extract_feature(window)

        prediction = model.predict(
            feature[np.newaxis, ...],
            verbose=0
        )

        score = float(prediction[0][0])

        scores.append(
            (
                score,
                start / SAMPLE_RATE,
                min(end, len(audio)) / SAMPLE_RATE
            )
        )

        start += hop_samples

        # Don't create excessive windows after the end
        if start >= len(audio):
            break

    scores.sort(reverse=True, key=lambda x: x[0])

    return scores


# ============================================================
# RUN ANALYSIS
# ============================================================

HOPS = [
    0.500,
    0.250,
    0.125,
]

for hop in HOPS:

    print("\n")
    print("=" * 75)
    print(f"WINDOW HOP = {hop * 1000:.0f} ms")
    print("=" * 75)

    for file in MISSED_FILES:

        scores = analyze_file(
            file,
            hop
        )

        best = scores[0]

        print("\n" + file)

        print(
            f"  MAX SCORE : {best[0]:.6f}"
        )

        print(
            f"  BEST WIN  : "
            f"{best[1]:.3f}s - {best[2]:.3f}s"
        )

        print("  TOP 3:")

        for score, start, end in scores[:3]:

            print(
                f"      {score:.6f} "
                f"({start:.3f}s - {end:.3f}s)"
            )


print("\n")
print("=" * 75)
print("DONE")
print("=" * 75)