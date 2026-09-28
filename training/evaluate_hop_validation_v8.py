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

MANIFEST_PATH = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8_windows"
) / "validation_v8.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "echoedge_v8_best.keras"

# ============================================================
# PARAMETERS
# ============================================================

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000

N_FFT = 400
HOP_LENGTH = 160
N_MELS = 40
FMIN = 20
FMAX = 7600

THRESHOLD = 0.46

HOPS = [
    0.500,
    0.250,
    0.125
]

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

    log_mel = (
        log_mel - mean
    ) / (std + 1e-8)

    if log_mel.shape[1] < 101:

        log_mel = np.pad(
            log_mel,
            (
                (0, 0),
                (0, 101 - log_mel.shape[1])
            ),
            mode="constant"
        )

    elif log_mel.shape[1] > 101:

        log_mel = log_mel[:, :101]

    return log_mel[..., np.newaxis].astype(
        np.float32
    )


# ============================================================
# LOAD MODEL + MANIFEST
# ============================================================

print("=" * 75)
print("V8 FULL VALIDATION WINDOW-HOP ANALYSIS")
print("=" * 75)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

manifest = pd.read_csv(
    MANIFEST_PATH
)

# Unique recordings
recordings = (
    manifest
    .groupby("source_file")
    .agg(
        target=("target", "max"),
        category=("category", "first")
    )
    .reset_index()
)

print(
    f"\nValidation recordings: "
    f"{len(recordings)}"
)

# ============================================================
# SCORE ONE RECORDING
# ============================================================

def score_recording(
    relative_path,
    hop_seconds
):

    path = DATASET_PATH / relative_path

    audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Same normalization used in previous analysis
    power = np.mean(audio ** 2)

    if power > 1e-12:

        audio = audio / np.sqrt(power)

    hop_samples = int(
        hop_seconds * SAMPLE_RATE
    )

    max_score = 0.0

    start = 0

    # IMPORTANT:
    # Only evaluate windows that contain
    # at least some real audio.
    while start < len(audio):

        end = start + WINDOW_SAMPLES

        window = audio[start:end]

        real_samples = len(window)

        if real_samples == 0:
            break

        if real_samples < WINDOW_SAMPLES:

            window = np.pad(
                window,
                (
                    0,
                    WINDOW_SAMPLES - real_samples
                ),
                mode="constant"
            )

        feature = extract_feature(window)

        score = float(
            model.predict(
                feature[np.newaxis, ...],
                verbose=0
            )[0][0]
        )

        max_score = max(
            max_score,
            score
        )

        start += hop_samples

    return max_score


# ============================================================
# EVALUATE EACH HOP
# ============================================================

for hop in HOPS:

    print("\n")
    print("=" * 75)
    print(
        f"HOP = {hop * 1000:.0f} ms"
    )
    print("=" * 75)

    results = []

    for _, row in recordings.iterrows():

        source_file = row["source_file"]

        target = int(row["target"])

        score = score_recording(
            source_file,
            hop
        )

        detected = int(
            score >= THRESHOLD
        )

        results.append(
            {
                "source_file": source_file,
                "category": row["category"],
                "target": target,
                "score": score,
                "detected": detected
            }
        )

    df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    tp = int(
        ((df.target == 1) &
         (df.detected == 1)).sum()
    )

    fn = int(
        ((df.target == 1) &
         (df.detected == 0)).sum()
    )

    fp = int(
        ((df.target == 0) &
         (df.detected == 1)).sum()
    )

    tn = int(
        ((df.target == 0) &
         (df.detected == 0)).sum()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn > 0
        else 0
    )

    print("\nOVERALL")

    print(f"TP        : {tp}")
    print(f"FN        : {fn}")
    print(f"FP        : {fp}")
    print(f"TN        : {tn}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"FPR       : {fpr:.4f}")

    # --------------------------------------------------------
    # Hard negatives only
    # --------------------------------------------------------

    hard_negative_categories = {
        "similar_words",
        "similar_phrases",
        "normal_commands",
        "background_speech"
    }

    hard = df[
        df["category"].isin(
            hard_negative_categories
        )
    ]

    hard_fp = int(
        (hard["detected"] == 1).sum()
    )

    hard_tn = int(
        (hard["detected"] == 0).sum()
    )

    hard_fpr = (
        hard_fp / len(hard)
        if len(hard) > 0
        else 0
    )

    print("\nHARD NEGATIVES ONLY")

    print(
        f"Recordings : {len(hard)}"
    )

    print(
        f"False positives : {hard_fp}"
    )

    print(
        f"FPR             : {hard_fpr:.4f}"
    )

    # --------------------------------------------------------
    # Positive categories
    # --------------------------------------------------------

    positives = df[df["target"] == 1]

    print("\nPOSITIVE CATEGORY RECALL")

    for category, group in (
        positives.groupby("category")
    ):

        detected_count = int(
            group["detected"].sum()
        )

        total_count = len(group)

        recall_cat = (
            detected_count / total_count
        )

        print(
            f"{category:25s} "
            f"{detected_count:2d}/"
            f"{total_count:2d} "
            f"({recall_cat:.3f})"
        )

print("\n")
print("=" * 75)
print("DONE")
print("=" * 75)

print(
    "\nTEST SET WAS NOT USED."
)