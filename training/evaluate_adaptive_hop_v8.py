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

MANIFEST_PATH = (
    Path(r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8_windows")
    / "validation_v8.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "echoedge_v8_best.keras"
)

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

FINAL_THRESHOLD = 0.46

BASE_HOP = 0.500
FINE_HOP = 0.250

# Test several suspicion thresholds.
SUSPICION_THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40
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
# MODEL
# ============================================================

print("=" * 80)
print("ECHOEDGE V8 ADAPTIVE-HOP VALIDATION")
print("=" * 80)

print("\nLoading V8...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("V8 loaded.")

# ============================================================
# VALIDATION RECORDINGS
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH
)

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
    f"\nValidation recordings: {len(recordings)}"
)

# ============================================================
# SCORE WINDOW
# ============================================================

def score_window(audio, start_sample):

    end_sample = start_sample + WINDOW_SAMPLES

    window = audio[
        start_sample:min(
            end_sample,
            len(audio)
        )
    ]

    real_samples = len(window)

    if real_samples == 0:
        return None

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

    return score


# ============================================================
# LOAD AUDIO
# ============================================================

audio_cache = {}

def load_audio(relative_path):

    if relative_path not in audio_cache:

        path = DATASET_PATH / relative_path

        audio, sr = librosa.load(
            path,
            sr=SAMPLE_RATE,
            mono=True
        )

        power = np.mean(audio ** 2)

        if power > 1e-12:
            audio = audio / np.sqrt(power)

        audio_cache[relative_path] = audio

    return audio_cache[relative_path]


# ============================================================
# ADAPTIVE SCORING
# ============================================================

def adaptive_score(
    relative_path,
    suspicion_threshold
):

    audio = load_audio(relative_path)

    base_hop_samples = int(
        BASE_HOP * SAMPLE_RATE
    )

    fine_hop_samples = int(
        FINE_HOP * SAMPLE_RATE
    )

    max_score = 0.0

    feature_evaluations = 0
    base_evaluations = 0
    fine_evaluations = 0

    fine_activations = 0

    # Prevent repeatedly evaluating the same
    # fine windows.
    evaluated = set()

    start = 0

    while start < len(audio):

        # ----------------------------------------------------
        # NORMAL 500-ms evaluation
        # ----------------------------------------------------

        if start not in evaluated:

            score = score_window(
                audio,
                start
            )

            evaluated.add(start)

            feature_evaluations += 1
            base_evaluations += 1

            if score is None:
                break

            max_score = max(
                max_score,
                score
            )

        else:

            score = None

        # ----------------------------------------------------
        # SUSPICIOUS -> FINE SEARCH
        # ----------------------------------------------------

        if (
            score is not None
            and score >= suspicion_threshold
        ):

            fine_activations += 1

            # Search neighboring 250-ms positions.
            #
            # We inspect:
            # start - 250 ms
            # start + 250 ms
            #
            # The current 500-ms window has already
            # been evaluated.

            candidate_starts = [
                start - fine_hop_samples,
                start + fine_hop_samples
            ]

            for fine_start in candidate_starts:

                if fine_start < 0:
                    continue

                if fine_start >= len(audio):
                    continue

                if fine_start in evaluated:
                    continue

                fine_score = score_window(
                    audio,
                    fine_start
                )

                evaluated.add(
                    fine_start
                )

                feature_evaluations += 1
                fine_evaluations += 1

                if fine_score is None:
                    continue

                max_score = max(
                    max_score,
                    fine_score
                )

        start += base_hop_samples

    detected = int(
        max_score >= FINAL_THRESHOLD
    )

    return {
        "score": max_score,
        "detected": detected,
        "feature_evaluations": feature_evaluations,
        "base_evaluations": base_evaluations,
        "fine_evaluations": fine_evaluations,
        "fine_activations": fine_activations
    }


# ============================================================
# METRICS
# ============================================================

HARD_NEGATIVE_CATEGORIES = {
    "similar_words",
    "similar_phrases",
    "normal_commands",
    "background_speech"
}


def calculate_metrics(df):

    tp = int(
        (
            (df["target"] == 1)
            & (df["detected"] == 1)
        ).sum()
    )

    fn = int(
        (
            (df["target"] == 1)
            & (df["detected"] == 0)
        ).sum()
    )

    fp = int(
        (
            (df["target"] == 0)
            & (df["detected"] == 1)
        ).sum()
    )

    tn = int(
        (
            (df["target"] == 0)
            & (df["detected"] == 0)
        ).sum()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn > 0
        else 0.0
    )

    hard = df[
        df["category"].isin(
            HARD_NEGATIVE_CATEGORIES
        )
    ]

    hard_fp = int(
        (hard["detected"] == 1).sum()
    )

    hard_fpr = (
        hard_fp / len(hard)
        if len(hard) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "hard_fp": hard_fp,
        "hard_fpr": hard_fpr
    }


# ============================================================
# RUN EXPERIMENT
# ============================================================

all_summary = []

for suspicion in SUSPICION_THRESHOLDS:

    print("\n")
    print("=" * 80)
    print(
        f"SUSPICION THRESHOLD = {suspicion:.2f}"
    )
    print(
        f"FINAL DETECTION THRESHOLD = "
        f"{FINAL_THRESHOLD:.2f}"
    )
    print("=" * 80)

    results = []

    total_features = 0
    total_base = 0
    total_fine = 0
    total_fine_activations = 0

    for _, row in recordings.iterrows():

        result = adaptive_score(
            row["source_file"],
            suspicion
        )

        results.append(
            {
                "source_file": row["source_file"],
                "category": row["category"],
                "target": int(row["target"]),
                **result
            }
        )

        total_features += (
            result["feature_evaluations"]
        )

        total_base += (
            result["base_evaluations"]
        )

        total_fine += (
            result["fine_evaluations"]
        )

        total_fine_activations += (
            result["fine_activations"]
        )

    df = pd.DataFrame(results)

    metrics = calculate_metrics(df)

    avg_features = (
        total_features / len(df)
    )

    avg_base = (
        total_base / len(df)
    )

    avg_fine = (
        total_fine / len(df)
    )

    avg_fine_activations = (
        total_fine_activations / len(df)
    )

    print("\nDETECTION")

    print(
        f"TP        : {metrics['tp']}"
    )

    print(
        f"FN        : {metrics['fn']}"
    )

    print(
        f"FP        : {metrics['fp']}"
    )

    print(
        f"TN        : {metrics['tn']}"
    )

    print(
        f"Precision : {metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"FPR       : {metrics['fpr']:.4f}"
    )

    print("\nHARD NEGATIVES")

    print(
        f"False positives : "
        f"{metrics['hard_fp']}"
    )

    print(
        f"Hard-negative FPR : "
        f"{metrics['hard_fpr']:.4f}"
    )

    print("\nCOMPUTATION")

    print(
        f"Total feature evaluations : "
        f"{total_features}"
    )

    print(
        f"Average feature evaluations/"
        f"recording : {avg_features:.2f}"
    )

    print(
        f"Average base evaluations : "
        f"{avg_base:.2f}"
    )

    print(
        f"Average fine evaluations : "
        f"{avg_fine:.2f}"
    )

    print(
        f"Average fine-search activations : "
        f"{avg_fine_activations:.2f}"
    )

    # Approximate feature extraction time.
    #
    # Current measured feature extraction ≈ 550 ms.
    # This is only a rough computational estimate,
    # NOT actual ESP32 timing.

    estimated_feature_ms = (
        avg_features * 550.0
    )

    print(
        f"Estimated feature-processing time/"
        f"recording : "
        f"{estimated_feature_ms:.0f} ms"
    )

    all_summary.append(
        {
            "suspicion": suspicion,
            **metrics,
            "avg_features": avg_features,
            "avg_base": avg_base,
            "avg_fine": avg_fine,
            "avg_fine_activations":
                avg_fine_activations,
            "estimated_feature_ms":
                estimated_feature_ms
        }
    )

# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    all_summary
)

print("\n")
print("=" * 80)
print("ADAPTIVE-HOP SUMMARY")
print("=" * 80)

print(
    summary[
        [
            "suspicion",
            "tp",
            "fn",
            "fp",
            "tn",
            "precision",
            "recall",
            "hard_fpr",
            "avg_features",
            "avg_fine_activations",
            "estimated_feature_ms"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# Save results
output_path = (
    PROJECT_ROOT
    / "models"
    / "echoedge_v8_adaptive_hop_results.csv"
)

summary.to_csv(
    output_path,
    index=False
)

print("\nSaved:")
print(output_path)

print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    "\nTEST SET WAS NOT USED."
)