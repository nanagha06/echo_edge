from pathlib import Path
import time
import numpy as np
import librosa


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8"
)

AUDIO_FILE = (
    DATASET_PATH
    / "wake_clean"
    / "wave30.wav"
)


# ============================================================
# V8 PARAMETERS
# ============================================================

SAMPLE_RATE = 16000

WINDOW_SAMPLES = 16000

N_FFT = 400
HOP_LENGTH = 160

N_MELS = 40

FMIN = 20
FMAX = 7600


# ============================================================
# LOAD AUDIO
# ============================================================

audio, sr = librosa.load(
    AUDIO_FILE,
    sr=SAMPLE_RATE,
    mono=True
)

# Pad to exactly one second
if len(audio) < WINDOW_SAMPLES:

    audio = np.pad(
        audio,
        (
            0,
            WINDOW_SAMPLES - len(audio)
        )
    )

audio = audio[:WINDOW_SAMPLES]


# ============================================================
# WARMUP
# ============================================================

for _ in range(3):

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

    feature = (
        log_mel - mean
    ) / (std + 1e-8)


# ============================================================
# BENCHMARK
# ============================================================

RUNS = 20

mel_times = []
log_times = []
norm_times = []
total_times = []


for _ in range(RUNS):

    t0 = time.perf_counter()

    # --------------------------------------------------------
    # MEL SPECTROGRAM
    # --------------------------------------------------------

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

    t1 = time.perf_counter()

    # --------------------------------------------------------
    # LOG MEL
    # --------------------------------------------------------

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    t2 = time.perf_counter()

    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

    mean = np.mean(log_mel)
    std = np.std(log_mel)

    feature = (
        log_mel - mean
    ) / (std + 1e-8)

    t3 = time.perf_counter()

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    mel_times.append(
        (t1 - t0) * 1000
    )

    log_times.append(
        (t2 - t1) * 1000
    )

    norm_times.append(
        (t3 - t2) * 1000
    )

    total_times.append(
        (t3 - t0) * 1000
    )


# ============================================================
# RESULTS
# ============================================================

print("=" * 70)
print("V8 FEATURE EXTRACTION PROFILE")
print("=" * 70)

print(
    f"\nAudio: {AUDIO_FILE}"
)

print(
    f"Sample rate: {SAMPLE_RATE}"
)

print(
    f"FFT: {N_FFT}"
)

print(
    f"FFT hop: {HOP_LENGTH}"
)

print(
    f"Mel bins: {N_MELS}"
)

print(
    f"Feature shape: {feature.shape}"
)


print("\nAverage timing over", RUNS, "runs:")

print(
    f"Mel spectrogram : "
    f"{np.mean(mel_times):.3f} ms"
)

print(
    f"Log-Mel         : "
    f"{np.mean(log_times):.3f} ms"
)

print(
    f"Normalization   : "
    f"{np.mean(norm_times):.3f} ms"
)

print(
    f"TOTAL           : "
    f"{np.mean(total_times):.3f} ms"
)


print("\nPercent of total:")

total = np.mean(total_times)

print(
    f"Mel spectrogram : "
    f"{100*np.mean(mel_times)/total:.1f}%"
)

print(
    f"Log-Mel         : "
    f"{100*np.mean(log_times)/total:.1f}%"
)

print(
    f"Normalization   : "
    f"{100*np.mean(norm_times)/total:.1f}%"
)

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)