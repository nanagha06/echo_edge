# ============================================================
# ECHOEDGE V8 - COMMON ML CONFIGURATION
# ============================================================

from pathlib import Path


# ------------------------------------------------------------
# PROJECT PATHS
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


# ------------------------------------------------------------
# AUDIO
# ------------------------------------------------------------

SAMPLE_RATE = 16000

WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = 16000

# Window-generation hop
WINDOW_HOP_SECONDS = 0.25
WINDOW_HOP_SAMPLES = 4000


# ------------------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------------------

N_FFT = 400

HOP_LENGTH = 160

N_MELS = 40

N_FRAMES = 101

FMIN = 20

FMAX = 7600


# ------------------------------------------------------------
# ANNOTATION
# ------------------------------------------------------------

# Small protection against slightly early/late
# manual timestamp boundaries.

ANNOTATION_MARGIN_SECONDS = 0.075


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

INPUT_SHAPE = (
    N_MELS,
    N_FRAMES,
    1
)

NUM_CLASSES = 1


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 0.001

RANDOM_SEED = 42


# ------------------------------------------------------------
# DETECTION
# ------------------------------------------------------------

# This is only the initial/reference threshold.
# We will determine the actual V8 threshold after evaluation.

INITIAL_WAKE_THRESHOLD = 0.60


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"

V8_MODEL_NAME = "echoedge_v8"


# ------------------------------------------------------------
# DATASET CATEGORIES
# ------------------------------------------------------------

POSITIVE_CATEGORIES = [
    "wake_clean",
    "wake_sentence_start",
    "wake_sentence_middle",
    "wake_sentence_end",
    "wake_sentence_natural",
    "wake_repeated",
    "wake_sentence_repeated",
    "wake_variation",
]

HARD_NEGATIVE_CATEGORIES = [
    "similar_words",
    "similar_phrases",
    "normal_commands",
    "background_speech",
]


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

def print_config():

    print()
    print("=" * 60)
    print("ECHOEDGE V8 CONFIGURATION")
    print("=" * 60)

    print("Sample rate       :", SAMPLE_RATE)
    print("Window            :", WINDOW_SECONDS, "sec")
    print("Window samples    :", WINDOW_SAMPLES)
    print("Window hop        :", WINDOW_HOP_SECONDS, "sec")

    print("FFT size           :", N_FFT)
    print("FFT hop            :", HOP_LENGTH)
    print("Mel bands          :", N_MELS)
    print("Frames             :", N_FRAMES)

    print("Feature shape      :", INPUT_SHAPE)

    print("Batch size         :", BATCH_SIZE)
    print("Epochs             :", EPOCHS)
    print("Learning rate      :", LEARNING_RATE)

    print("Annotation margin  :",
          ANNOTATION_MARGIN_SECONDS,
          "sec")

    print("=" * 60)