import os

# Keep TensorFlow output quiet
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\nanag_ltzlj6d\esp\EchoEdge_WakeNet_Min_STREAMING"
)

WINDOW_DATASET = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8_windows"
)

MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_FEATURES = (
    WINDOW_DATASET
    / "features"
    / "train_features.npz"
)

VALIDATION_FEATURES = (
    WINDOW_DATASET
    / "features"
    / "validation_features.npz"
)

TRAIN_MANIFEST = (
    WINDOW_DATASET
    / "train_v8.csv"
)

VALIDATION_MANIFEST = (
    WINDOW_DATASET
    / "validation_v8.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 32
EPOCHS = 70

INITIAL_LR = 0.001
MIN_LR = 0.00005

OUTPUT_MODEL = (
    MODEL_DIR
    / "echoedge_v8_1_best.keras"
)

OUTPUT_HISTORY = (
    MODEL_DIR
    / "echoedge_v8_1_training_history.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("ECHOEDGE V8.1 - AUGMENTED TRAINING")
print("=" * 80)

print("\nLoading training data...")

train_data = np.load(TRAIN_FEATURES)

X_train = train_data["X"].astype(
    np.float32
)

y_train = train_data["y"].astype(
    np.float32
)

w_train = train_data["sample_weight"].astype(
    np.float32
)

print(
    f"Training X: {X_train.shape}"
)

print(
    f"Training positives: "
    f"{int(np.sum(y_train == 1))}"
)

print(
    f"Training negatives: "
    f"{int(np.sum(y_train == 0))}"
)

print("\nLoading validation data...")

val_data = np.load(VALIDATION_FEATURES)

X_val = val_data["X"].astype(
    np.float32
)

y_val = val_data["y"].astype(
    np.float32
)

w_val = val_data["sample_weight"].astype(
    np.float32
)

print(
    f"Validation X: {X_val.shape}"
)

print(
    f"Validation positives: "
    f"{int(np.sum(y_val == 1))}"
)

print(
    f"Validation negatives: "
    f"{int(np.sum(y_val == 0))}"
)


# ============================================================
# SANITY CHECK
# ============================================================

if X_train.ndim != 4:
    raise RuntimeError(
        f"Unexpected training shape: {X_train.shape}"
    )

if X_val.ndim != 4:
    raise RuntimeError(
        f"Unexpected validation shape: {X_val.shape}"
    )

if X_train.shape[1:] != (40, 101, 1):
    raise RuntimeError(
        f"Unexpected input shape: {X_train.shape[1:]}"
    )

if X_val.shape[1:] != (40, 101, 1):
    raise RuntimeError(
        f"Unexpected validation shape: {X_val.shape[1:]}"
    )


# ============================================================
# V8.1 AUGMENTATION
#
# IMPORTANT:
# Augmentation is applied ONLY to training data.
#
# Validation data is NEVER augmented.
#
# We operate on the existing Log-Mel features.
# ============================================================

def augment_feature(x, rng):

    x = x.copy()

    # --------------------------------------------------------
    # 1. Small feature gain variation
    # --------------------------------------------------------

    if rng.random() < 0.50:

        gain = rng.uniform(
            0.90,
            1.10
        )

        x *= gain

    # --------------------------------------------------------
    # 2. Small additive feature noise
    # --------------------------------------------------------

    if rng.random() < 0.35:

        noise_std = rng.uniform(
            0.005,
            0.02
        )

        noise = rng.normal(
            0.0,
            noise_std,
            size=x.shape
        ).astype(
            np.float32
        )

        x += noise

    # --------------------------------------------------------
    # 3. Frequency masking
    #
    # Mask a small number of Mel bins.
    # --------------------------------------------------------

    if rng.random() < 0.25:

        width = int(
            rng.integers(
                1,
                4
            )
        )

        start = int(
            rng.integers(
                0,
                max(1, 40 - width)
            )
        )

        x[
            start:start + width,
            :,
            :
        ] = 0.0

    # --------------------------------------------------------
    # 4. Time masking
    #
    # Small temporal masking.
    # --------------------------------------------------------

    if rng.random() < 0.25:

        width = int(
            rng.integers(
                1,
                6
            )
        )

        start = int(
            rng.integers(
                0,
                max(1, 101 - width)
            )
        )

        x[
            :,
            start:start + width,
            :
        ] = 0.0

    return x.astype(
        np.float32
    )


class AugmentedSequence(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        X,
        y,
        weights,
        batch_size,
        seed
    ):

        self.X = X
        self.y = y
        self.weights = weights

        self.batch_size = batch_size

        self.rng = np.random.default_rng(
            seed
        )

        self.indices = np.arange(
            len(X)
        )

        self.on_epoch_end()

    def __len__(self):

        return int(
            np.ceil(
                len(self.indices)
                / self.batch_size
            )
        )

    def __getitem__(self, index):

        start = (
            index
            * self.batch_size
        )

        end = min(
            start + self.batch_size,
            len(self.indices)
        )

        batch_indices = (
            self.indices[start:end]
        )

        batch_X = self.X[
            batch_indices
        ].copy()

        batch_y = self.y[
            batch_indices
        ].copy()

        batch_w = self.weights[
            batch_indices
        ].copy()

        # ----------------------------------------------------
        # Apply augmentation primarily to positives.
        #
        # Hard negatives remain unchanged.
        # ----------------------------------------------------

        for i in range(
            len(batch_X)
        ):

            if batch_y[i] == 1:

                # Augment approximately 70%
                # of positive examples.

                if self.rng.random() < 0.70:

                    batch_X[i] = augment_feature(
                        batch_X[i],
                        self.rng
                    )

        return (
            batch_X,
            batch_y,
            batch_w
        )

    def on_epoch_end(self):

        self.rng.shuffle(
            self.indices
        )


# ============================================================
# MODEL
# ============================================================

def build_model():

    inputs = tf.keras.Input(
        shape=(40, 101, 1),
        name="input"
    )

    x = tf.keras.layers.Conv2D(
        8,
        (3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.DepthwiseConv2D(
        (3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        8,
        (1, 1),
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.DepthwiseConv2D(
        (3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        16,
        (1, 1),
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    return tf.keras.Model(
        inputs,
        outputs,
        name="echoedge_v8_1"
    )


model = build_model()


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LR
    ),

    loss=tf.keras.losses.BinaryCrossentropy(),

    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)


print("\nModel:")
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=OUTPUT_MODEL,

        monitor="val_auc",

        mode="max",

        save_best_only=True,

        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=6,

        min_lr=MIN_LR,

        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_auc",

        mode="max",

        patience=12,

        restore_best_weights=True,

        verbose=1
    )
]


# ============================================================
# TRAINING SEQUENCE
# ============================================================

train_sequence = AugmentedSequence(

    X_train,

    y_train,

    w_train,

    BATCH_SIZE,

    SEED
)


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 80)
print("STARTING V8.1 TRAINING")
print("=" * 80)

history = model.fit(

    train_sequence,

    validation_data=(
        X_val,
        y_val,
        w_val
    ),

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if OUTPUT_MODEL.exists():

    print("\nLoading best V8.1 model...")

    model = tf.keras.models.load_model(
        OUTPUT_MODEL,
        compile=False
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=MIN_LR
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n")
print("=" * 80)
print("FINAL V8.1 VALIDATION")
print("=" * 80)

results = model.evaluate(
    X_val,
    y_val,
    sample_weight=w_val,
    verbose=0,
    return_dict=True
)

for key, value in results.items():

    print(
        f"{key:<15}: {value:.6f}"
    )


# ============================================================
# SAVE HISTORY
# ============================================================

history_data = {
    key: [
        float(v)
        for v in values
    ]
    for key, values
    in history.history.items()
}

with open(
    OUTPUT_HISTORY,
    "w"
) as f:

    json.dump(
        history_data,
        f,
        indent=2
    )


# ============================================================
# FINAL INFO
# ============================================================

print("\n")
print("=" * 80)
print("V8.1 TRAINING COMPLETE")
print("=" * 80)

print("\nBest model:")
print(OUTPUT_MODEL)

print("\nTraining history:")
print(OUTPUT_HISTORY)

print("\nIMPORTANT:")
print("TEST SET WAS NOT USED.")

print("\nDataset was NOT modified.")
print("No recordings were added.")
print("No annotations were changed.")