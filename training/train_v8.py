from pathlib import Path
import json
import numpy as np
import tensorflow as tf


# ============================================================
# EchoEdge V8 Training
# ============================================================
#
# TRAIN:
#   train_features.npz
#
# VALIDATION:
#   validation_features.npz
#
# TEST:
#   NOT USED HERE
#
# Goal:
#   Train a very small DS-CNN suitable for ESP32-S3/TFLite
#   Micro while using the new V8 dataset and sample weights.
#
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Users\nanag_ltzlj6d\esp\EchoEdge_WakeNet_Min_STREAMING"
)

FEATURE_PATH = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8_windows\features"
)

MODEL_PATH = PROJECT_ROOT / "models"

TRAIN_FILE = FEATURE_PATH / "train_features.npz"
VALIDATION_FILE = FEATURE_PATH / "validation_features.npz"


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

INPUT_SHAPE = (40, 101, 1)

BATCH_SIZE = 32
EPOCHS = 60

LEARNING_RATE = 0.001

RANDOM_SEED = 42

MODEL_NAME = "echoedge_v8"


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ------------------------------------------------------------
# Load feature dataset
# ------------------------------------------------------------

def load_feature_file(path):

    print()
    print("=" * 70)
    print(f"Loading: {path.name}")
    print("=" * 70)

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{path}"
        )

    data = np.load(
        path,
        allow_pickle=False,
    )

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    sample_weight = data["sample_weight"].astype(np.float32)

    print(f"X shape          : {X.shape}")
    print(f"y shape          : {y.shape}")
    print(f"sample_weight    : {sample_weight.shape}")

    print(
        f"Positive         : {np.sum(y == 1):.0f}"
    )

    print(
        f"Negative         : {np.sum(y == 0):.0f}"
    )

    print(
        f"Weight range     : "
        f"{sample_weight.min():.2f} - "
        f"{sample_weight.max():.2f}"
    )

    return X, y, sample_weight


# ------------------------------------------------------------
# Build Tiny DS-CNN
# ------------------------------------------------------------

def build_v8_model():

    inputs = tf.keras.Input(
        shape=INPUT_SHAPE,
        name="input_features",
    )

    # --------------------------------------------------------
    # Initial spatial convolution
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=True,
        name="conv_initial",
    )(inputs)

    x = tf.keras.layers.BatchNormalization(
        name="bn_initial",
    )(x)

    x = tf.keras.layers.ReLU(
        name="relu_initial",
    )(x)

    # --------------------------------------------------------
    # Depthwise-separable block 1
    # --------------------------------------------------------

    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        padding="same",
        use_bias=True,
        name="depthwise_1",
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="bn_depthwise_1",
    )(x)

    x = tf.keras.layers.ReLU(
        name="relu_depthwise_1",
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=(1, 1),
        padding="same",
        use_bias=True,
        name="pointwise_1",
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="bn_pointwise_1",
    )(x)

    x = tf.keras.layers.ReLU(
        name="relu_pointwise_1",
    )(x)

    # --------------------------------------------------------
    # Depthwise-separable block 2
    # --------------------------------------------------------

    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        padding="same",
        use_bias=True,
        name="depthwise_2",
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="bn_depthwise_2",
    )(x)

    x = tf.keras.layers.ReLU(
        name="relu_depthwise_2",
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(1, 1),
        padding="same",
        use_bias=True,
        name="pointwise_2",
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="bn_pointwise_2",
    )(x)

    x = tf.keras.layers.ReLU(
        name="relu_pointwise_2",
    )(x)

    # --------------------------------------------------------
    # Global average pooling
    # --------------------------------------------------------

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pool",
    )(x)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="wake_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=MODEL_NAME,
    )

    return model


# ------------------------------------------------------------
# Model summary
# ------------------------------------------------------------

def print_model_information(model):

    print()
    print("=" * 70)
    print("V8 MODEL")
    print("=" * 70)

    model.summary()

    trainable_params = (
        np.sum(
            [
                np.prod(variable.shape)
                for variable in model.trainable_variables
            ]
        )
    )

    non_trainable_params = (
        np.sum(
            [
                np.prod(variable.shape)
                for variable in model.non_trainable_variables
            ]
        )
    )

    total_params = (
        trainable_params +
        non_trainable_params
    )

    print()
    print(
        f"Trainable parameters    : "
        f"{trainable_params:,}"
    )

    print(
        f"Non-trainable parameters: "
        f"{non_trainable_params:,}"
    )

    print(
        f"Total parameters        : "
        f"{total_params:,}"
    )


# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

def create_callbacks():

    MODEL_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        MODEL_PATH /
        "echoedge_v8_best.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),

    ]

    return callbacks


# ------------------------------------------------------------
# Save training history
# ------------------------------------------------------------

def save_training_history(history):

    history_path = (
        MODEL_PATH /
        "echoedge_v8_training_history.json"
    )

    history_data = {
        key: [
            float(value)
            for value in values
        ]
        for key, values in history.history.items()
    }

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history_data,
            file,
            indent=2,
        )

    print()
    print(
        f"Training history saved to:\n"
        f"{history_path}"
    )


# ------------------------------------------------------------
# Main training
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("EchoEdge V8 TRAINING")
    print("=" * 70)

    print()
    print("Important:")
    print("  TEST SET WILL NOT BE USED.")
    print("  Only TRAIN + VALIDATION are used.")
    print()

    # --------------------------------------------------------
    # Load train
    # --------------------------------------------------------

    X_train, y_train, w_train = load_feature_file(
        TRAIN_FILE
    )

    # --------------------------------------------------------
    # Load validation
    # --------------------------------------------------------

    X_val, y_val, w_val = load_feature_file(
        VALIDATION_FILE
    )

    # --------------------------------------------------------
    # Verify shapes
    # --------------------------------------------------------

    if X_train.shape[1:] != INPUT_SHAPE:
        raise RuntimeError(
            f"Unexpected train shape: "
            f"{X_train.shape}"
        )

    if X_val.shape[1:] != INPUT_SHAPE:
        raise RuntimeError(
            f"Unexpected validation shape: "
            f"{X_val.shape}"
        )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_v8_model()

    print_model_information(model)

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
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
            ),
        ],
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = create_callbacks()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    history = model.fit(
        X_train,
        y_train,
        sample_weight=w_train,

        validation_data=(
            X_val,
            y_val,
            w_val,
        ),

        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,

        callbacks=callbacks,

        verbose=1,
    )

    # --------------------------------------------------------
    # Save final restored model
    # --------------------------------------------------------

    final_model_path = (
        MODEL_PATH /
        "echoedge_v8_float32.keras"
    )

    model.save(
        final_model_path
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Final model saved to:\n"
        f"{final_model_path}"
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    save_training_history(history)

    # --------------------------------------------------------
    # Best validation result
    # --------------------------------------------------------

    val_loss = history.history["val_loss"]
    val_accuracy = history.history["val_accuracy"]
    val_precision = history.history["val_precision"]
    val_recall = history.history["val_recall"]
    val_auc = history.history["val_auc"]

    best_epoch = int(
        np.argmin(val_loss)
    )

    print()
    print("=" * 70)
    print("BEST VALIDATION EPOCH")
    print("=" * 70)

    print(
        f"Epoch     : {best_epoch + 1}"
    )

    print(
        f"Val loss  : "
        f"{val_loss[best_epoch]:.6f}"
    )

    print(
        f"Val acc   : "
        f"{val_accuracy[best_epoch]:.6f}"
    )

    print(
        f"Val prec  : "
        f"{val_precision[best_epoch]:.6f}"
    )

    print(
        f"Val recall : "
        f"{val_recall[best_epoch]:.6f}"
    )

    print(
        f"Val AUC   : "
        f"{val_auc[best_epoch]:.6f}"
    )

    print()
    print("TEST SET WAS NOT USED.")
    print("Threshold optimization has NOT been performed.")
    print("TFLite export has NOT been performed.")
    print("=" * 70)


if __name__ == "__main__":
    main()