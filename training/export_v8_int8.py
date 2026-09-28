from pathlib import Path
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(
    r"C:\Users\nanag_ltzlj6d\esp\EchoEdge_WakeNet_Min_STREAMING"
)

MODEL_PATH = PROJECT_ROOT / "models" / "echoedge_v8_best.keras"

FEATURES_PATH = (
    Path(r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8_windows")
    / "features"
    / "train_features.npz"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "echoedge_v8_int8.tflite"
)


def representative_dataset():
    data = np.load(FEATURES_PATH)
    X = data["X"].astype(np.float32)

    print("Calibration data:", X.shape)

    # Deterministic calibration subset
    count = min(300, len(X))

    for i in range(count):
        yield [X[i:i + 1]]


def main():

    print("=" * 60)
    print("ECHOEDGE V8 -> FULL INT8 TFLITE")
    print("=" * 60)

    print("\nLoading model:")
    print(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Input shape:", model.input_shape)
    print("Parameters:", model.count_params())

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("\nConverting...")

    tflite_model = converter.convert()

    OUTPUT_PATH.write_bytes(tflite_model)

    print("\nSUCCESS")
    print("Output:", OUTPUT_PATH)
    print("Size:", len(tflite_model), "bytes")

    # ---------------------------------------------------------
    # Verify
    # ---------------------------------------------------------

    interpreter = tf.lite.Interpreter(
        model_content=tflite_model
    )

    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    print("\nINPUT")
    print("shape =", inp["shape"])
    print("dtype =", inp["dtype"])
    print("quant =", inp["quantization"])

    print("\nOUTPUT")
    print("shape =", out["shape"])
    print("dtype =", out["dtype"])
    print("quant =", out["quantization"])

    print("\n" + "=" * 60)
    print("V8 INT8 READY FOR ESP32")
    print("=" * 60)


if __name__ == "__main__":
    main()from pathlib import Path
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(
    r"C:\Users\nanag_ltzlj6d\esp\EchoEdge_WakeNet_Min_STREAMING"
)

MODEL_PATH = PROJECT_ROOT / "models" / "echoedge_v8_best.keras"

FEATURES_PATH = (
    Path(r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8_windows")
    / "features"
    / "train_features.npz"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "echoedge_v8_int8.tflite"
)


def representative_dataset():
    data = np.load(FEATURES_PATH)
    X = data["X"].astype(np.float32)

    print("Calibration data:", X.shape)

    # Deterministic calibration subset
    count = min(300, len(X))

    for i in range(count):
        yield [X[i:i + 1]]


def main():

    print("=" * 60)
    print("ECHOEDGE V8 -> FULL INT8 TFLITE")
    print("=" * 60)

    print("\nLoading model:")
    print(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Input shape:", model.input_shape)
    print("Parameters:", model.count_params())

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("\nConverting...")

    tflite_model = converter.convert()

    OUTPUT_PATH.write_bytes(tflite_model)

    print("\nSUCCESS")
    print("Output:", OUTPUT_PATH)
    print("Size:", len(tflite_model), "bytes")

    # ---------------------------------------------------------
    # Verify
    # ---------------------------------------------------------

    interpreter = tf.lite.Interpreter(
        model_content=tflite_model
    )

    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    print("\nINPUT")
    print("shape =", inp["shape"])
    print("dtype =", inp["dtype"])
    print("quant =", inp["quantization"])

    print("\nOUTPUT")
    print("shape =", out["shape"])
    print("dtype =", out["dtype"])
    print("quant =", out["quantization"])

    print("\n" + "=" * 60)
    print("V8 INT8 READY FOR ESP32")
    print("=" * 60)


if __name__ == "__main__":
    main()