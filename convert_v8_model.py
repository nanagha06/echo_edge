from pathlib import Path

src = Path("models/echoedge_v8_int8.tflite")
dst = Path("main/echoedge_v8_model.cc")

data = src.read_bytes()

with dst.open("w", newline="\n") as f:
    f.write('#include "echoedge_v8_model.h"\n\n')
    f.write("const unsigned char g_echoedge_v8_model[] = {\n")

    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        f.write("    ")
        f.write(", ".join(f"0x{b:02x}" for b in chunk))
        f.write(",\n")

    f.write("};\n\n")
    f.write(f"const unsigned int g_echoedge_v8_model_len = {len(data)};\n")

print(f"Created: {dst}")
print(f"Model size: {len(data)} bytes")
