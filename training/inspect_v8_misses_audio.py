import os
from pathlib import Path
import numpy as np
import soundfile as sf

DATASET = Path(r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8")

FILES = [
    "wake_clean/wave20.wav",
    "wake_clean/wave33.wav",
    "wake_sentence_middle/WhatsApp Audio 2026-09-22 at 9.22.21 PM (2).wav",
    "wake_sentence_natural/WhatsApp Audio 2026-09-22 at 9.59.59 PM.wav",
    "wake_sentence_middle/WhatsApp Audio 2026-09-22 at 9.22.20 PM.wav",
]

def inspect_audio(path):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))

    if peak > 0:
        dbfs_peak = 20 * np.log10(peak)
    else:
        dbfs_peak = -np.inf

    if rms > 0:
        dbfs_rms = 20 * np.log10(rms)
    else:
        dbfs_rms = -np.inf

    print("=" * 70)
    print("FILE:", path.relative_to(DATASET))
    print("Sample rate :", sr)
    print("Samples     :", len(audio))
    print("Duration    :", round(len(audio) / sr, 3), "sec")
    print("Peak        :", round(peak, 6))
    print("Peak dBFS   :", round(dbfs_peak, 2))
    print("RMS         :", round(rms, 6))
    print("RMS dBFS    :", round(dbfs_rms, 2))
    print("Mean        :", round(float(np.mean(audio)), 6))
    print("Std         :", round(float(np.std(audio)), 6))


def main():
    print("=" * 70)
    print("ECHOEDGE V8 - AUDIO INSPECTION OF MISSED RECORDINGS")
    print("=" * 70)

    for relative_path in FILES:
        path = DATASET / relative_path

        if not path.exists():
            print("\nMISSING:", path)
            continue

        inspect_audio(path)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()