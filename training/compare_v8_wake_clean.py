from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


DATASET = Path(r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8")

FILES = [
    "wake_clean/wave20.wav",
    "wake_clean/wave33.wav",
    "wake_clean/wave34.wav",
]


def main():

    fig, axes = plt.subplots(
        3, 2,
        figsize=(14, 12)
    )

    for row, filename in enumerate(FILES):

        path = DATASET / filename

        audio, sr = librosa.load(
            path,
            sr=16000,
            mono=True
        )

        # -----------------------------
        # Waveform
        # -----------------------------
        time = np.arange(len(audio)) / sr

        axes[row, 0].plot(time, audio)

        axes[row, 0].set_title(
            f"Waveform - {filename}"
        )

        axes[row, 0].set_xlabel("Time (seconds)")
        axes[row, 0].set_ylabel("Amplitude")
        axes[row, 0].grid(True)

        # -----------------------------
        # Mel spectrogram
        # -----------------------------
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=400,
            hop_length=160,
            n_mels=40,
            fmin=20,
            fmax=7600,
            power=2
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        img = librosa.display.specshow(
            mel_db,
            sr=sr,
            hop_length=160,
            x_axis="time",
            y_axis="mel",
            fmin=20,
            fmax=7600,
            ax=axes[row, 1]
        )

        axes[row, 1].set_title(
            f"Mel Spectrogram - {filename}"
        )

        fig.colorbar(
            img,
            ax=axes[row, 1],
            format="%+2.0f dB"
        )

    plt.tight_layout()

    output = DATASET / "v8_wake_clean_comparison.png"

    plt.savefig(
        output,
        dpi=150
    )

    print("=" * 70)
    print("Comparison saved to:")
    print(output)
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()