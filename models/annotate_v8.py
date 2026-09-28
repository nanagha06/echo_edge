import csv
import subprocess
import tempfile
import wave
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# ECHOEDGE V8 - MANUAL WAKE-WORD ANNOTATION TOOL
# ============================================================
#
# Supports:
#   - Existing annotations
#   - Multiple ECHOEDGE occurrences per recording
#   - 7 positive categories
#   - Original WAV files remain untouched
#
# CSV format:
#
# category,file,start_time,end_time
#
# One row = one ECHOEDGE occurrence.
# ============================================================


# ============================================================
# PATHS
# ============================================================

DATASET = Path(
    r"C:\Users\nanag_ltzlj6d\Desktop\dataset_v8"
)

ANNOTATION_FILE = DATASET / "annotations.csv"


# ============================================================
# POSITIVE CATEGORIES THAT REQUIRE LOCALIZATION
# ============================================================

TARGET_FOLDERS = [
    "wake_clean",
    "wake_sentence_start",
    "wake_sentence_middle",
    "wake_sentence_end",
    "wake_sentence_natural",
    "wake_repeated",
    "wake_sentence_repeated",
    "wake_variation",
]


EXPECTED_SAMPLE_RATE = 16000


# ============================================================
# LOAD EXISTING ANNOTATIONS
# ============================================================

# Dictionary:
#
# (category, relative_file)
#       ->
# list of annotations
#
# Example:
#
# {
#   ("wake_repeated", "wake_repeated/file.wav"):
#       [
#           {"start": 0.5, "end": 1.2},
#           {"start": 1.8, "end": 2.5}
#       ]
# }

annotations = {}


def load_existing_annotations():

    if not ANNOTATION_FILE.exists():
        return

    with open(
        ANNOTATION_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:
            return

        required = {
            "category",
            "file",
            "start_time",
            "end_time"
        }

        if not required.issubset(
            reader.fieldnames
        ):

            raise ValueError(
                "\nUnexpected annotations.csv format.\n"
                f"Found columns: {reader.fieldnames}\n"
                "Expected:\n"
                "category,file,start_time,end_time"
            )

        for row in reader:

            category = row["category"].strip()

            relative_file = row["file"].strip()

            relative_file = relative_file.replace(
                "\\",
                "/"
            )

            try:

                start = float(
                    row["start_time"]
                )

                end = float(
                    row["end_time"]
                )

            except ValueError:

                continue

            key = (
                category,
                relative_file
            )

            if key not in annotations:

                annotations[key] = []

            annotations[key].append({
                "start": start,
                "end": end
            })


load_existing_annotations()


# ============================================================
# FIND WAV FILES
# ============================================================

audio_files = []


for category in TARGET_FOLDERS:

    folder = DATASET / category

    if not folder.exists():

        print(
            f"WARNING: Folder not found: {folder}"
        )

        continue

    for wav_file in sorted(
        folder.rglob("*.wav")
    ):

        audio_files.append(
            (
                category,
                wav_file
            )
        )


if not audio_files:

    raise SystemExit(
        "ERROR: No WAV files found."
    )


# ============================================================
# GLOBAL STATE
# ============================================================

current_index = 0

samples = None
duration = 0

# Current unsaved marker
current_start = None
current_end = None


# ============================================================
# HELPERS
# ============================================================

def current_key():

    category, wav_file = (
        audio_files[current_index]
    )

    relative_file = str(
        wav_file.relative_to(DATASET)
    ).replace(
        "\\",
        "/"
    )

    return (
        category,
        relative_file
    )


def get_saved_annotations():

    key = current_key()

    return annotations.get(
        key,
        []
    )


def count_total_annotations():

    return sum(
        len(items)
        for items in annotations.values()
    )


# ============================================================
# SAVE ALL ANNOTATIONS
# ============================================================

def save_annotations():

    with open(
        ANNOTATION_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "category",
            "file",
            "start_time",
            "end_time"
        ])

        # Preserve the dataset/file order
        # instead of dictionary order.

        for category, wav_file in audio_files:

            relative_file = str(
                wav_file.relative_to(DATASET)
            ).replace(
                "\\",
                "/"
            )

            key = (
                category,
                relative_file
            )

            if key not in annotations:
                continue

            # Sort occurrences by start time

            occurrences = sorted(
                annotations[key],
                key=lambda x: x["start"]
            )

            for data in occurrences:

                writer.writerow([
                    category,
                    relative_file,
                    f"{data['start']:.4f}",
                    f"{data['end']:.4f}"
                ])


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio():

    global samples
    global duration
    global current_start
    global current_end

    category, wav_file = (
        audio_files[current_index]
    )

    with wave.open(
        str(wav_file),
        "rb"
    ) as wf:

        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()

        raw = wf.readframes(
            frame_count
        )

    if sample_rate != EXPECTED_SAMPLE_RATE:

        raise ValueError(
            f"{wav_file.name}: expected "
            f"16000 Hz, got {sample_rate} Hz"
        )

    if channels != 1:

        raise ValueError(
            f"{wav_file.name}: expected mono audio."
        )

    if sample_width != 2:

        raise ValueError(
            f"{wav_file.name}: expected 16-bit PCM."
        )

    samples = np.frombuffer(
        raw,
        dtype=np.int16
    ).astype(np.float32)

    samples /= 32768.0

    duration = (
        len(samples) /
        EXPECTED_SAMPLE_RATE
    )

    # Reset unsaved marker

    current_start = None
    current_end = None

    update_plot()
    update_labels()


# ============================================================
# UPDATE WAVEFORM
# ============================================================

def update_plot():

    ax.clear()

    if samples is None:
        return

    time_axis = (
        np.arange(len(samples)) /
        EXPECTED_SAMPLE_RATE
    )

    ax.plot(
        time_axis,
        samples,
        linewidth=0.7
    )

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Amplitude"
    )

    category, wav_file = (
        audio_files[current_index]
    )

    ax.set_title(
        f"{category} / {wav_file.name}"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    # --------------------------------------------------------
    # Already saved annotations
    # --------------------------------------------------------

    saved = get_saved_annotations()

    for i, item in enumerate(
        saved,
        start=1
    ):

        start = item["start"]
        end = item["end"]

        ax.axvline(
            start,
            linewidth=1.5
        )

        ax.axvline(
            end,
            linewidth=1.5
        )

        ax.axvspan(
            start,
            end,
            alpha=0.15
        )

        # Label occurrence number

        midpoint = (
            start + end
        ) / 2

        y_min, y_max = ax.get_ylim()

        ax.text(
            midpoint,
            y_max * 0.85,
            f"#{i}",
            ha="center",
            fontsize=9
        )

    # --------------------------------------------------------
    # Current unsaved annotation
    # --------------------------------------------------------

    if current_start is not None:

        ax.axvline(
            current_start,
            linewidth=2
        )

    if current_end is not None:

        ax.axvline(
            current_end,
            linewidth=2
        )

    if (
        current_start is not None
        and
        current_end is not None
    ):

        ax.axvspan(
            current_start,
            current_end,
            alpha=0.25
        )

    canvas.draw()


# ============================================================
# UPDATE TEXT
# ============================================================

def update_labels():

    category, wav_file = (
        audio_files[current_index]
    )

    position_label.config(
        text=(
            f"{current_index + 1} / "
            f"{len(audio_files)}"
        )
    )

    file_label.config(
        text=f"{category}  |  {wav_file.name}"
    )

    saved = get_saved_annotations()

    if saved:

        parts = []

        for i, item in enumerate(
            saved,
            start=1
        ):

            parts.append(
                f"#{i}: "
                f"{item['start']:.3f}s → "
                f"{item['end']:.3f}s"
            )

        timestamp_label.config(
            text=(
                f"Saved ECHOEDGE "
                f"({len(saved)} occurrence"
                f"{'s' if len(saved) != 1 else ''}): "
                + " | ".join(parts)
            )
        )

    else:

        timestamp_label.config(
            text="ECHOEDGE: Not marked"
        )

    # Current unsaved marker

    if (
        current_start is not None
        and
        current_end is not None
    ):

        status_label.config(
            text=(
                f"Current: "
                f"{current_start:.3f}s → "
                f"{current_end:.3f}s"
            )
        )

    elif (
        current_start is not None
    ):

        status_label.config(
            text=(
                f"Current start: "
                f"{current_start:.3f}s"
            )
        )


# ============================================================
# CLICK WAVEFORM
# ============================================================

def on_click(event):

    global current_start
    global current_end

    if event.inaxes != ax:
        return

    if event.xdata is None:
        return

    x = max(
        0,
        min(
            duration,
            float(event.xdata)
        )
    )

    # First click = start

    if current_start is None:

        current_start = x

    # Second click = end

    elif current_end is None:

        if x <= current_start:

            messagebox.showwarning(
                "Invalid end time",
                "End time must be after start time."
            )

            return

        current_end = x

    # Third click starts a NEW unsaved occurrence

    else:

        current_start = x
        current_end = None

    update_plot()
    update_labels()


# ============================================================
# CLEAR CURRENT UNSAVED MARKER
# ============================================================

def clear_markers():

    global current_start
    global current_end

    current_start = None
    current_end = None

    update_plot()
    update_labels()


# ============================================================
# CREATE TEMP WAV
# ============================================================

def create_temp_wav(audio_samples):

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    temp_path = temp_file.name

    temp_file.close()

    audio_int16 = np.clip(
        audio_samples * 32767,
        -32768,
        32767
    ).astype(np.int16)

    with wave.open(
        temp_path,
        "wb"
    ) as wf:

        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(
            EXPECTED_SAMPLE_RATE
        )

        wf.writeframes(
            audio_int16.tobytes()
        )

    return temp_path


# ============================================================
# PLAY AUDIO USING WINDOWS
# ============================================================

def play_with_windows(audio_samples):

    try:

        temp_path = create_temp_wav(
            audio_samples
        )

        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "$p = New-Object "
                "System.Media.SoundPlayer "
                f"'{temp_path}'; "
                "$p.PlaySync(); "
                f"Remove-Item '{temp_path}' -Force"
            )
        ]

        subprocess.Popen(
            command,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

    except Exception as e:

        messagebox.showerror(
            "Playback error",
            str(e)
        )


# ============================================================
# PLAY FULL AUDIO
# ============================================================

def play_audio():

    if samples is not None:
        play_with_windows(samples)


# ============================================================
# PLAY CURRENT SELECTED ECHOEDGE
# ============================================================

def play_region():

    if (
        current_start is None
        or
        current_end is None
    ):

        messagebox.showwarning(
            "No region",
            "Mark START and END of ECHOEDGE first."
        )

        return

    start_sample = int(
        current_start *
        EXPECTED_SAMPLE_RATE
    )

    end_sample = int(
        current_end *
        EXPECTED_SAMPLE_RATE
    )

    region = samples[
        start_sample:end_sample
    ]

    play_with_windows(region)


# ============================================================
# SAVE CURRENT OCCURRENCE
# ============================================================

def save_current_occurrence():

    global current_start
    global current_end

    if (
        current_start is None
        or
        current_end is None
    ):

        messagebox.showwarning(
            "Missing annotation",
            "Mark both START and END of ECHOEDGE first."
        )

        return False

    if current_end <= current_start:

        messagebox.showwarning(
            "Invalid annotation",
            "END must be after START."
        )

        return False

    key = current_key()

    if key not in annotations:

        annotations[key] = []

    # --------------------------------------------------------
    # Prevent accidental duplicate
    # --------------------------------------------------------

    for existing in annotations[key]:

        if (
            abs(
                existing["start"] -
                current_start
            ) < 0.001
            and
            abs(
                existing["end"] -
                current_end
            ) < 0.001
        ):

            messagebox.showwarning(
                "Duplicate",
                "This ECHOEDGE occurrence is already saved."
            )

            return False

    # --------------------------------------------------------
    # Add occurrence
    # --------------------------------------------------------

    annotations[key].append({
        "start": current_start,
        "end": current_end
    })

    # Sort by time

    annotations[key] = sorted(
        annotations[key],
        key=lambda x: x["start"]
    )

    save_annotations()

    # Clear only the current unsaved marker

    current_start = None
    current_end = None

    update_plot()
    update_labels()

    status_label.config(
        text=(
            "Saved occurrence ✓   "
            f"Total annotations: "
            f"{count_total_annotations()}"
        )
    )

    return True


# ============================================================
# NEXT FILE
# ============================================================

def next_file():

    global current_index

    # If a marker is currently being edited,
    # don't silently lose it.

    if (
        current_start is not None
        or
        current_end is not None
    ):

        result = messagebox.askyesno(
            "Unsaved annotation",
            "You have an unsaved ECHOEDGE marker.\n\n"
            "Save it before moving to the next file?"
        )

        if result:

            if not save_current_occurrence():
                return

        else:

            clear_markers()

    if current_index < len(audio_files) - 1:

        current_index += 1

        load_audio()

        status_label.config(
            text="Ready"
        )

    else:

        messagebox.showinfo(
            "Complete",
            "All target positive recordings have been reviewed.\n\n"
            f"Total saved annotations: "
            f"{count_total_annotations()}"
        )


# ============================================================
# PREVIOUS FILE
# ============================================================

def previous_file():

    global current_index

    if (
        current_start is not None
        or
        current_end is not None
    ):

        result = messagebox.askyesno(
            "Unsaved annotation",
            "You have an unsaved marker.\n\n"
            "Discard it and go back?"
        )

        if not result:
            return

        clear_markers()

    if current_index == 0:
        return

    current_index -= 1

    load_audio()

    status_label.config(
        text=""
    )


# ============================================================
# KEYBOARD
# ============================================================

def keyboard(event):

    if event.keysym == "Right":

        next_file()

    elif event.keysym == "Left":

        previous_file()

    elif event.keysym == "space":

        play_audio()

    elif event.keysym.lower() == "r":

        play_region()

    elif event.keysym.lower() == "c":

        clear_markers()

    elif event.keysym.lower() == "s":

        save_current_occurrence()


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "EchoEdge V8 - Wake Word Annotation"
)

root.geometry(
    "1250x850"
)

root.bind(
    "<Key>",
    keyboard
)


# ============================================================
# TOP INFORMATION
# ============================================================

top_frame = tk.Frame(root)

top_frame.pack(
    fill="x",
    padx=10,
    pady=5
)


position_label = tk.Label(
    top_frame,
    text="",
    font=("Arial", 12, "bold")
)

position_label.pack(
    side="left",
    padx=10
)


file_label = tk.Label(
    top_frame,
    text="",
    font=("Arial", 11)
)

file_label.pack(
    side="left",
    padx=10
)


timestamp_label = tk.Label(
    root,
    text="ECHOEDGE: Not marked",
    font=("Arial", 11, "bold"),
    wraplength=1150
)

timestamp_label.pack(
    pady=5
)


# ============================================================
# WAVEFORM
# ============================================================

figure, ax = plt.subplots(
    figsize=(12, 5)
)

canvas = FigureCanvasTkAgg(
    figure,
    master=root
)

canvas_widget = canvas.get_tk_widget()

canvas_widget.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=5
)

canvas.mpl_connect(
    "button_press_event",
    on_click
)


# ============================================================
# BUTTONS
# ============================================================

button_frame = tk.Frame(root)

button_frame.pack(
    pady=10
)


tk.Button(
    button_frame,
    text="▶ Play Full",
    command=play_audio,
    width=14
).grid(
    row=0,
    column=0,
    padx=5
)


tk.Button(
    button_frame,
    text="▶ Play Selection",
    command=play_region,
    width=18
).grid(
    row=0,
    column=1,
    padx=5
)


tk.Button(
    button_frame,
    text="Save Occurrence",
    command=save_current_occurrence,
    width=18
).grid(
    row=0,
    column=2,
    padx=5
)


tk.Button(
    button_frame,
    text="Clear Current",
    command=clear_markers,
    width=15
).grid(
    row=0,
    column=3,
    padx=5
)


tk.Button(
    button_frame,
    text="← Previous",
    command=previous_file,
    width=14
).grid(
    row=0,
    column=4,
    padx=5
)


tk.Button(
    button_frame,
    text="Next →",
    command=next_file,
    width=14
).grid(
    row=0,
    column=5,
    padx=5
)


# ============================================================
# STATUS
# ============================================================

status_label = tk.Label(
    root,
    text="",
    font=("Arial", 11)
)

status_label.pack(
    pady=5
)


# ============================================================
# INSTRUCTIONS
# ============================================================

instructions = tk.Label(
    root,
    text=(
        "Click START of ECHOEDGE, then END.\n"
        "Save Occurrence = save this ECHOEDGE. "
        "For repeated recordings, repeat for every occurrence.\n"
        "Space = play full | R = play current selection | "
        "S = save | C = clear current | ←/→ = previous/next"
    ),
    font=("Arial", 10)
)

instructions.pack(
    pady=5
)


# ============================================================
# START
# ============================================================

load_audio()

root.mainloop()