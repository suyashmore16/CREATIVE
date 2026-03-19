import sys
import select
import time
import numpy as np
import mne

from acquisition.lsl_eeg_reader import LSLEEGReader
from acquisition.raw_logger import RawEEGLogger
from acquisition.event_logger import EventLogger
from session_metadata import write_session_metadata

from preprocessing_pipeline import (
    apply_filters,
    rereference,
    epoch_data,
    normalize_epochs,
    remove_artifacts_ica
)

def key_pressed():
    return select.select([sys.stdin], [], [], 0)[0]


def main():
    subject_id = input("Subject ID: ").strip()
    notes = input("Session notes (optional): ").strip()

    reader = LSLEEGReader(expected_srate=250.0, buffer_seconds=10.0)
    reader.connect()

    write_session_metadata(
        subject_id=subject_id,
        srate=reader.srate or 250.0,
        n_channels=reader.n_channels,
        channels=[f"ch{i}" for i in range(reader.n_channels)],
        notes=notes,
    )

    raw_logger = RawEEGLogger(subject_id=subject_id, session_label="raw")
    event_logger = EventLogger(subject_id=subject_id)
    raw_logger.write_header(reader.n_channels)

    # ---------------------------
    # REAL-TIME BUFFER
    # ---------------------------
    buffer = []
    buffer_size = int(5 * reader.srate)  # 5 seconds window

    # Create MNE info
    info = mne.create_info(
        ch_names=[f"ch{i}" for i in range(reader.n_channels)],
        sfreq=reader.srate,
        ch_types="eeg"
    )

    print("\nControls:")
    print("  b → baseline start")
    print("  s → stress start")
    print("  r → recovery start")
    print("  q → quit\n")

    try:
        while True:
            chunk = reader.pull_chunk(max_samples=256, timeout=0.0)

            if chunk is not None:
                raw_logger.log_chunk(chunk)

                # Add to buffer
                buffer.append(chunk)

                # Flatten buffer
                data = np.concatenate(buffer, axis=0)

                # Keep buffer size fixed
                if len(data) > buffer_size:
                    data = data[-buffer_size:]
                    buffer = [data]

                # ---------------------------
                # RUN PREPROCESSING
                # ---------------------------
                if len(data) >= buffer_size:
                    raw = mne.io.RawArray(data.T, info, verbose=False)

                    # Apply lightweight real-time steps
                    raw = apply_filters(raw)
                    raw = rereference(raw)

                    # ICA occasionally (expensive)
                    # You can run this every N seconds instead
                    # raw, _ = remove_artifacts_ica(raw)

                    epochs = epoch_data(raw, epoch_length=2.0, overlap=0.5)
                    epochs = normalize_epochs(epochs)

                    # 👉 This is where your ML model would go
                    print(f"Processed {len(epochs)} epochs")

            if key_pressed():
                key = sys.stdin.readline().strip().lower()

                if key == "b":
                    event_logger.mark("baseline_start")
                elif key == "s":
                    event_logger.mark("stress_start")
                elif key == "r":
                    event_logger.mark("recovery_start")
                elif key == "q":
                    break

            time.sleep(0.002)

    finally:
        raw_logger.close()
        event_logger.close()
        print("Session saved cleanly.")


if __name__ == "__main__":
    main()
