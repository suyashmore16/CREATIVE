from acquisition.lsl_eeg_reader import LSLEEGReader
from acquisition.raw_logger import RawEEGLogger
from acquisition.event_logger import EventLogger
from session_metadata import write_session_metadata

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

    print("\nControls:")
    print("  b → baseline start")
    print("  s → stress start")
    print("  r → recovery start")
    print("  q → quit\n")

    try:
        while True:
            chunk = reader.pull_chunk(max_samples=256)
            if chunk is not None:
                raw_logger.log_chunk(chunk)

            key = input().strip().lower()
            if key == "b":
                event_logger.mark("baseline_start")
            elif key == "s":
                event_logger.mark("stress_start")
            elif key == "r":
                event_logger.mark("recovery_start")
            elif key == "q":
                break
    finally:
        raw_logger.close()
        event_logger.close()
        print("Session saved cleanly.")

if __name__ == "__main__":
    main()
