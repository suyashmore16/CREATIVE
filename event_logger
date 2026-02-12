import csv
from pathlib import Path
import time


class EventLogger:
    def __init__(
        self,
        out_dir: str = "data/raw",
        subject_id: str = "S001",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.out_dir / f"{subject_id}_events.csv"

        self._file = open(self.filepath, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "event"])

    def mark(self, event: str):
        ts = time.time()
        self._writer.writerow([ts, event])
        print(f"[EVENT] {event} @ {ts:.2f}")

    def close(self):
        self._file.close()
