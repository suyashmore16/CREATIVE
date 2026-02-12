import csv
from pathlib import Path
from typing import Optional

import numpy as np
from acquisition.lsl_eeg_reader import EEGChunk


class RawEEGLogger:
    """
    Logs raw EEG samples + timestamps to CSV.
    """

    def __init__(
        self,
        out_dir: str = "data/raw",
        subject_id: str = "S001",
        session_label: str = "baseline",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.filepath = self.out_dir / f"{subject_id}_{session_label}_raw.csv"
        self._writer: Optional[csv.writer] = None
        self._file = open(self.filepath, "w", newline="")
        self._writer = csv.writer(self._file)

    def write_header(self, n_channels: int):
        header = ["timestamp"] + [f"ch{i}" for i in range(n_channels)]
        self._writer.writerow(header)

    def log_chunk(self, chunk: EEGChunk):
        for i in range(chunk.data.shape[0]):
            row = [chunk.timestamps[i]] + chunk.data[i].tolist()
            self._writer.writerow(row)

    def close(self):
        self._file.close()
