# src/acquisition/lsl_eeg_reader.py
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
from pylsl import StreamInlet, resolve_byprop


@dataclass
class EEGChunk:
    data: np.ndarray          # shape: (n_samples, n_channels)
    timestamps: np.ndarray    # shape: (n_samples,)


class LSLEEGReader:
    """
    Connects to an LSL EEG stream and maintains a rolling buffer.
    """
    def __init__(
        self,
        stream_type: str = "EEG",
        stream_name: Optional[str] = None,
        buffer_seconds: float = 10.0,
        expected_srate: Optional[float] = None,
        timeout_resolve: float = 10.0,
    ) -> None:
        self.stream_type = stream_type
        self.stream_name = stream_name
        self.buffer_seconds = float(buffer_seconds)
        self.expected_srate = expected_srate
        self.timeout_resolve = float(timeout_resolve)

        self.inlet: Optional[StreamInlet] = None
        self.n_channels: Optional[int] = None
        self.srate: Optional[float] = None

        self._buf_x: Deque[np.ndarray] = deque()
        self._buf_t: Deque[float] = deque()
        self._max_samples: Optional[int] = None

    def connect(self) -> None:
        # Resolve stream
        if self.stream_name is not None:
            streams = resolve_byprop("name", self.stream_name, timeout=self.timeout_resolve)
            if not streams:
                raise RuntimeError(f"No LSL stream found with name='{self.stream_name}'.")
        else:
            streams = resolve_byprop("type", self.stream_type, timeout=self.timeout_resolve)
            if not streams:
                raise RuntimeError(f"No LSL stream found with type='{self.stream_type}'.")

        # Connect inlet
        self.inlet = StreamInlet(streams[0], max_buflen=60, recover=True)

        info = self.inlet.info()
        self.n_channels = info.channel_count()
        self.srate = float(info.nominal_srate())  # can be 0 for irregular streams

        if self.expected_srate is not None and self.srate not in (0.0, self.expected_srate):
            # Not fatal, but should be noticed early
            print(f"[WARN] Stream nominal_srate={self.srate} but expected {self.expected_srate}")

        # Determine buffer length in samples (fallback if srate=0)
        effective_srate = self.srate if self.srate and self.srate > 0 else (self.expected_srate or 250.0)
        self._max_samples = int(round(self.buffer_seconds * effective_srate))

        print(f"[LSL] Connected: name='{info.name()}', type='{info.type()}', "
              f"channels={self.n_channels}, srate={self.srate}")

    def pull_chunk(self, max_samples: int = 256, timeout: float = 0.0) -> Optional[EEGChunk]:
        """
        Pull up to max_samples from LSL. Returns None if no samples available.
        """
        if self.inlet is None:
            raise RuntimeError("Not connected. Call connect() first.")

        samples, timestamps = self.inlet.pull_chunk(timeout=timeout, max_samples=max_samples)
        if not timestamps:
            return None

        x = np.asarray(samples, dtype=np.float32)
        t = np.asarray(timestamps, dtype=np.float64)

        # Ensure 2D shape: (n_samples, n_channels)
        if x.ndim == 1:
            x = x[:, None]

        # Append to rolling buffer
        for i in range(x.shape[0]):
            self._buf_x.append(x[i])
            self._buf_t.append(float(t[i]))

        # Trim buffer
        assert self._max_samples is not None
        while len(self._buf_t) > self._max_samples:
            self._buf_x.popleft()
            self._buf_t.popleft()

        return EEGChunk(data=x, timestamps=t)

    def get_buffer(self) -> EEGChunk:
        """
        Return the current rolling buffer as arrays.
        """
        if self.n_channels is None:
            raise RuntimeError("Not connected.")
        if not self._buf_t:
            return EEGChunk(data=np.zeros((0, self.n_channels), dtype=np.float32),
                            timestamps=np.zeros((0,), dtype=np.float64))

        data = np.stack(list(self._buf_x), axis=0)  # (n_samples, n_channels)
        ts = np.asarray(list(self._buf_t), dtype=np.float64)
        return EEGChunk(data=data, timestamps=ts)

# src/main.py
import time
from acquisition.lsl_eeg_reader import LSLEEGReader

def main():
    reader = LSLEEGReader(stream_type="EEG", expected_srate=250.0, buffer_seconds=10.0)
    reader.connect()

    t0 = time.time()
    n_total = 0

    while True:
        chunk = reader.pull_chunk(max_samples=256, timeout=0.0)
        if chunk is not None:
            n_total += chunk.data.shape[0]

        # Print status every ~1 second
        if time.time() - t0 >= 1.0:
            buf = reader.get_buffer()
            print(f"samples_read_last_sec={n_total}, buffer_samples={buf.data.shape[0]}, "
                  f"channels={buf.data.shape[1] if buf.data.size else 'NA'}")
            n_total = 0
            t0 = time.time()

        time.sleep(0.005)

if __name__ == "__main__":
    main()

# src/signal_processing/bandpass.py
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    lowcut: float = 1.0,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """
    Bandpass filter EEG data.

    data: (n_samples, n_channels)
    sfreq: sampling frequency (Hz)
    """
    nyq = 0.5 * sfreq
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0)

from signal_processing.bandpass import bandpass_filter

buf = reader.get_buffer()

if buf.data.shape[0] > 0:
    filtered = bandpass_filter(
        buf.data,
        sfreq=reader.srate,
        lowcut=1.0,
        highcut=40.0
    )


