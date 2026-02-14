import mne
from eeg_preprocessing import preprocess_eeg
import numpy as np
from eeg_features import extract_features
import joblib
from LSL import LSLEEGReader
from raw_logger import RawEEGLogger
from event_logger import EventLogger
from session_metadata import write_session_metadata
# --------------------
# LOAD + PREPROCESS
# --------------------
raw = mne.io.read_raw_edf("subject01.edf", preload=True)
epochs, ica = preprocess_eeg(raw)

sfreq = int(raw.info['sfreq'])

# --------------------
# FEATURE EXTRACTION
# --------------------
all_features = []

data = epochs.get_data()  # (n_epochs, n_channels, n_samples)

for epoch in data:
    features = extract_features(epoch, sfreq)
    all_features.append(features)

all_features = np.array(all_features)

print("Feature matrix shape:", all_features.shape)

