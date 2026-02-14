import mne
from eeg_preprocessing import preprocess_eeg
import numpy as np
from eeg_features import extract_features
import joblib
from LSL import LSLEEGReader
from raw_logger import RawEEGLogger
from event_logger import EventLogger
from session_metadata import write_session_metadata

#preprocessing code
raw = mne.io.read_raw_edf("subject01.edf", preload=True)
epochs, ica = preprocess_eeg(raw)

#feature extraction main
sfreq = 256

# example EEG window: (channels, samples)
epoch = np.random.randn(4, 512)

features = extract_features(epoch, sfreq)

print(features)
print(features.shape)

