import mne
from eeg_preprocessing import preprocess_eeg

raw = mne.io.read_raw_edf("subject01.edf", preload=True)
epochs, ica = preprocess_eeg(raw)
