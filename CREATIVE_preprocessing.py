import mne
import numpy as np


# ---------------------------
# 1. Load & Prepare Raw EEG
# ---------------------------

def prepare_raw(raw, montage="standard_1020"):
    """
    Set montage and ensure EEG channels are correctly labeled
    """
    raw.pick_types(eeg=True, eog=True, ecg=True, exclude="bads")
    raw.set_montage(montage, match_case=False)
    return raw


# ---------------------------
# 2. Filtering
# ---------------------------

def apply_filters(raw, l_freq=1.0, h_freq=40.0, notch_freq=60):
    """
    Bandpass + Notch filtering
    """
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin")
    raw.notch_filter(freqs=[notch_freq])
    return raw


# ---------------------------
# 3. Re-referencing
# ---------------------------

def rereference(raw, ref_type="average"):
    """
    Re-reference EEG signals
    """
    raw.set_eeg_reference(ref_type=ref_type)
    return raw


# ---------------------------
# 4. ICA Artifact Removal
# ---------------------------

def remove_artifacts_ica(raw, n_components=15, random_state=42):
    """
    Removes EOG/ECG artifacts using ICA
    """
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=random_state,
        max_iter="auto"
    )

    ica.fit(raw)

    # Detect eye blink components
    eog_inds, _ = ica.find_bads_eog(raw)
    ecg_inds, _ = ica.find_bads_ecg(raw)

    ica.exclude = list(set(eog_inds + ecg_inds))
    raw = ica.apply(raw)

    return raw, ica


# ---------------------------
# 5. Epoching
# ---------------------------

def epoch_data(raw, epoch_length=2.0, overlap=0.5):
    """
    Create fixed-length epochs for ML
    """
    events = mne.make_fixed_length_events(
        raw,
        duration=epoch_length,
        overlap=epoch_length * overlap
    )

    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=epoch_length,
        baseline=None,
        preload=True
    )

    return epochs


# ---------------------------
# 6. Normalization
# ---------------------------

def normalize_epochs(epochs, method="zscore"):
    """
    Normalize per-channel per-epoch
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

    if method == "zscore":
        mean = data.mean(axis=2, keepdims=True)
        std = data.std(axis=2, keepdims=True) + 1e-8
        data = (data - mean) / std

    elif method == "minmax":
        min_val = data.min(axis=2, keepdims=True)
        max_val = data.max(axis=2, keepdims=True)
        data = (data - min_val) / (max_val - min_val + 1e-8)

    epochs._data = data
    return epochs


# ---------------------------
# 7. Full Pipeline
# ---------------------------

def preprocess_eeg(
    raw,
    l_freq=1.0,
    h_freq=40.0,
    notch_freq=60,
    epoch_length=2.0,
    overlap=0.5
):
    """
    Complete EEG preprocessing pipeline
    """
    raw = prepare_raw(raw)
    raw = apply_filters(raw, l_freq, h_freq, notch_freq)
    raw = rereference(raw)
    raw, ica = remove_artifacts_ica(raw)
    epochs = epoch_data(raw, epoch_length, overlap)
    epochs = normalize_epochs(epochs)

    return epochs, ica
