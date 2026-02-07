import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

# -----------------------------
# Frequency bands (Hz)
# -----------------------------
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

# -----------------------------
# Bandpower using Welch PSD
# -----------------------------
def bandpower(signal, sf, band):
    """
    signal : 1D numpy array
    sf     : sampling frequency (Hz)
    band   : (low, high) frequency band
    """
    f, psd = welch(signal, sf, nperseg=sf * 2)
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(psd[idx], f[idx])

# -----------------------------
# Relative band powers
# -----------------------------
def relative_bandpowers(signal, sf):
    total_power = bandpower(signal, sf, (0.5, 45))
    rel_powers = {}

    for band in BANDS:
        bp = bandpower(signal, sf, BANDS[band])
        rel_powers[band] = bp / total_power if total_power > 0 else 0.0

    return rel_powers

# -----------------------------
# Stress-related ratios
# -----------------------------
def stress_ratios(bandpowers):
    return {
        "beta_alpha": bandpowers["beta"] / bandpowers["alpha"]
        if bandpowers["alpha"] > 0 else 0.0,

        "theta_beta": bandpowers["theta"] / bandpowers["beta"]
        if bandpowers["beta"] > 0 else 0.0
    }

# -----------------------------
# Spectral entropy
# -----------------------------
def spectral_entropy(signal, sf):
    f, psd = welch(signal, sf, nperseg=sf * 2)
    psd_norm = psd / np.sum(psd)
    return entropy(psd_norm)

# -----------------------------
# Hjorth parameters
# -----------------------------
def hjorth_parameters(signal):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    var0 = np.var(signal)
    var1 = np.var(first_deriv)
    var2 = np.var(second_deriv)

    activity = var0
    mobility = np.sqrt(var1 / var0) if var0 > 0 else 0.0
    complexity = (np.sqrt(var2 / var1) / mobility
                  if var1 > 0 and mobility > 0 else 0.0)

    return activity, mobility, complexity

# -----------------------------
# Full feature extraction
# -----------------------------
def extract_features(epoch, sf):
    """
    epoch : numpy array of shape (n_channels, n_samples)
    sf    : sampling frequency (Hz)

    returns: 1D numpy array (feature vector)
    """
    features = []

    for ch in range(epoch.shape[0]):
        signal = epoch[ch, :]

        # Absolute band powers
        bp = {band: bandpower(signal, sf, BANDS[band]) for band in BANDS}

        # Relative band powers
        rbp = relative_bandpowers(signal, sf)

        # Ratios
        ratios = stress_ratios(bp)

        # Spectral entropy
        spec_ent = spectral_entropy(signal, sf)

        # Hjorth parameters
        activity, mobility, complexity = hjorth_parameters(signal)

        # Concatenate features
        features.extend([
            bp["delta"], bp["theta"], bp["alpha"], bp["beta"], bp["gamma"],
            rbp["delta"], rbp["theta"], rbp["alpha"], rbp["beta"], rbp["gamma"],
            ratios["beta_alpha"], ratios["theta_beta"],
            spec_ent,
            activity, mobility, complexity
        ])

    return np.array(features)
