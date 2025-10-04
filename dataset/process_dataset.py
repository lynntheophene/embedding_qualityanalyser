import mne
import numpy as np

# Load the EEG data
raw = mne.io.read_raw_edf('S001R04.edf', preload=True)

# Basic preprocessing
raw.filter(0.5, 45)  # Bandpass filter
raw.set_eeg_reference('average')  # Re-reference

# Extract epochs (2-second windows)
events = mne.make_fixed_length_events(raw, duration=2.0)
epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None)

# Create embeddings (PSD features)
psds = epochs.compute_psd(method='welch', fmin=0.5, fmax=45, n_fft=256)
psds_data = psds.get_data()
freqs = psds.freqs

# Flatten to create embeddings
embeddings = psds_data.reshape(psds_data.shape[0], -1)

# Save
np.save('physionet_embeddings.npy', embeddings)
print(f"✓ Created {embeddings.shape[0]} samples with {embeddings.shape[1]} features")