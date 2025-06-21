import numpy as np
import librosa


def extract_mfcc_patches(signal, sr=48000, n_mfcc=13, frame_length=2048, hop_length=1024, frames_per_patch=128):
    # 1. Compute MFCCs over whole signal
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length, center=False)# mfcc shape: (n_mfcc, n_time_frames)
    specs = []
    for i in range(0, mfcc.shape[1], frames_per_patch):
        patch = mfcc[:, i:i+frames_per_patch]
        if patch.shape[1] < frames_per_patch:
            pad_width = ((0,0), (0, frames_per_patch - patch.shape[1]))
            patch = np.pad(patch, pad_width, mode='constant')
        # normalize each patch
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)
        specs.append(patch)
    return np.stack(specs, axis=0)

def preprocess_framed_audio_to_mfcc_spectrograms(
    training_data, 
    training_labels, 
    sr=48000,          # Sampling rate = 48 kHz
    n_mfcc=128,        # Number of MFCC coefficients
    frames_per_spectrogram=128  # Group 128 frames into one spectrogram
):
    """
    Converts pre-framed audio (50k frames of 128 samples each) into 128x128 MFCC spectrograms.
    
    Args:
        training_data (list of np.ndarray): List of audio data, where each item is 
                                           a 2D array of shape (n_frames, 128).
        training_labels (list): Labels for each recording.
        sr (int): Sampling rate (default: 48000 Hz).
        n_mfcc (int): Number of MFCCs to compute (default: 128).
        frames_per_spectrogram (int): Number of frames per output spectrogram (default: 128).
    
    Returns:
        tuple: (spectrograms, labels)
            spectrograms (np.ndarray): Shape (n_spectrograms, 128, 128).
            labels (np.ndarray): Repeated labels for each spectrogram.
    """
    spectrograms = []
    labels = []
    hop_length = int(frames_per_spectrogram / 2)
    framed_data = []
    for signal, label in zip(training_data, training_labels):
        frames = librosa.util.frame(signal, frame_length=n_mfcc, hop_length=hop_length).T

        for i in range(0, len(frames), hop_length):
            framed_data.append(frames[i:i + frames_per_spectrogram])
            # Zero-pad if the last group is shorter than 128 frames
            if len(framed_data[-1]) < frames_per_spectrogram:
                pad_width = ((0, 0), (0, frames_per_spectrogram - len(framed_data[-1])))
                framed_data[-1] = np.pad(framed_data[-1], pad_width, mode='constant')
        # Append the label for each frame
            labels.append(label)
    # Convert lists to numpy arrays
    framed_data = np.array(framed_data)
    labels = np.array(labels)

    print(f"Framed data shape: {framed_data.shape}")
    print(f"Labels shape: {labels.shape}")

    count = 0
    for frames in framed_data:
        # Step 1: Compute MFCCs for each 128-sample frame
        mfccs = []
        print(f"Frame {count} / {len(framed_data)}")
        count += 1
        for frame in frames:
            
            mfcc = librosa.feature.mfcc(
                y=frame, 
                sr=sr, 
                n_mfcc=n_mfcc,
                n_fft=frames_per_spectrogram,       # FFT window size = frame length
                hop_length=frames_per_spectrogram,  # No overlap between frames
                center=False     # Disable centering to align with frame boundaries
            )
            mfccs.append(mfcc.squeeze())  # Shape: (128,)

        mfccs = np.array(mfccs).T  # Shape: (128, n_frames)
        # Step 2: Group every 128 MFCC vectors into a 128x128 spectrogram
        spectrogram = mfccs
        # Zero-pad if the last group is shorter than 128 frames
        if spectrogram.shape[1] < frames_per_spectrogram:
            pad_width = ((0, 0), (0, frames_per_spectrogram - spectrogram.shape[1]))
            spectrogram = np.pad(spectrogram, pad_width, mode='constant')

        # Normalize (0 mean, 1 std)
        spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
        spectrograms.append(spectrogram)

    return np.array(spectrograms), np.array(labels)


def test_audio_to_spectrograms(
    signal, 
    sr=48000,          # Sampling rate = 48 kHz
    n_mfcc=128,        # Number of MFCC coefficients
    frames_per_spectrogram=128  # Group 128 frames into one spectrogram
):
    spectrograms = []
    hop_length = 64
    framed_data = []
    frames = librosa.util.frame(signal, frame_length=n_mfcc, hop_length=hop_length).T
    for i in range(0, len(frames), hop_length):
        framed_data.append(frames[i:i + frames_per_spectrogram])

        # Zero-pad if the last group is shorter than 128 frames
        if len(framed_data[-1]) < frames_per_spectrogram:
            pad_width = ((0, 0), (0, frames_per_spectrogram - len(framed_data[-1])))
            framed_data[-1] = np.pad(framed_data[-1], pad_width, mode='constant')

    framed_data = np.array(framed_data)

    print(f"Framed data shape: {framed_data.shape}")

    count = 0
    for frames in framed_data:
        # Step 1: Compute MFCCs for each 128-sample frame
        mfccs = []
        print(f"Frame {count} / {len(framed_data)}")
        count += 1
        for frame in frames:
            
            mfcc = librosa.feature.mfcc(
                y=frame, 
                sr=sr, 
                n_mfcc=n_mfcc,
                n_fft=frames_per_spectrogram,       # FFT window size = frame length
                hop_length=frames_per_spectrogram,  # No overlap between frames
                center=False     # Disable centering to align with frame boundaries
            )
            mfccs.append(mfcc.squeeze())  # Shape: (128,)

        mfccs = np.array(mfccs).T  # Shape: (128, n_frames)
        # Step 2: Group every 128 MFCC vectors into a 128x128 spectrogram
        spectrogram = mfccs
        # Zero-pad if the last group is shorter than 128 frames
        if spectrogram.shape[1] < frames_per_spectrogram:
            pad_width = ((0, 0), (0, frames_per_spectrogram - spectrogram.shape[1]))
            spectrogram = np.pad(spectrogram, pad_width, mode='constant')

        # Normalize (0 mean, 1 std)
        spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
        spectrograms.append(spectrogram)

    return np.array(spectrograms)
