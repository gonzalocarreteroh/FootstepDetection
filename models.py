from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import librosa
import numpy as np

def build_cnn_model(input_shape=(128, 128, 1), num_classes=2):
    model = Sequential()

    # First Convolutional Block
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Second Convolutional Block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Third Convolutional Block
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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

    framed_data = np.array(framed_data, dtype=object)

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