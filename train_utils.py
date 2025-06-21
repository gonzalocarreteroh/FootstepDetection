from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import os
import librosa

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """
    Trains the CNN model.

    Args:
        model (tf.keras.Model): Compiled CNN model.
        X_train (np.array): Training data.
        y_train (np.array): One-hot encoded training labels.
        X_val (np.array): Validation data.
        y_val (np.array): One-hot encoded validation labels.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.

    Returns:
        history: Training history object containing loss and accuracy metrics.
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs
    )
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on test data.

    Args:
        model (tf.keras.Model): Trained CNN model.
        X_test (np.array): Test data.
        y_test (np.array): One-hot encoded test labels.

    Returns:
        None: Prints out the classification report and accuracy.
    """
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    print("Classification Report:")
    print(classification_report(y_test_classes, y_pred_classes))
    print(f"Accuracy: {accuracy_score(y_test_classes, y_pred_classes):.4f}")


def get_train_data(directory, file_extension=".wav", label=0, sampling_rate=48000):
    """
    Loads audio files from a specified directory, extracts features, and assigns labels.

    Args:
        directory (str): Path to the directory containing audio files.
        file_extension (str): File extension of the audio files to load.
        label (int): Label to assign to the loaded audio files.

    Returns:
        tuple: Tuple containing raw audio data and corresponding class labels.
    """
    raw_audio_data = []
    class_labels = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        
        if filename.endswith(file_extension):
            # Load the audio file
            file_path = os.path.join(directory, filename)
            signal, sr = librosa.load(file_path, sr=sampling_rate)
            # Append the raw audio data and label
            raw_audio_data.append(signal)
            class_labels.append(label)
    
    return np.array(raw_audio_data), np.array(class_labels)