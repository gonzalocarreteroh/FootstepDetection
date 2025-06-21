

from sklearn.model_selection import train_test_split
from data2 import preprocess_framed_audio_to_mfcc_spectrograms, extract_mfcc_patches
from models import build_cnn_model
from train_utils import train_model, evaluate_model, get_train_data
import numpy as np
import librosa
import time
from tensorflow.keras.utils import to_categorical

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf



# if __name__ == '__main__':
#     # Assuming `raw_audio_data` and `class_labels` are available
#     start_time = time.time()
#     raw_audio_data_g, class_labels_g = get_train_data("/home/timi/Desktop/COMP4531/project/samples/GonzaloDemo")
#     raw_audio_data_t, class_labels_t = get_train_data("/home/timi/Desktop/COMP4531/project/samples/TimDemo", label=1)
#     # raw_audio_data_e, class_labels_e = get_train_data("/home/timi/Desktop/COMP4531/project/samples/ElizabethDemo", label=2)
#     # raw_audio_data_y, class_labels_y = get_train_data("/home/timi/Desktop/COMP4531/project/samples/YeongDemo", label=3)

    
#     # Combine the data from both directories
#     # raw_audio_data = np.concatenate((raw_audio_data_g, raw_audio_data_t, raw_audio_data_e, raw_audio_data_y), axis=0)
#     # class_labels = np.concatenate((class_labels_g, class_labels_t, class_labels_e, class_labels_y), axis=0)
#     raw_audio_data = np.concatenate((raw_audio_data_g, raw_audio_data_t), axis=0)
#     class_labels = np.concatenate((class_labels_g, class_labels_t), axis=0)

#     # One-hot encode the labels
#     num_classes = len(set(class_labels))  # Infer the number of classes
#     class_labels = to_categorical(class_labels, num_classes=num_classes)

#     # Preprocess the data
#     X, y = preprocess_framed_audio_to_mfcc_spectrograms(raw_audio_data, class_labels, sr=48000, n_mfcc=128, frames_per_spectrogram=256)

#     #save the training data X and y
#     np.savez_compressed('/home/timi/Desktop/COMP4531/project/samples/training_data2.npz', X=X, y=y)
    
#     print("Preprocessed data shape:", X.shape)
#     print("Preprocessed labels shape:", y.shape)
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#     X_train = np.expand_dims(X_train, axis=-1)
#     X_val = np.expand_dims(X_val, axis=-1)     
#     X_test = np.expand_dims(X_test, axis=-1)
#     # Build the model
#     cnn_model = build_cnn_model()

#     # Train the model
#     with tf.device('/GPU:0'):
#         history = train_model(cnn_model, X_train, y_train, X_val, y_val, epochs = 20)

#     # Evaluate the model
#     evaluate_model(cnn_model, X_test, y_test)

#     model_save_path = "/home/timi/Desktop/COMP4531/project/saved_model3/cnn_model"
#     cnn_model.save(model_save_path)
#     print(f"Model saved to {model_save_path}")
#     # Calculate and print the elapsed time
#     elapsed_time = time.time() - start_time
#     print(f"Total execution time: {elapsed_time:.2f} seconds")
if __name__ == '__main__':
    # 1) Load raw signals and integer labels
    tf.config.list_physical_devices('GPU')
    print("GPU available:", tf.test.is_gpu_available())
    print("GPU device name:", tf.config.list_physical_devices('GPU'))
    raw_g, labels_g = get_train_data("./TimRealTrain", label=0)
    raw_t, labels_t = get_train_data("./GonzaloRealTrain",     label=1)
    # raw_e, labels_e = get_train_data("./samples/ElizabethDemo", label=2)
    # raw_y, labels_y = get_train_data("./samples/YeongDemo",     label=3)

    raw = np.concatenate((raw_g, raw_t), axis=0)
    labels = np.concatenate((labels_g, labels_t), axis=0)
    # raw = np.concatenate((raw_g, raw_t, raw_e, raw_y), axis=0)
    # labels = np.concatenate((labels_g, labels_t, labels_e, labels_y), axis=0)
    # print("raw shape:", raw.shape)
    # 2) Extract patches and repeat labels
    all_specs, all_lbls = [], []
    for sig, lbl in zip(raw, labels):
        specs = extract_mfcc_patches(
            sig, sr=48000,
            n_mfcc=13,
            frame_length=2048,
            hop_length=1024,
            frames_per_patch=128
        )
        all_specs.append(specs)
        all_lbls.append(np.full((specs.shape[0],), lbl, dtype=int))

    X = np.vstack(all_specs)
    y = np.concatenate(all_lbls)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    # 3) One‐hot encode
    num_classes = len(np.unique(y))
    y = to_categorical(y, num_classes)

    # 4) Train/test split & reshape
    X_train, X_temp, y_train, y_temp = train_test_split(X, y,
                                        test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test   = train_test_split(X_temp, y_temp,
                                        test_size=0.5, random_state=42)

    X_train = X_train[..., np.newaxis]
    X_val   = X_val[...,   np.newaxis]
    X_test  = X_test[...,  np.newaxis]

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)

    # 5) Build & train
    cnn = build_cnn_model(input_shape=(13, 128, 1), num_classes=num_classes)
    history = train_model(cnn, X_train, y_train, X_val, y_val, epochs=25)

    # 6) Evaluate & save
    evaluate_model(cnn, X_test, y_test)
    cnn.save("./timZalo/cnn_model")