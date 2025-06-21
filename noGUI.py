import queue
import socket
import struct
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import threading
from threading import Lock
from distutils.command import sdist
import sounddevice as sd
import tensorflow as tf
import os
import numpy as np
from models import test_audio_to_spectrograms
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity
import librosa
from data2 import extract_mfcc_patches

# ID-to-name mapping (update this as needed)
id_to_name = {
    0: "Gonzalo",
    1: "Tim",
    3: "Elizabeth",
    4: "Yoeng"
}

log_index = 1  # Start numbering
latest_prediction = 1  # Simulated example; replace with live model input

model = tf.keras.models.load_model('./timZalo/cnn_model')
layer_names = ['conv2d', 'maxpool2d', 'dropout', 'conv2d_1', 'maxpool2d_1', 'dropout_1', 'conv2d_2', 'max_pooling2d_2', 'dropout_2', 'flatten', 'dense', 'dropout_3', 'dense_1']
outputs = [model.get_layer(name).output for name in layer_names if name in [l.name for l in model.layers]]
embedding_models = [tf.keras.models.Model(inputs=model.input, outputs=output) for output in outputs]

result_queue = queue.Queue()
file_queue = queue.Queue()
processing_mutex = Lock()

RECORD_SECONDS = 60


# 1. Socket listener runs in background
def listen_for_data():
    global is_processing
    global testfile
    HOST = '0.0.0.0'
    PORT = 65432
    SAMPLERATE = 48000
    CHANNELS = 1
    CHUNK = 1024  # Must match client
    FRAMES_PER_RECORDING = SAMPLERATE * RECORD_SECONDS

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)
    print(f"Server listening on {HOST}:{PORT}...")

    conn, addr = sock.accept()
    print(f"Connected by {addr}")

    # Buffer to accumulate samples
    buffer = []

    session_counter = 1
    os.makedirs("recordings", exist_ok=True)

    try:
        with sd.OutputStream(samplerate=SAMPLERATE, channels=CHANNELS, blocksize=CHUNK) as stream:
            while True:
                data = conn.recv(CHUNK * CHANNELS)
                if not data:
                    break

                if processing_mutex.locked():
                    continue

                # Unpack and convert to float32
                samples = struct.unpack('<' + 'f' * (len(data) // 4), data)
                samples_np = np.array(samples, dtype=np.float32).reshape(-1, CHANNELS)

                # Play back the audio
                # stream.write(samples_np)

                # Append to buffer
                buffer.append(samples_np)

                # Check if enough samples are buffered to save
                total_samples = sum(chunk.shape[0] for chunk in buffer)
                if total_samples >= FRAMES_PER_RECORDING:
                    # Concatenate and save to file
                    full_audio = np.concatenate(buffer, axis=0)[:FRAMES_PER_RECORDING]
                    filename = f"./recordings/audio_{session_counter}.wav"
                    sf.write(filename, full_audio, SAMPLERATE)
                    print(f"Saved {filename}")
                    
                    file_queue.put(filename, block=True)

                    buffer = []
                    session_counter += 1

    finally:
        conn.close()
        sock.close()


def is_similar(test_sample, saved_embeddings_path, threshold=0.8, layer_index=-3, embedding_models=None):
    # Load saved embeddings
    saved_data = np.load(saved_embeddings_path)

    # Get test embedding
    test_embedding = embedding_models[layer_index].predict(test_sample)
    test_embedding = np.mean(test_embedding, axis=0)

    max_similarity = 0
    max_key = -1

    for key in saved_data.files:
        saved_embedding = saved_data[key]

        # Compute cosine similarity
        similarity = cosine_similarity(test_embedding.reshape(1, -1), saved_embedding.reshape(1, -1))[0][0]
        print(key, ": ", similarity)
        if (max_similarity < similarity):
            max_similarity = similarity
            max_key = int(key)
    
    return max_key, max_similarity

def run_model_and_enqueue_result(filename):
    # Load and preprocess audio as needed
    path = filename
    signal, sr = librosa.load(path, sr=48000)
    label = -1;

    # X_test = test_audio_to_spectrograms(signal, sr=48000, n_mfcc=13, frames_per_spectrogram=128)
    # X_test = np.expand_dims(X_test, axis = -1)

    X_test = extract_mfcc_patches(
            signal, sr=48000,
            n_mfcc=13,
            frame_length=2048,
            hop_length=1024,
            frames_per_patch=128
        )
    X_test = np.expand_dims(X_test, axis=-1) 

    test_result, similarity = is_similar(
        X_test,
        "./real_embeddings/train_embeddings.npz",
        threshold= .95,
        embedding_models=embedding_models
    )
    if (similarity > .95):
        label = test_result
    return label


def update_names(prediction):
    global log_index

    timestamp = datetime.now().strftime("%H:%M:%S")
    if prediction == -1:
        log_entry = f"[{log_index}] Unknown person detected at {timestamp}."
    else:
        name = id_to_name.get(prediction, f"ID {prediction}")
        log_entry = f"[{log_index}] Person detected at {timestamp}. Individual identified: {name}"

    print(f"Adding: {log_entry}")

def check_prediction():
    testfile = file_queue.get(block=True)
    processing_mutex.acquire()
    prediction = run_model_and_enqueue_result(testfile)
    update_names(prediction)
    processing_mutex.release()


# Start GUI loop
threading.Thread(target=listen_for_data, args=(), daemon=True).start()
while True:
    # check_prediction()
    continue
    