import queue
import socket
import struct
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import threading
from distutils.command import sdist
import sounddevice as sd
import tensorflow as tf
import os
import numpy as np
from models import test_audio_to_spectrograms
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity

# ID-to-name mapping (update this as needed)
id_to_name = {
    0: "Gonzalo",
    1: "Tim",
    3: "Elizabeth",
    4: "Yoeng"
}

log_index = 1  # Start numbering
latest_prediction = 1  # Simulated example; replace with live model input
is_processing = False

model = tf.keras.models.load_model('./saved_model/cnn_model')
layer_names = ['conv2d', 'maxpool2d', 'dropout', 'conv2d_1', 'maxpool2d_1', 'dropout_1', 'conv2d_2', 'max_pooling2d_2', 'dropout_2', 'flatten', 'dense', 'dropout_3', 'dense_1']
outputs = [model.get_layer(name).output for name in layer_names if name in [l.name for l in model.layers]]
embedding_models = [tf.keras.models.Model(inputs=model.input, outputs=output) for output in outputs]

result_queue = queue.Queue()
RECORD_SECONDS = 10


# 1. Socket listener runs in background
def listen_for_data():
    global is_processing
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
                data = conn.recv(CHUNK)
                if not data:
                    break
                if is_processing:
                    # print("Dropping incoming audio chunk — model is still processing")
                    continue  # Skip this chunk
                if not is_processing:

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
                        filename = f"recordings/audio_{session_counter}.wav"
                        sf.write(filename, full_audio, SAMPLERATE)
                        print(f"Saved {filename}")
                        # is_processing = True

                        # threading.Thread(target=run_model_and_enqueue_result, args=(filename,), daemon=True).start()

                        # Reset buffer (keep extra samples if overfilled)
                        #extra = np.concatenate(buffer, axis=0)[FRAMES_PER_RECORDING:]
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
    # if max_similarity >= threshold:
    #     return max_key, max_similarity
    # else:
    #     return -1, max_similarity

def run_model_and_enqueue_result(filename):
    global is_processing

    # Load and preprocess audio as needed
    audio_data, _ = sf.read(filename)
    X_test = test_audio_to_spectrograms(audio_data)  # e.g., reshape or extract features
    X_test = np.expand_dims(X_test, axis = -1)

    test_result, similarity = is_similar(
        X_test,
        "./embeddings2/train_embeddings.npz",
        threshold= .9,
        embedding_models=embedding_models
    )
    if (similarity > .90):
        result_queue.put(test_result)
    else:
        result_queue.put(-1)
    is_processing = False  # Allow socket to read again


def update_names(prediction):
    global log_index

    timestamp = datetime.now().strftime("%H:%M:%S")
    if prediction == -1:
        log_entry = f"[{log_index}] Unknown person detected at {timestamp}."
    else:
        name = id_to_name.get(prediction, f"ID {prediction}")
        log_entry = f"[{log_index}] Person detected at {timestamp}. Individual identified: {name}"

    print(f"Adding: {log_entry}")
    name_listbox.insert(tk.END, log_entry)
    name_listbox.yview_moveto(1.0)  # Auto-scroll to bottom
    log_index += 1

# def auto_refresh():
#     # Replace this simulated input with actual prediction logic
#     global latest_prediction
#     update_names(latest_prediction)

#     # For testing: cycle through values (can be removed in production)
#     latest_prediction = (latest_prediction + 1) % 6 - 1  # Cycles through -1 to 4

#     # Repeat every 3 seconds
#     root.after(3000, auto_refresh)

def check_prediction_queue():
    try:
        prediction = result_queue.get()
        update_names(prediction)
    except queue.Empty:
        pass
    finally:
        root.after(10, check_prediction_queue);

# Create GUI window
root = tk.Tk()
root.title("Footsteps Detection Log")
root.geometry("600x400")
root.configure(bg="#f5f5f5")

# Style
style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background="#f5f5f5", font=("Helvetica", 16, "bold"))
style.configure("TListbox", font=("Helvetica", 12))

# Title
label = ttk.Label(root, text="Footstep Detection Log")
label.pack(pady=20)

# Scrollable listbox
frame = ttk.Frame(root)
frame.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
name_listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, height=10,
                          bg="white", fg="black", font=("Helvetica", 12),
                          highlightthickness=1, bd=1, relief="solid")
scrollbar.config(command=name_listbox.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
name_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Start refreshing
# auto_refresh()

# Start GUI loop
threading.Thread(target=listen_for_data, args=(), daemon=True).start()
check_prediction_queue()
root.mainloop()
