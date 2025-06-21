import socket
import sounddevice as sd
import numpy as np
import struct
import soundfile as sf
import time
import os

HOST = '0.0.0.0'
PORT = 65432
SAMPLERATE = 48000
CHANNELS = 1
CHUNK = 1024  # Must match client
RECORD_SECONDS = 60
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

with sd.OutputStream(samplerate=SAMPLERATE, channels=CHANNELS, blocksize=CHUNK) as stream:
    while True:
        data = conn.recv(CHUNK)
        if not data:
            break

        # Unpack and convert to float32
        samples = struct.unpack('<' + 'f' * (len(data) // 4), data)
        samples_np = np.array(samples, dtype=np.float32).reshape(-1, CHANNELS)

        # Play back the audio
        stream.write(samples_np)

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

            # Reset buffer (keep extra samples if overfilled)
            extra = np.concatenate(buffer, axis=0)[FRAMES_PER_RECORDING:]
            buffer = [extra] if extra.size > 0 else []
            session_counter += 1

conn.close()
sock.close()