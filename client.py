import socket
import sounddevice as sd
import numpy as np
import struct

SERVER_IP = '172.20.10.2'  # Replace with your laptop IP
PORT = 65432
SAMPLERATE = 48000
CHANNELS = 1
CHUNK = 1024  # Frames per send

# Open TCP socket connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, PORT))
print("Connected to server, streaming audio...")

def callback(indata, frames, time, status):
    if status:
        print(status)
    # Convert float32 numpy array to bytes
    data_bytes = struct.pack('<' + 'f' * len(indata), *indata.flatten())
    sock.sendall(data_bytes)

# Start audio stream and send data
with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS,
                    blocksize=CHUNK, callback=callback):
    input("Press Enter to stop...\n")

sock.close()