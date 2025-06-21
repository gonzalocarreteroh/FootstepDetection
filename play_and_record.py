import time
import numpy as np
import sounddevice as sd
import threading
import matplotlib.pyplot as plt
import argparse

recording = None
accu_frames = 0  # Initialize a counter for accumulated frames
start_idx = 0
mic1_channel = None
# Ignore Frames collected in first 500ms
ignore_frame = 44100 / 1000 * 500

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


def play_and_record(channels, input_device, output_device, samplerate):

    def audio_callback(indata, frames, time):
        global accu_frames, recording, mic1_channel, ignore_frame

        if mic1_channel is None:
            threshold = 0.75
            max_amplitudes = np.max(np.abs(indata), axis=0)
            exceeding_signals = np.where(max_amplitudes > threshold)[0]
            if len(exceeding_signals) > 0:
                mic1_channel = exceeding_signals[0]
                print(f"Self-Checking Complete, Mic1 on {mic1_channel}")
            return
        else:
            if ignore_frame > 0:
                ignore_frame -= frames
                return
        
        # Determine the desired end index
        end_index = accu_frames + frames

        # Check if we need to expand the recording array
        if end_index > recording.shape[0]:
            # Calculate the new size of the recording array
            new_size = max(end_index, recording.shape[0] * 2)  # Double the size or at least reach end_index
            new_recording = np.zeros((new_size, channels))  # Create a new larger array
            new_recording[:recording.shape[0]] = recording  # Copy old data to the new array
            recording = new_recording  # Point `recording` to the new array

        # Accumulate data into the recording array
        shifted_indata = np.roll(indata, -mic1_channel, axis=1)
        recording[accu_frames:end_index] = shifted_indata[:frames]

        # Update the number of accumulated frames
        accu_frames += frames

    def output_callback(outdata, frames, time):
        global start_idx, mic1_channel
        if mic1_channel is None or ignore_frame > 0:
            return
        t = (start_idx + np.arange(frames)) / samplerate
        t = t.reshape(-1, 1)
        outdata[:] = args.amplitude * np.sin(2 * np.pi * args.frequency * t)
        start_idx += frames

    def callback(indata, outdata, frames, time, status):
        audio_callback(indata, frames, time)
        output_callback(outdata, frames, time)

    def finalize_recording():
        print("Recording Stop, generating plots")
        global recording, accu_frames
        # Resize recording to remove unused zeros
        if accu_frames < recording.shape[0]:
            recording = recording[:accu_frames]

    with sd.Stream(device=(args.input_device, args.output_device),
                   samplerate=args.samplerate,
                   channels=args.channels, callback=callback):
        print(f"Start Recording, press ENTER to stop (if self-checking is enabled, tap MIC1 to complete self-checking)")
        input()

    finalize_recording()

    return recording

def save_plots(recording):
    for channel in range(recording.shape[1]):
        plt.figure()
        plt.plot(recording[:, channel])
        plt.title(f'Channel {channel + 1} Recording')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.savefig(f'channel_{channel + 1}.png')
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play a sine wave and record audio.')
    parser.add_argument('-c', '--channels', type=int, default=8, help='Number of recording channels (default: 8)')
    parser.add_argument('--list-devices', action='store_true', help='List audio devices and exit')
    parser.add_argument('--self-checking', action='store_true', help='Run the self-checking mechanism to reorder your channels')
    parser.add_argument('--input-device', type=int_or_str, help='Input device (numeric ID or substring)')
    parser.add_argument('--output-device', type=int_or_str, help='Output device (numeric ID or substring)')
    parser.add_argument('--samplerate', type=int, default=44100, help='Sample rate (default: 44100 Hz)')
    parser.add_argument('frequency', nargs='?', metavar='FREQUENCY', type=float, default=500, help='frequency in Hz (default: %(default)s)')
    parser.add_argument('-a', '--amplitude', type=float, default=0.2, help='amplitude (default: %(default)s)')
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        exit()

    if not args.self_checking:
        mic1_channel = 0

    # global recording
    # Intialize a recording space for 10s (we will expand it if needed)
    recording = np.zeros((args.samplerate * 10, args.channels))

    # Play the audio file and record
    recording = play_and_record(args.channels, args.input_device, args.output_device, args.samplerate)

    # Save the recording as plots
    save_plots(recording)
    print("Plots saved for each channel.")
