import librosa
import soundfile as sf
import os
import re

# Define source and destination directories
source_dir = "/home/zhyuan/Desktop/Qwen-Audio/ambisonics_1to50"
dest_dir = "/home/zhyuan/Desktop/Qwen-Audio/W_components"

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Iterate over all .wav files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith(".wav"):
        input_file = os.path.join(source_dir, filename)

        # Extract the identifier (R000X) from the filename using regex
        match = re.search(r"(R\d{4})", filename)
        if match:
            file_id = match.group(1)  # Extracted file ID (e.g., R0005)
        else:
            print(f"Skipping {filename}: Unable to extract ID")
            continue

        # Load the ambisonic file
        audio, sr = librosa.load(input_file, mono=False)

        # Check if the file has multiple channels
        if audio.ndim == 1:
            print(f"Skipping {filename}: Not a multi-channel file")
            continue

        # Extract the W component (first channel)
        W_component = audio[0, :]

        # Define the output filename
        output_file = os.path.join(dest_dir, f"W_component_{file_id}.wav")

        # Save the W component as a mono WAV file
        sf.write(output_file, W_component, sr)

        print(f"Extracted W component saved to: {output_file}")

print("Processing complete.")
