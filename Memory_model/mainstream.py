import time
import os
import librosa
import soundfile as sf
import numpy as np
from queue import Queue
import argparse
import torch
import gc
import sys
import traceback
import pandas as pd
import re
import json
import traceback

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
from echoic_memory import EchoicMemory
from pattern_change_detector import PatternChangeDetector
from Qwen_processor import QwenAudioDescription


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=320)
    parser.add_argument("--mel_bins", type=int, default=64)
    parser.add_argument("--fmin", type=int, default=50)
    parser.add_argument("--fmax", type=int, default=14000)
    parser.add_argument("--output_size", type=int, default=527)
    parser.add_argument("--checkpoint_path", type=str,
                        default='/home/zhyuan/Desktop/audioset_tagging_cnn/Cnn14_mAP=0.431.pth')
    return parser.parse_args()


def convert_array_to_wav(recall_array, output_path="recalled_audio.wav", sample_rate=16000):
    waveform = recall_array.flatten()
    sf.write(output_path, waveform, sample_rate)
    return output_path


def split_audio_into_chunks(audio_path, chunk_length=1, output_dir="chunks"):
    os.makedirs(output_dir, exist_ok=True)
    y, sr = librosa.load(audio_path, sr=None)
    chunk_paths = []
    num_chunks = len(y) // (chunk_length * sr)

    for i in range(num_chunks):
        start_sample = i * chunk_length * sr
        end_sample = (i + 1) * chunk_length * sr
        chunk = y[start_sample:end_sample]
        chunk_filename = os.path.join(output_dir, f"{os.path.basename(audio_path)}_chunk_{i}.wav")
        sf.write(chunk_filename, chunk, sr)
        chunk_paths.append(chunk_filename)

    return chunk_paths



def process_audio(test_audio_path, output_file):
    # Initialize models and queues fresh for each file
    args = parse_args()

    # Create fresh queues for each file
    audio_input_queue = Queue()
    recall_queue = Queue()
    description_queue = Queue()

    # Initialize models for each file
    echoic_memory = EchoicMemory()
    pattern_detector = PatternChangeDetector(args, similarity_threshold=0.945)
    # pattern_detector = PatternChangeDetector(args, similarity_threshold=0.0)
    qwen_description = QwenAudioDescription(recall_queue, description_queue, device='cuda:0')

    if not os.path.exists(test_audio_path):
        print(f"🔴 Error: File {test_audio_path} does not exist. Exiting...")
        return

    # Split audio into 1-second chunks for analysis
    # Create a unique chunks directory for this audio file
    chunks_dir = os.path.join(os.path.dirname(test_audio_path), f"chunks_{os.path.basename(test_audio_path)}")
    os.makedirs(chunks_dir, exist_ok=True)
    chunk_paths = split_audio_into_chunks(test_audio_path, chunk_length=1, output_dir=chunks_dir)

    def extract_file_index(file_path):
        """Extract the R00XX file index from file path"""
        match = re.search(r'R\d{4}', file_path)
        if match:
            return match.group(0)
        return "Unknown"

    def extract_classification(description):
        """Extract classification (calming/tranquil, lively/active, or neither) from description"""
        # First, try to find question 3 and its answer
        # Look for patterns like:
        # "3. In general, how would you categorize the environment you just experienced? Please use one word to answer the question."
        # "Lively/active."

        # Pattern 1: Match the exact format in the example with two styles of question numbering
        q3_pattern = re.search(
            r'(?:3\.|3\)|3:)\s+In general,\s+how would you categorize the environment you just experienced\?.*?\n(.*?)(?:\n|$)',
            description, re.IGNORECASE | re.DOTALL)
        if q3_pattern:
            answer = q3_pattern.group(1).strip().lower()
            # Check for exact classifications in the direct answer
            if "lively/active" in answer:
                return "lively/active"
            elif "calming/tranquil" in answer:
                return "calming/tranquil"
            elif "neither" in answer:
                return "neither"

        # Pattern 2: Try to extract using the exact format layout you provided
        # This pattern handles the numbered list with indentation that Qwen might output
        format_pattern = re.search(
            r'\d+\.\s+What sounds do you hear\?.*?\n.*?\n\s*\d+\.\s+What type of public place.*?\n.*?\n\s*\d+\.\s+In general,.*?\n(.*?)(?:\n|$)',
            description, re.IGNORECASE | re.DOTALL)
        if format_pattern:
            answer = format_pattern.group(1).strip().lower()
            # Check for exact classifications in the direct answer
            if "lively/active" in answer:
                return "lively/active"
            elif "calming/tranquil" in answer:
                return "calming/tranquil"
            elif "neither" in answer:
                return "neither"

        # Try a more general pattern for question 3
        q3_alt_pattern = re.search(r'(?:3\.|\bthird|\bthree\b).*categorize.*environment.*?\n(.*?)(?:\n|$)', description,
                                   re.IGNORECASE | re.DOTALL)
        if q3_alt_pattern:
            answer = q3_alt_pattern.group(1).strip().lower()
            # Check for exact classifications in the direct answer
            if "lively/active" in answer or "lively" in answer:
                return "lively/active"
            elif "calming/tranquil" in answer or "tranquil" in answer or "calming" in answer:
                return "calming/tranquil"
            elif "neither" in answer:
                return "neither"

        # Try another pattern for differently formatted descriptions
        q3_alt_pattern = re.search(
            r'(?:categorize|classify).*environment.*?\s+(lively/active|calming/tranquil|neither)\.?', description,
            re.IGNORECASE)
        if q3_alt_pattern:
            return q3_alt_pattern.group(1).lower()

        # If we can't find the question format, fall back to the original keyword approach
        lower_desc = description.lower()

        # Define expanded keywords for classification
        lively_keywords = ["lively/active", "lively", "active", "car", "busy", "traffic",
                           "noisy", "loud", "bustling", "crowded", "vehicle", "engine", "train", "airport"]

        tranquil_keywords = ["calming/tranquil", "tranquil", "calming", "quiet", "peaceful",
                             "serene", "gentle", "relaxing"]

        # First check for explicit 'neither' classification
        if re.search(r'\bneither\b', lower_desc):
            return "neither"

        # Check for direct mentions using expanded keywords
        lively_count = 0
        tranquil_count = 0

        # Count occurrences of keywords from each category
        for keyword in lively_keywords:
            if re.search(rf'\b{keyword}\b', lower_desc):
                lively_count += 1

        for keyword in tranquil_keywords:
            if re.search(rf'\b{keyword}\b', lower_desc):
                tranquil_count += 1

        # Determine classification based on keyword frequency
        if lively_count > tranquil_count:
            return "lively/active"
        elif tranquil_count > lively_count:
            return "calming/tranquil"
        elif lively_count > 0 and tranquil_count > 0:
            # If both types of keywords are present but equal count, look for explicit classification
            match = re.search(r'(calming/tranquil|lively/active|neither)', lower_desc)
            if match:
                return match.group(1)
            else:
                return "neither"  # If ambiguous, default to neither
        else:
            # Try to find an explicit answer (backup)
            match = re.search(r'(calming/tranquil|lively/active|neither)', lower_desc)
            if match:
                return match.group(1)

            # Default case if we couldn't extract
            return "unclassified"

    try:
        # Get file ID
        file_id = extract_file_index(test_audio_path)
        print(f"Processing file: {file_id} ({test_audio_path})")

        # Prepare data for Excel output
        excel_data = []

        # Variables to track time sent to Qwen (in seconds)
        time_length_sent_to_Qwen = 0  # Total unique seconds sent to Qwen
        max_processed_position = 0  # Maximum position in seconds we've processed

        # Track the last pattern change classification and description
        last_pattern_change_classification = None
        last_pattern_change_description = None
        last_pattern_change_position = None

        # Process the file according to the new logic
        n_chunks = len(chunk_paths)

        if n_chunks < 15:
            # If file is shorter than 15 seconds, process the whole file
            print(f"  File is too short ({n_chunks} seconds). Processing entire file.")
            description = qwen_description.generate_description(test_audio_path)

            # Debug: Write full description to file for verification
            output_file.write(f"RAW DESCRIPTION (SHORT FILE):\n{description}\n\n")

            classification = extract_classification(description)

            # Store as the only classification (technically not a pattern change)
            last_pattern_change_classification = classification
            last_pattern_change_description = description
            last_pattern_change_position = 0

            # All time was sent to Qwen
            time_length_sent_to_Qwen = n_chunks

            # Write to output file
            output_file.write(f"File: {file_id}\n")
            output_file.write(f"Classification: {classification}\n")
            output_file.write(f"Description: {description}\n")
            output_file.write(f"Time length sent to Qwen: {time_length_sent_to_Qwen} seconds\n\n")
        else:
            # First, always process the initial 15 seconds
            initial_segment_chunks = chunk_paths[:15]

            # Combine chunks for description
            temp_dir = os.path.dirname(test_audio_path)
            initial_segment_path = os.path.join(temp_dir, f"initial_segment.wav")
            initial_segment_audio = [librosa.load(chunk_path, sr=args.sample_rate)[0] for chunk_path in
                                     initial_segment_chunks]
            initial_segment_audio = np.concatenate(initial_segment_audio)
            sf.write(initial_segment_path, initial_segment_audio, args.sample_rate)

            # Process initial 15 seconds with Qwen
            initial_description = qwen_description.generate_description(initial_segment_path)

            # Debug: Write full description to file for verification
            output_file.write(f"RAW INITIAL DESCRIPTION:\n{initial_description}\n\n")

            initial_classification = extract_classification(initial_description)

            # Set initial classification as the first "change"
            last_pattern_change_classification = initial_classification
            last_pattern_change_description = initial_description
            last_pattern_change_position = 0

            # Mark these segments as processed and count them
            time_length_sent_to_Qwen = 15

            # Write to output file
            output_file.write(f"=== INITIAL SEGMENT (0 to 15 seconds) ===\n")
            output_file.write(f"Classification: {initial_classification}\n")
            output_file.write(f"Description: {initial_description}\n")
            output_file.write(f"Time sent to Qwen: {time_length_sent_to_Qwen} seconds\n")
            output_file.write("======================================\n\n")

            # Initialize pattern detector with data from the 15th second
            pattern_detector.prev_embedding = None

            # Start pattern change detection from 15th second onwards
            change_points = []  # List to store detected change points

            # Create a 4-second window starting at the 15th second
            for chunk_index in range(15, n_chunks):
                # We need a 4-second window, so we look back 3 seconds
                if chunk_index >= 18:  # We need at least 18 seconds of data for the first 4-second window starting at 15
                    # Create 4-second window (current position - 3 to current position)
                    detection_window = chunk_paths[chunk_index - 3:chunk_index + 1]

                    # Combine the 4 chunks for pattern detection
                    detection_audio_path = os.path.join(temp_dir, f"detection_window_{chunk_index}.wav")
                    detection_audio = [librosa.load(chunk_path, sr=args.sample_rate)[0] for chunk_path in
                                       detection_window]
                    detection_audio = np.concatenate(detection_audio)
                    sf.write(detection_audio_path, detection_audio, args.sample_rate)

                    # Calculate similarity
                    if pattern_detector.prev_embedding is None:
                        _, embedding = pattern_detector.detect_pattern_change(detection_audio_path)
                        pattern_detector.prev_embedding = embedding
                    else:
                        similarity, embedding = pattern_detector.detect_pattern_change(detection_audio_path)

                        # Check if pattern change detected
                        if similarity < pattern_detector.similarity_threshold:
                            # Record the change point (current chunk index)
                            change_points.append(chunk_index)

                            # Create a 15-second segment going backward from the current change point
                            # This segment ends at the current chunk_index and goes back 14 seconds
                            # segment_end = chunk_index + 1  # +1 because slice end is exclusive
                            # segment_start = max(0, segment_end - 15)  # Go back 15 seconds (including current second)

                            segment_end = chunk_index + 14  # +1 because slice end is exclusive
                            segment_end = min(n_chunks, segment_end)
                            segment_start = max(0, segment_end - 15)  # Go forward 15 seconds (including current second)

                            # Get the chunks for this segment
                            segment_chunks = chunk_paths[segment_start:segment_end]

                            # Combine chunks for description
                            segment_audio_path = os.path.join(temp_dir, f"segment_at_{chunk_index}.wav")
                            segment_audio = [librosa.load(chunk_path, sr=args.sample_rate)[0] for chunk_path in
                                             segment_chunks]
                            segment_audio = np.concatenate(segment_audio)
                            sf.write(segment_audio_path, segment_audio, args.sample_rate)

                            # Generate description using Qwen
                            description = qwen_description.generate_description(segment_audio_path)

                            # Debug: Write full description to file for verification
                            output_file.write(f"RAW DESCRIPTION:\n{description}\n\n")

                            classification = extract_classification(description)

                            # Update the last pattern change classification
                            last_pattern_change_classification = classification
                            last_pattern_change_description = description
                            last_pattern_change_position = chunk_index

                            # Calculate newly processed seconds based on the example logic
                            # time_length_sent_to_Qwen is a cumulative counter of all unique seconds sent to Qwen
                            # We need to check how many new seconds this segment adds

                            # Track maximum position we've processed so far
                            max_processed_position = time_length_sent_to_Qwen  # This is after initial 15 seconds

                            # Count only seconds we haven't processed before
                            new_seconds = 0
                            for pos in range(segment_start, segment_end):
                                if pos >= max_processed_position:
                                    new_seconds += 1

                            # Update total time sent to Qwen
                            time_length_sent_to_Qwen += new_seconds

                            # Update maximum processed position if this segment extends it
                            if segment_end > max_processed_position:
                                max_processed_position = segment_end

                            # Write to output file
                            output_file.write(f"=== PATTERN CHANGE AT {chunk_index} SECONDS ===\n")
                            output_file.write(f"Processing segment from {segment_start} to {segment_end} seconds\n")
                            output_file.write(f"New seconds sent to Qwen: {new_seconds}\n")
                            output_file.write(f"Classification: {classification}\n")
                            output_file.write(f"Description: {description}\n")
                            output_file.write("======================================\n\n")

                        # Update embedding
                        pattern_detector.prev_embedding = embedding

            print(change_points)

        # Write to output file
        output_file.write(f"\n=== FINAL RESULTS ===\n")
        output_file.write(f"Last Pattern Change Position: {last_pattern_change_position} seconds\n")
        output_file.write(f"Last Pattern Change Classification: {last_pattern_change_classification}\n")
        output_file.write(f"Last Pattern Change Description: {last_pattern_change_description}\n")
        output_file.write(f"Total time length sent to Qwen: {time_length_sent_to_Qwen} seconds\n")
        output_file.write(f"Total file length: {n_chunks} seconds\n")
        output_file.write("======================================\n")

        # Add to Excel data
        excel_data.append({
            'File_Index': file_id,
            'Last_Pattern_Change_Classification': last_pattern_change_classification,
            'Last_Pattern_Change_Description': last_pattern_change_description,
            'Last_Pattern_Change_Position': last_pattern_change_position,
            'Num_Change_Points': len(change_points) if n_chunks >= 15 else 0,
            'Time_Length_Sent_To_Qwen': time_length_sent_to_Qwen,
            'Total_File_Length': n_chunks
        })

        # Write data to Excel file
        df = pd.DataFrame(excel_data)
        excel_path = os.path.join('/home/zhyuan/Desktop/Qwen-Audio/results/excel', f"{file_id}_classification.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"Results written to {excel_path}")

        # Return data for potential aggregation
        return {
            'File_Index': file_id,
            'Last_Pattern_Change_Classification': last_pattern_change_classification,
            'Last_Pattern_Change_Description': last_pattern_change_description,
            'Last_Pattern_Change_Position': last_pattern_change_position,
            'Num_Change_Points': len(change_points) if n_chunks >= 15 else 0,
            'Time_Length_Sent_To_Qwen': time_length_sent_to_Qwen,
            'Total_File_Length': n_chunks
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        traceback.print_exc()

        # Try to save any data collected so far
        if 'excel_data' in locals() and len(excel_data) > 0:
            try:
                df = pd.DataFrame(excel_data)
                df.to_excel(f"{file_id}_classification_partial.xlsx", index=False)
                print(f"Partial results saved to {file_id}_classification_partial.xlsx")
            except:
                print("Could not save partial results")

    finally:
        # Clean up resources
        try:
            echoic_memory.stop()
        except:
            pass

        try:
            pattern_detector.stop()
        except:
            pass

        try:
            qwen_description.stop()
        except:
            pass

        # Clean up temporary files - but only after successful execution
        # Only clean up if no error occurred (to preserve files for debugging)
        if 'e' not in locals():
            temp_dir = os.path.dirname(test_audio_path)
            for temp_file in ["combined_4s_window.wav", "recalled_audio.wav", "initial_segment.wav"]:
                full_path = os.path.join(temp_dir, temp_file)
                if os.path.exists(full_path):
                    try:
                        os.remove(full_path)
                    except:
                        pass

            # Clean up detection window files
            for i in range(n_chunks):
                window_file = os.path.join(temp_dir, f"detection_window_{i}.wav")
                if os.path.exists(window_file):
                    try:
                        os.remove(window_file)
                    except:
                        pass

            # Clean up segment files
            for i in range(n_chunks):
                segment_file = os.path.join(temp_dir, f"segment_at_{i}.wav")
                if os.path.exists(segment_file):
                    try:
                        os.remove(segment_file)
                    except:
                        pass

            # Clean up chunks directory
            try:
                if 'chunks_dir' in locals() and os.path.exists(chunks_dir):
                    for file in os.listdir(chunks_dir):
                        os.remove(os.path.join(chunks_dir, file))
                    os.rmdir(chunks_dir)
            except:
                pass

        # Force Python garbage collection
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Explicitly delete model objects
        del echoic_memory
        del pattern_detector
        del qwen_description
        del audio_input_queue
        del recall_queue
        del description_queue

        # Force garbage collection again
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



"For all samples"
# def main():
#     dataset_path = "/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio"
#     results_path = "/home/zhyuan/Desktop/Qwen-Audio/results/TXT"
#     excel_results_path = "/home/zhyuan/Desktop/Qwen-Audio/results/excel"
#
#     # Create results directories if they don't exist
#     os.makedirs(results_path, exist_ok=True)
#     os.makedirs(excel_results_path, exist_ok=True)
#
#     # List to collect all file results
#     all_file_data = []
#
#     for file_name in os.listdir(dataset_path):
#         if file_name.endswith(".wav"):
#             file_path = os.path.join(dataset_path, file_name)
#             txt_file_name = file_name.replace(".wav", ".txt")
#             txt_file_path = os.path.join(results_path, txt_file_name)
#
#             print(f"Processing {file_name}...")
#
#             with open(txt_file_path, 'w') as output_file:
#                 # Process the audio and get data for this file
#                 file_data = process_audio(file_path, output_file)
#
#                 # If valid data was returned, add it to our collection
#                 if file_data:
#                     all_file_data.append(file_data)
#
#             print(f"Output for {file_name} saved to {txt_file_path}")
#
#             # Additional cleanup between files
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#
#     # After all files are processed, create a single Excel file with all results
#     if all_file_data:
#         print(f"Creating combined Excel file with {len(all_file_data)} entries...")
#         try:
#             df = pd.DataFrame(all_file_data)
#             excel_path = os.path.join(excel_results_path, "audio_classifications_threshold1.0_1to133.xlsx")
#             df.to_excel(excel_path, index=False)
#             print(f"All results written to {excel_path}")
#         except Exception as e:
#             print(f"Error creating combined Excel file: {str(e)}")
#             traceback.print_exc()
#     else:
#         print("No valid data collected, skipping combined Excel file creation")



"For single sample test"
def main():
    # Specify the single audio file to process.
    # audio_file = {'audio': '/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio/R0031_segment_ambisonics.wav'}
    audio_file = {'audio': '/home/zhyuan/Desktop/PCD/data/LU_AVS/audio_dataset/0AFOq7PKDDo.wav'}

    results_path = "/home/zhyuan/Desktop/Qwen-Audio/results/TXT"
    excel_results_path = "/home/zhyuan/Desktop/Qwen-Audio/results/excel"

    # Create results directories if they don't exist
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(excel_results_path, exist_ok=True)

    # Extract file path and name
    file_path = audio_file['audio']
    file_name = os.path.basename(file_path)

    # Create output file names
    txt_file_name = file_name.replace(".wav", ".txt")
    txt_file_path = os.path.join(results_path, txt_file_name)

    print(f"Processing {file_name}...")

    # List to collect file results (keeping the list structure for Excel compatibility)
    all_file_data = []

    with open(txt_file_path, 'w') as output_file:
        # Process the audio and get data for this file
        file_data = process_audio(file_path, output_file)

        # If valid data was returned, add it to our collection
        if file_data:
            all_file_data.append(file_data)

    print(f"Output for {file_name} saved to {txt_file_path}")

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create Excel file with the single file result
    if all_file_data:
        print(f"Creating Excel file with the processed result...")
        try:
            df = pd.DataFrame(all_file_data)
            excel_path = os.path.join(excel_results_path,
                                      f"{file_name.replace('.wav', '')}_classification_threshold1.0.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"Result written to {excel_path}")
        except Exception as e:
            print(f"Error creating Excel file: {str(e)}")
            traceback.print_exc()
    else:
        print("No valid data collected, skipping Excel file creation")


"Do not need anymore"
# def main():
#     dataset_path = "/home/zhyuan/Desktop/Qwen-Audio/W_components"
#     results_path = "/home/zhyuan/Desktop/Qwen-Audio/results"
#
#     # Create results directory if it doesn't exist
#     os.makedirs(results_path, exist_ok=True)
#
#     # Clean up chunk directory before starting
#     chunks_dir = "chunks"
#     if os.path.exists(chunks_dir):
#         for file in os.listdir(chunks_dir):
#             file_path = os.path.join(chunks_dir, file)
#             try:
#                 if os.path.isfile(file_path):
#                     os.remove(file_path)
#             except Exception as e:
#                 print(f"Error removing {file_path}: {e}")
#
#     for file_name in os.listdir(dataset_path):
#         if file_name.endswith(".wav"):
#             file_path = os.path.join(dataset_path, file_name)
#             txt_file_name = file_name.replace(".wav", ".txt")
#             txt_file_path = os.path.join(results_path, txt_file_name)
#
#             print(f"Processing {file_name}...")
#
#             with open(txt_file_path, 'w') as output_file:
#                 process_audio(file_path, output_file)
#
#             print(f"Output for {file_name} saved to {txt_file_path}")
#
#             # Additional cleanup between files
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#
#
# def main():
#     dataset_path = "/home/zhyuan/Desktop/Qwen-Audio/W_components"
#     results_path = "/home/zhyuan/Desktop/Qwen-Audio/results/TXT"
#
#     # Create results directory if it doesn't exist
#     os.makedirs(results_path, exist_ok=True)
#
#     # Clean up chunk directory before starting
#     chunks_dir = "chunks"
#     if os.path.exists(chunks_dir):
#         for file in os.listdir(chunks_dir):
#             file_path = os.path.join(chunks_dir, file)
#             try:
#                 if os.path.isfile(file_path):
#                     os.remove(file_path)
#             except Exception as e:
#                 print(f"Error removing {file_path}: {e}")
#
#     # List to collect all file results
#     all_file_data = []
#
#     for file_name in os.listdir(dataset_path):
#         if file_name.endswith(".wav"):
#             file_path = os.path.join(dataset_path, file_name)
#             txt_file_name = file_name.replace(".wav", ".txt")
#             txt_file_path = os.path.join(results_path, txt_file_name)
#
#             print(f"Processing {file_name}...")
#
#             with open(txt_file_path, 'w') as output_file:
#                 # Process the audio and get data for this file
#                 file_data = process_audio(file_path, output_file)
#
#                 # If valid data was returned, add it to our collection
#                 if file_data:
#                     all_file_data.append(file_data)
#
#             print(f"Output for {file_name} saved to {txt_file_path}")
#
#             # Additional cleanup between files
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#
#     # After all files are processed, create a single Excel file with all results
#     if all_file_data:
#         print(f"Creating Excel file with {len(all_file_data)} entries...")
#         try:
#             df = pd.DataFrame(all_file_data)
#             excel_path = os.path.join(results_path, "audio_classifications.xlsx")
#             df.to_excel(excel_path, index=False)
#             print(f"Results written to {excel_path}")
#         except Exception as e:
#             print(f"Error creating Excel file: {str(e)}")
#             traceback.print_exc()
#     else:
#         print("No valid data collected, skipping Excel file creation")



if __name__ == "__main__":
    main()
