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
from datetime import datetime  # Add this import

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


def extract_file_index(file_path):
    """Extract the R00XX file index from file path"""
    match = re.search(r'R\d{4}', file_path)
    if match:
        return match.group(0)
    return "Unknown"


def process_audio(test_audio_path):
    # Initialize models and queues fresh for each file
    args = parse_args()

    # Create fresh queues for each file
    audio_input_queue = Queue()
    recall_queue = Queue()
    description_queue = Queue()

    # Initialize models for each file
    echoic_memory = EchoicMemory()
    pattern_detector = PatternChangeDetector(args, similarity_threshold=0.945)
    qwen_description = QwenAudioDescription(recall_queue, description_queue, device='cuda:0')

    # Initialize variables that will be used in finally block
    n_chunks = 0
    chunks_dir = None
    temp_dir = None

    if not os.path.exists(test_audio_path):
        print(f"🔴 Error: File {test_audio_path} does not exist. Exiting...")
        return None

    # Split audio into 1-second chunks for analysis
    chunks_dir = os.path.join(os.path.dirname(test_audio_path), f"chunks_{os.path.basename(test_audio_path)}")
    os.makedirs(chunks_dir, exist_ok=True)
    chunk_paths = split_audio_into_chunks(test_audio_path, chunk_length=1, output_dir=chunks_dir)

    try:
        # Get file ID
        file_id = extract_file_index(test_audio_path)
        print(f"Processing file: {file_id} ({test_audio_path})")

        # Initialize result structure
        result = {
            "file_id": file_id,
            "audio_path": test_audio_path,
            "total_file_length": len(chunk_paths),
            "processing_date": datetime.now().isoformat(),
            "initial_segment": {},
            "pattern_changes": [],
            "change_points": [],
            "time_length_sent_to_qwen": 0
        }

        n_chunks = len(chunk_paths)
        temp_dir = os.path.dirname(test_audio_path)
        time_length_sent_to_qwen = 0

        if n_chunks < 15:
            # If file is shorter than 15 seconds, process the whole file
            print(f"  File is too short ({n_chunks} seconds). Processing entire file.")
            description = qwen_description.generate_description(test_audio_path)

            result["initial_segment"] = {
                "description": description,
                "timestamp": 0,
                "length": n_chunks
            }
            result["time_length_sent_to_qwen"] = n_chunks

        else:
            # First, always process the initial 15 seconds
            initial_segment_chunks = chunk_paths[:15]

            # Combine chunks for description
            initial_segment_path = os.path.join(temp_dir, f"initial_segment.wav")
            initial_segment_audio = [librosa.load(chunk_path, sr=args.sample_rate)[0] for chunk_path in
                                     initial_segment_chunks]
            initial_segment_audio = np.concatenate(initial_segment_audio)
            sf.write(initial_segment_path, initial_segment_audio, args.sample_rate)

            # Process initial 15 seconds with Qwen
            initial_description = qwen_description.generate_description(initial_segment_path)

            result["initial_segment"] = {
                "description": initial_description,
                "timestamp": 0,
                "length": 15
            }

            time_length_sent_to_qwen = 15
            max_processed_position = 15

            # Initialize pattern detector
            pattern_detector.prev_embedding = None

            # Start pattern change detection from 15th second onwards
            change_points = []

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
                            print(f"Pattern change detected! Cosine similarity: {similarity:.2f}")

                            # Record the change point
                            change_points.append(chunk_index)
                            result["change_points"].append(chunk_index)

                            # Create a 15-second segment going forward from the current change point
                            segment_end = chunk_index + 14
                            segment_end = min(n_chunks, segment_end)
                            segment_start = max(0, segment_end - 15)

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
                            print(f"Generated Description: {description}")

                            # Calculate newly processed seconds
                            new_seconds = 0
                            for pos in range(segment_start, segment_end):
                                if pos >= max_processed_position:
                                    new_seconds += 1

                            # Update total time sent to Qwen
                            time_length_sent_to_qwen += new_seconds

                            # Update maximum processed position if this segment extends it
                            if segment_end > max_processed_position:
                                max_processed_position = segment_end

                            # Store pattern change information
                            pattern_change = {
                                "timestamp": chunk_index,
                                "cosine_similarity": similarity,
                                "description": description,
                                "segment_start": segment_start,
                                "segment_end": segment_end,
                                "new_seconds_processed": new_seconds
                            }
                            result["pattern_changes"].append(pattern_change)

                        # Update embedding
                        pattern_detector.prev_embedding = embedding

            print(f"Change points: {change_points}")
            result["time_length_sent_to_qwen"] = time_length_sent_to_qwen

        return result

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        traceback.print_exc()
        return None

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

        # Clean up temporary files
        if temp_dir:  # Only clean up if temp_dir was set
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
            if chunks_dir and os.path.exists(chunks_dir):
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


def main():
    dataset_path = "/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio"
    results_path = "/home/zhyuan/Desktop/Qwen-Audio/results"

    # Create results directory if it doesn't exist
    os.makedirs(results_path, exist_ok=True)

    # List to collect all file results
    all_results = []

    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(dataset_path, file_name)

            print(f"Processing {file_name}...")

            # Process the audio and get data for this file
            file_data = process_audio(file_path)

            # If valid data was returned, add it to our collection
            if file_data:
                all_results.append(file_data)

            print(f"Finished processing {file_name}")

            # Additional cleanup between files
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save all results to JSON file
    if all_results:
        json_output = {
            "dataset_info": {
                "dataset_path": dataset_path,
                "total_files_processed": len(all_results),
                "processing_date": datetime.now().isoformat(),
                "similarity_threshold": 0.945
            },
            "results": all_results
        }

        json_path = os.path.join(results_path, "pattern_change_results.json")

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False)
            print(f"All results saved to {json_path}")

            # Print summary
            total_changes = sum(len(result["pattern_changes"]) for result in all_results)
            print(f"Summary: Processed {len(all_results)} files with {total_changes} total pattern changes detected")

        except Exception as e:
            print(f"Error saving JSON file: {str(e)}")
            traceback.print_exc()
    else:
        print("No valid data collected, skipping JSON file creation")


if __name__ == "__main__":
    main()
