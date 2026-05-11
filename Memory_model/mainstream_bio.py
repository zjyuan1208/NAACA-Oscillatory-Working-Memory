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
import pandas as pd
import re
import json
import traceback
from datetime import datetime


import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from echoic_memory import EchoicMemory
from biodetector import BioOSSPatternChangeDetector
from Qwen_processor import QwenAudioDescription

blues = ['#115699', '#0E6DB3', '#5CAAD7', '#95C6DE']
reds = ['#8E0D29', '#BB1E38', '#D35B4D', '#F6BCA9']
yellows = ['#E19D49', '#E8B547', '#EFC99B']
greens = ['#365C3B', '#4A6C4C', '#7DA47C', '#ABD0A7']

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BioOSS Multi-Metric Audio Change Detection')

    # Basic audio processing arguments
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--device", type=str, default="cuda:1",
                        help="Device for detector (cpu/cuda/auto). Auto selects cuda if available")
    parser.add_argument("--sample_rate", type=int, default=32000, help="Audio sample rate")
    parser.add_argument("--window_size", type=int, default=1024, help="STFT window size")
    parser.add_argument("--hop_size", type=int, default=320, help="STFT hop size")
    parser.add_argument("--mel_bins", type=int, default=64, help="Number of mel frequency bins")
    parser.add_argument("--fmin", type=int, default=0, help="Minimum frequency")
    parser.add_argument("--fmax", type=int, default=14000, help="Maximum frequency")
    parser.add_argument("--output_size", type=int, default=527, help="PANN output dimension")
    parser.add_argument("--checkpoint_path", type=str,
                        default='/home/zhyuan/Desktop/audioset_tagging_cnn/Cnn14_mAP=0.431.pth',
                        help="Path to PANN checkpoint")

    # Multi-metric detection parameters
    # parser.add_argument('--energy_weight', type=float, default=0.4,
    #                     help='Weight for energy metric in multi-metric detection')
    # parser.add_argument('--pca_weight', type=float, default=0.35,
    #                     help='Weight for PCA metric in multi-metric detection')
    # parser.add_argument('--cosine_weight', type=float, default=0.25,
    #                     help='Weight for cosine metric in multi-metric detection')
    parser.add_argument('--energy_weight', type=float, default=0.4,
                        help='Weight for energy metric')
    parser.add_argument('--pca_weight', type=float, default=0.0,
                        help='Weight for PCA metric')
    parser.add_argument('--cosine_weight', type=float, default=0.2,
                        help='Weight for cosine metric')
    parser.add_argument('--consensus_threshold', type=float, default=0.35,
                        help='Consensus threshold for change detection')
    parser.add_argument('--min_persistence', type=int, default=2,
                        help='Minimum persistence duration for detection')
    parser.add_argument('--min_confidence_threshold', type=float, default=0.3)

    # BioOSS FDTD parameters
    parser.add_argument('--freq_min', type=float, default=50.0,
                        help='Minimum frequency for BioOSS initialization')
    parser.add_argument('--freq_max', type=float, default=1200.0,
                        help='Maximum frequency for BioOSS initialization')
    parser.add_argument('--grid_size', type=int, default=64,
                        help='BioOSS grid size')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='BioOSS time step')

    # Processing parameters
    parser.add_argument('--window_length', type=float, default=4.0,
                        help='Sliding window length in seconds for change detection')
    parser.add_argument('--stride_length', type=float, default=1.0,
                        help='Stride length in seconds for change detection')

    # Debug and output options
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug output for BioOSS detection')
    parser.add_argument('--verbose_metrics', action='store_true', default=False,
                        help='Print detailed metrics for each timestep')

    return parser.parse_args()


def get_device(args):
    """Determine the device to use for processing"""
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = "cpu"

    print(f"Using device: {device}")
    return device


def create_sliding_windows_realtime(audio, sr, window_length, stride_length):
    """Create sliding windows similar to the batch scripts for consistency"""
    window_samples = int(window_length * sr)
    stride_samples = int(stride_length * sr)

    windows = []
    for start in range(0, len(audio) - window_samples + 1, stride_samples):
        end = start + window_samples
        windows.append(audio[start:end])

    return windows


def load_audio_file_realtime(audio_path, sample_rate):
    """Load audio file similar to batch scripts"""
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    return audio, sr


def extract_file_index(file_path):
    """Extract the R00XX file index from file path"""
    match = re.search(r'R\d{4}', file_path)
    return match.group(0) if match else "Unknown"


def calculate_time_sent_to_qwen(change_points, total_length):
    """
    Calculate the total time sent to Qwen with proper overlap handling

    Args:
        change_points: List of timestamps where changes were detected
        total_length: Total length of the audio file in seconds (can be float)

    Returns:
        total_time: Total unique seconds sent to Qwen
    """
    # Convert total_length to integer for range operations
    total_length_int = int(total_length)

    # Always process initial 15 seconds
    processed_intervals = [(0, 15)]
    total_time = 15

    for change_point in change_points:
        # For each change point, we process a 15-second attention window
        # that includes 4s echoic memory (t-3 to t+1) + 11s forward (t+1 to t+12)
        # This captures the transition from old pattern to new pattern
        segment_start = max(0, int(change_point - 3))  # Include 4s echoic memory (t-3 to t+1)
        segment_end = min(int(change_point + 12), total_length_int)  # 12s forward from change point

        # Ensure we always have 15 seconds if possible
        if segment_end - segment_start < 15 and segment_start > 0:
            segment_start = max(0, segment_end - 15)
        elif segment_end - segment_start < 15 and segment_end < total_length_int:
            segment_end = min(segment_start + 15, total_length_int)

        # Calculate overlap with existing processed intervals
        new_seconds = 0
        segment_range = set(range(segment_start, segment_end))

        # Remove already processed seconds
        for start, end in processed_intervals:
            processed_range = set(range(int(start), int(end)))
            segment_range -= processed_range

        new_seconds = len(segment_range)

        if new_seconds > 0:
            # Add new interval to processed intervals
            processed_intervals.append((segment_start, segment_end))
            total_time += new_seconds

    return total_time


def process_short_audio(test_audio_path, qwen_description, duration):
    """Process audio files shorter than 15 seconds"""
    print(f"  File is too short ({duration:.1f} seconds). Processing entire file.")

    description = qwen_description.generate_description(test_audio_path)

    return {
               "description": description,
               "timestamp": 0,
               "length": duration
           }, duration  # Return the actual time sent to Qwen


def process_initial_segment_realtime(audio, sr, qwen_description, pattern_detector, args):
    """Process the initial 15-second segment and initialize detector with baseline data"""
    # Create initial 15-second segment
    initial_duration = 15.0
    initial_samples = int(initial_duration * sr)
    initial_audio = audio[:initial_samples]

    # Save initial segment for Qwen processing
    temp_dir = "/tmp"
    initial_segment_path = os.path.join(temp_dir, "initial_segment_realtime.wav")
    sf.write(initial_segment_path, initial_audio, sr)

    # Process with Qwen
    initial_description = qwen_description.generate_description(initial_segment_path)

    # IMPORTANT: Run pattern detection on initial segment too (like batch scripts)
    print("Running pattern detection on initial 15 seconds for baseline establishment...")

    # Create windows exactly like batch scripts do
    window_length_samples = int(args.window_length * sr)  # 4.0 * 32000 = 128000 samples
    stride_length_samples = int(args.stride_length * sr)  # 1.0 * 32000 = 32000 samples

    # Generate baseline windows (matching batch script logic)
    baseline_windows = []
    for start_sample in range(0, len(initial_audio) - window_length_samples + 1, stride_length_samples):
        end_sample = start_sample + window_length_samples
        if end_sample <= len(initial_audio):  # Ensure we don't exceed 15 seconds
            baseline_windows.append(initial_audio[start_sample:end_sample])

    # Process each baseline window WITH DETECTION (matching batch script logic exactly)
    baseline_changes = []
    for i, window in enumerate(baseline_windows):
        # Save window temporarily
        window_path = os.path.join(temp_dir, f"baseline_window_realtime_{i}.wav")
        sf.write(window_path, window, sr)

        # CRITICAL: Run full detection process (not just for baseline stats)
        # This matches exactly what batch scripts do - they run detection from timestep 0
        similarity, baseline_result = pattern_detector.detect_pattern_change(window_path)

        # Check for changes during baseline period (like batch scripts do)
        if baseline_result and baseline_result.get('change_detected', False):
            baseline_changes.append(i)
            if args.debug:
                print(
                    f"  Baseline change detected at window {i} (time: {i * args.stride_length:.1f}s) with confidence {baseline_result.get('confidence', 0):.4f}")

        if args.debug and baseline_result:
            detailed_metrics = baseline_result.get('detailed_metrics', {})
            if detailed_metrics:
                raw_metrics = detailed_metrics.get('raw_metrics', {})
                thresholds = detailed_metrics.get('thresholds', {})
                print(
                    f"  Baseline window {i}: Energy={raw_metrics.get('energy', 0):.4f} (threshold: {thresholds.get('energy', 0):.4f}), Change: {baseline_result.get('change_detected', False)}")

        # Clean up temporary file
        try:
            os.remove(window_path)
        except:
            pass

    if baseline_changes:
        print(f"Detected {len(baseline_changes)} changes during baseline period: {baseline_changes}")

    print("BioOSS detector baseline establishment complete with full detection processing.")

    # Clean up initial segment file
    try:
        os.remove(initial_segment_path)
    except:
        pass

    return {
        "description": initial_description,
        "timestamp": 0,
        "length": 15,
        "baseline_changes": baseline_changes  # Return any changes detected during baseline
    }


def detect_pattern_changes_realtime(audio, sr, pattern_detector, args, start_time=0.0):
    """Detect pattern changes using BioOSS multi-metric system with real-time sliding windows"""
    change_points = []
    temp_dir = "/tmp"

    # Calculate total duration and create sliding windows exactly like batch scripts
    total_duration = len(audio) / sr

    # Create windows exactly like batch scripts: 4.0s window, 1.0s stride
    window_length_samples = int(args.window_length * sr)  # 4.0 * 32000 = 128000 samples
    stride_length_samples = int(args.stride_length * sr)  # 1.0 * 32000 = 32000 samples

    # Generate windows exactly like create_sliding_windows in batch scripts
    audio_windows = []
    for start_sample in range(0, len(audio) - window_length_samples + 1, stride_length_samples):
        end_sample = start_sample + window_length_samples
        audio_windows.append(audio[start_sample:end_sample])

    # CRITICAL: Start from window 0, not window 15!
    # The batch scripts process ALL timesteps from 0, we should too
    start_window_idx = int(start_time / args.stride_length)  # Should be 0, not 15!

    print(f"Starting change detection from window {start_window_idx} (time: {start_time}s)")
    print(f"Total windows: {len(audio_windows)}, Total duration: {total_duration:.1f}s")

    # Process ALL windows starting from 0 (exactly like batch scripts)
    for window_idx in range(start_window_idx, len(audio_windows)):
        # Calculate timestamp exactly like batch scripts
        # In batch scripts: timestep t corresponds to t * stride_length seconds
        timestamp_seconds = window_idx * args.stride_length

        # Skip if we don't have enough previous windows for 4s window (first few windows)
        # Batch scripts only start meaningful detection after they have enough context
        if window_idx < 3:  # Need at least 4 windows (0,1,2,3) for first 4s detection window
            continue

        # Get current window
        current_window = audio_windows[window_idx]

        # Save window temporarily
        window_path = os.path.join(temp_dir, f"detection_window_realtime_{window_idx}.wav")
        sf.write(window_path, current_window, sr)

        # Run BioOSS detection (exactly like batch scripts call)
        biooss_confidence, biooss_result = pattern_detector.detect_pattern_change(window_path)

        # Check if change detected (exactly like batch scripts)
        if biooss_result and biooss_result.get('change_detected', False):
            # Use window_idx as the timestamp (like batch scripts use timestep t)
            change_timestamp = window_idx  # This should match batch script timesteps!
            change_points.append(change_timestamp)

            if args.debug:
                print(
                    f"Change detected at window {window_idx} (time: {timestamp_seconds:.1f}s) with confidence {biooss_result.get('confidence', 0):.4f}")
                detailed_metrics = biooss_result.get('detailed_metrics', {})
                if detailed_metrics:
                    raw_metrics = detailed_metrics.get('raw_metrics', {})
                    thresholds = detailed_metrics.get('thresholds', {})
                    print(
                        f"  Energy: {raw_metrics.get('energy', 'N/A'):.4f} (threshold: {thresholds.get('energy', 'N/A'):.4f})")
                    print(f"  PCA: {raw_metrics.get('pca', 'N/A'):.4f} (threshold: {thresholds.get('pca', 'N/A'):.4f})")
                    print(
                        f"  Cosine: {raw_metrics.get('cosine', 'N/A'):.4f} (threshold: {thresholds.get('cosine', 'N/A'):.4f})")
                    print(f"  Consensus: {detailed_metrics.get('consensus_score', 'N/A'):.4f}")

        elif args.verbose_metrics:
            # Log all detection attempts for debugging
            detailed_metrics = biooss_result.get('detailed_metrics', {}) if biooss_result else {}
            if detailed_metrics:
                raw_metrics = detailed_metrics.get('raw_metrics', {})
                thresholds = detailed_metrics.get('thresholds', {})
                consensus = detailed_metrics.get('consensus_score', 0)
                print(
                    f"No change at window {window_idx} (time: {timestamp_seconds:.1f}s): E={raw_metrics.get('energy', 0):.3f}({thresholds.get('energy', 0):.3f}), "
                    f"P={raw_metrics.get('pca', 0):.3f}({thresholds.get('pca', 0):.3f}), "
                    f"C={raw_metrics.get('cosine', 0):.3f}({thresholds.get('cosine', 0):.3f}), "
                    f"Consensus={consensus:.3f}")

        # Clean up temporary file
        try:
            os.remove(window_path)
        except:
            pass

    return change_points


def process_change_point_realtime(timestamp, audio, sr, qwen_description, args, pattern_detector):
    """Process a detected change point and generate description"""
    total_duration = len(audio) / sr
    temp_dir = "/tmp"

    # Create 15-second attention window that includes 4s echoic memory
    # Echoic memory: t-3 to t+1 (4 seconds including the detection window)
    # Forward attention: t+1 to t+12 (11 seconds after change point)
    # Total: 15 seconds covering the transition
    segment_start = max(0.0, float(timestamp - 3))  # Include 4s echoic memory (t-3 to t+1)
    segment_end = min(float(timestamp + 12), total_duration)  # 12s forward from change point

    # Ensure we always have 15 seconds if possible
    if segment_end - segment_start < 15 and segment_start > 0:
        segment_start = max(0.0, segment_end - 15)
    elif segment_end - segment_start < 15 and segment_end < total_duration:
        segment_end = min(segment_start + 15, total_duration)

    # Extract audio segment
    start_samples = int(segment_start * sr)
    end_samples = int(segment_end * sr)
    segment_audio = audio[start_samples:end_samples]

    # Save segment for processing
    segment_audio_path = os.path.join(temp_dir, f"segment_at_realtime_{timestamp}.wav")
    sf.write(segment_audio_path, segment_audio, sr)

    # Generate description
    description = qwen_description.generate_description(segment_audio_path)

    # Get BioOSS detection details for this segment
    _, biooss_result = pattern_detector.detect_pattern_change(segment_audio_path)

    # Calculate new seconds processed for this segment
    new_seconds_processed = int(segment_end - segment_start)

    # Prepare pattern change data (matching expected format)
    pattern_change = {
        "timestamp": int(timestamp),
        "description": description,
        "segment_start": segment_start,
        "segment_end": segment_end,
        "new_seconds_processed": new_seconds_processed,
        "echoic_memory_start": max(0.0, float(timestamp - 3)),  # t-3
        "echoic_memory_end": min(float(timestamp + 1), total_duration),  # t+1 (4s window)
        "change_point": int(timestamp),
        "forward_attention_end": min(float(timestamp + 12), total_duration),
        "biooss_metrics": {
            "energy": biooss_result.get('energy', 0) if biooss_result else 0,
            "detection_confidence": biooss_result.get('confidence', 0) if biooss_result else 0
        }
    }

    # Add detailed metrics if available
    if biooss_result:
        detailed_metrics = biooss_result.get('detailed_metrics', {})
        if detailed_metrics:
            raw_metrics = detailed_metrics.get('raw_metrics', {})
            thresholds = detailed_metrics.get('thresholds', {})
            pattern_change["biooss_metrics"].update({
                "energy_metric": raw_metrics.get('energy', 0),
                "pca_metric": raw_metrics.get('pca', 0),
                "cosine_metric": raw_metrics.get('cosine', 0),
                "energy_threshold": thresholds.get('energy', 0),
                "pca_threshold": thresholds.get('pca', 0),
                "cosine_threshold": thresholds.get('cosine', 0),
                "consensus_score": detailed_metrics.get('consensus_score', 0),
                "persistence_ratio": detailed_metrics.get('persistence_ratio', 0)
            })

    # Clean up temporary file
    try:
        os.remove(segment_audio_path)
    except:
        pass

    return pattern_change


def process_audio(test_audio_path):
    """Main audio processing function with real-time sliding window approach matching batch scripts"""
    args = parse_args()
    device = get_device(args)

    # Initialize components
    audio_input_queue = Queue()
    recall_queue = Queue()
    description_queue = Queue()

    echoic_memory = EchoicMemory()

    # Initialize BioOSS detector with device support
    pattern_detector = BioOSSPatternChangeDetector(
        args,
        energy_weight=args.energy_weight,
        pca_weight=args.pca_weight,
        cosine_weight=args.cosine_weight,
        consensus_threshold=args.consensus_threshold,
        min_persistence=args.min_persistence,
        min_confidence_threshold=args.min_confidence_threshold
    )

    # Move detector to specified device
    if hasattr(pattern_detector, 'biooss_model'):
        pattern_detector.biooss_model.to(device)
    if hasattr(pattern_detector, 'audio_encoder'):
        pattern_detector.audio_encoder.model.to(device)
        pattern_detector.audio_encoder.device = device

    qwen_description = QwenAudioDescription(recall_queue, description_queue, device='cuda:1')

    if not os.path.exists(test_audio_path):
        print(f"🔴 Error: File {test_audio_path} does not exist. Exiting...")
        return None

    try:
        file_id = extract_file_index(test_audio_path)
        print(f"Processing file: {file_id} ({test_audio_path})")

        # Load entire audio file (but process it in sliding windows)
        audio, sr = load_audio_file_realtime(test_audio_path, args.sample_rate)
        total_duration = len(audio) / sr

        # Initialize result structure (matching the reference format)
        result = {
            "file_id": file_id,
            "audio_path": test_audio_path,
            "total_file_length": total_duration,
            "processing_date": datetime.now().isoformat(),
            "detection_method": "BioOSS_Multi_Metric_FDTD_Realtime",
            "detection_parameters": {
                "energy_weight": args.energy_weight,
                "pca_weight": args.pca_weight,
                "cosine_weight": args.cosine_weight,
                "consensus_threshold": args.consensus_threshold,
                "min_persistence": args.min_persistence,
                "window_length": args.window_length,
                "stride_length": args.stride_length,
                "device": device
            },
            "initial_segment": {},
            "pattern_changes": [],
            "change_points": [],
            "time_length_sent_to_qwen": 0
        }

        if total_duration < 15:
            # Process short audio files
            initial_segment, time_sent = process_short_audio(test_audio_path, qwen_description, total_duration)
            result["initial_segment"] = initial_segment
            result["time_length_sent_to_qwen"] = time_sent
        else:
            # CRITICAL FIX: Reset detector completely before processing
            print("Resetting detector state for consistency with batch scripts...")
            pattern_detector.reset_fields()

            # Process the entire audio with one continuous detection pass
            # This exactly mimics how batch scripts work
            all_change_points = process_audio_completely_realtime(audio, sr, qwen_description, pattern_detector, args)

            # Filter change points: only keep those at 15s or later for Qwen processing
            filtered_change_points = [cp for cp in all_change_points if cp >= 15]

            print(f"All detected change points: {all_change_points}")
            print(f"Change points for Qwen processing (>=15s): {filtered_change_points}")

            result["change_points"] = all_change_points  # Store all detected changes
            result["qwen_processed_changes"] = filtered_change_points  # Store which ones were sent to Qwen

            # Process only the filtered change points (>=15s) with Qwen
            for timestamp in filtered_change_points:
                pattern_change = process_change_point_realtime(
                    timestamp, audio, sr, qwen_description, args, pattern_detector
                )
                result["pattern_changes"].append(pattern_change)

                if args.verbose_metrics:
                    print(
                        f"Processed change at {timestamp}s with Qwen: {pattern_change['biooss_metrics']['detection_confidence']:.4f}")

            # Generate initial description (always process first 15 seconds)
            initial_audio = audio[:int(15.0 * sr)]
            temp_path = "/tmp/initial_segment_for_description.wav"
            sf.write(temp_path, initial_audio, sr)
            initial_description = qwen_description.generate_description(temp_path)
            try:
                os.remove(temp_path)
            except:
                pass

            result["initial_segment"] = {
                "description": initial_description,
                "timestamp": 0,
                "length": 15
            }

            # Calculate total time sent to Qwen with filtered change points
            result["time_length_sent_to_qwen"] = calculate_time_sent_to_qwen(filtered_change_points, total_duration)

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

        # Clean up models
        del echoic_memory, pattern_detector, qwen_description
        del audio_input_queue, recall_queue, description_queue

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def process_audio_completely_realtime(audio, sr, qwen_description, pattern_detector, args):
    """Process entire audio in one pass, exactly like batch scripts"""
    change_points = []
    temp_dir = "/tmp"

    # Create windows exactly like batch scripts: 4.0s window, 1.0s stride
    window_length_samples = int(args.window_length * sr)  # 4.0 * 32000 = 128000 samples
    stride_length_samples = int(args.stride_length * sr)  # 1.0 * 32000 = 32000 samples

    # Generate windows exactly like create_sliding_windows in batch scripts
    audio_windows = []
    for start_sample in range(0, len(audio) - window_length_samples + 1, stride_length_samples):
        end_sample = start_sample + window_length_samples
        audio_windows.append(audio[start_sample:end_sample])

    print(f"Processing {len(audio_windows)} windows (exactly like batch scripts)")

    # Process ALL windows from 0 (exactly like batch scripts do)
    for window_idx in range(len(audio_windows)):
        if window_idx % 20 == 0:
            print(f"Processing timestep {window_idx + 1}/{len(audio_windows)}")

        # Get current window
        current_window = audio_windows[window_idx]

        # Save window temporarily
        window_path = os.path.join(temp_dir, f"realtime_window_{window_idx}.wav")
        sf.write(window_path, current_window, sr)

        # Run BioOSS detection (exactly like batch scripts call)
        biooss_confidence, biooss_result = pattern_detector.detect_pattern_change(window_path)

        # Check if change detected (exactly like batch scripts)
        if biooss_result and biooss_result.get('change_detected', False):
            change_points.append(window_idx)
            print(f"Change detected at timestep {window_idx} (confidence: {biooss_result.get('confidence', 0):.3f})")

        # Clean up temporary file
        try:
            os.remove(window_path)
        except:
            pass

    return change_points


def main():
    """Main function for processing complete dataset"""
    args = parse_args()
    device = get_device(args)

    # dataset_path = "/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio"
    dataset_path = "/home/zhyuan/Desktop/Qwen-Audio/esc50_attention_test"
    results_path = "/home/zhyuan/Desktop/Qwen-Audio/results"

    # Create results directory
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

            # Memory cleanup between files
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save all results to JSON file (matching reference format exactly)
    if all_results:
        json_output = {
            "dataset_info": {
                "dataset_path": dataset_path,
                "total_files_processed": len(all_results),
                "processing_date": datetime.now().isoformat(),
                "detection_method": "BioOSS_Multi_Metric_FDTD_Realtime",
                "device": device,
                "detection_parameters": {
                    "energy_weight": args.energy_weight,
                    "pca_weight": args.pca_weight,
                    "cosine_weight": args.cosine_weight,
                    "consensus_threshold": args.consensus_threshold,
                    "min_persistence": args.min_persistence
                }
            },
            "results": all_results
        }

        # json_path = os.path.join(results_path, "biooss_multimetric_pattern_change_results_realtime.json")
        json_path = os.path.join(results_path, "biooss_multimetric_pattern_change_results_esc50.json")

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False)
            print(f"All results saved to {json_path}")

            # Print summary
            total_changes = sum(len(result["pattern_changes"]) for result in all_results)
            total_time_sent = sum(result["time_length_sent_to_qwen"] for result in all_results)
            print(f"Summary: Processed {len(all_results)} files with {total_changes} total pattern changes detected")
            print(f"Total time sent to Qwen: {total_time_sent} seconds")

        except Exception as e:
            print(f"Error saving JSON file: {str(e)}")
            traceback.print_exc()
    else:
        print("No valid data collected, skipping JSON file creation")


def main_single_test():
    """Main function for testing single audio file"""
    args = parse_args()
    device = get_device(args)

    # audio_file = {'audio': '/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio/R0002_segment_ambisonics.wav'}
    audio_file = {'audio': '/home/zhyuan/Desktop/Qwen-Audio/esc50_attention_test/composite_trial_1.wav'}
    results_path = "/home/zhyuan/Desktop/Qwen-Audio/results"

    # Create results directory
    os.makedirs(results_path, exist_ok=True)

    file_path = audio_file['audio']
    file_name = os.path.basename(file_path)

    print(f"Processing {file_name} with BioOSS multi-metric detector on {device}...")

    # Process the audio
    file_data = process_audio(file_path)

    # if file_data:
    #     # For single test, create a structure similar to main() but with single result
    #     json_output = {
    #         "dataset_info": {
    #             "file_path": file_path,
    #             "processing_date": datetime.now().isoformat(),
    #             "detection_method": "BioOSS_Multi_Metric_FDTD_Realtime",
    #             "device": device,
    #             "detection_parameters": {
    #                 "energy_weight": args.energy_weight,
    #                 "pca_weight": args.pca_weight,
    #                 "cosine_weight": args.cosine_weight,
    #                 "consensus_threshold": args.consensus_threshold,
    #                 "min_persistence": args.min_persistence
    #             }
    #         },
    #         "results": [file_data]  # Single result in array format for consistency
    #     }
    #
    #     json_path = os.path.join(results_path, f"{file_name.replace('.wav', '')}_biooss_multimetric_result.json")
    #
    #     try:
    #         with open(json_path, 'w', encoding='utf-8') as f:
    #             json.dump(json_output, f, indent=2, ensure_ascii=False)
    #         print(f"Result saved to {json_path}")
    #
    #         # Print summary with clear distinction between detected and processed changes
    #         all_changes = file_data['change_points']
    #         qwen_processed = file_data.get('qwen_processed_changes', [])
    #         early_changes = [cp for cp in all_changes if cp < 15]
    #
    #         print(f"Summary:")
    #         print(f"  Total changes detected: {len(all_changes)} at timesteps {all_changes}")
    #         print(f"  Changes processed with Qwen: {len(qwen_processed)} at timesteps {qwen_processed}")
    #         if early_changes:
    #             print(f"  Early changes (not sent to Qwen): {len(early_changes)} at timesteps {early_changes}")
    #         print(f"  Time sent to Qwen: {file_data['time_length_sent_to_qwen']} seconds")
    #
    #     except Exception as e:
    #         print(f"Error saving JSON file: {str(e)}")
    #         traceback.print_exc()
    # else:
    #     print("No valid data collected")


def create_enhanced_mel_spectrogram_with_detection(audio, sr, detection_results, stride_length,
                                                   output_path, title_suffix=""):
    """
    Create enhanced mel spectrogram with energy overlay and detection markers
    Similar to the second code's visualization approach

    Args:
        audio: Audio signal (numpy array)
        sr: Sample rate
        detection_results: Dict containing detection data with metrics
        stride_length: Stride length in seconds
        output_path: Path to save visualization
        title_suffix: Optional title suffix
    """
    print("Generating enhanced mel spectrogram with energy overlay...")

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, fmax=8000, hop_length=512, n_fft=2048
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot mel spectrogram as background
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
                             fmax=8000, cmap='viridis', alpha=0.8, ax=ax)

    # Set x-axis limits
    audio_duration = len(audio) / sr
    ax.set_xlim(0, min(audio_duration, 60))

    # Extract detection data
    change_points = detection_results.get('change_points', [])
    change_times = [cp * stride_length for cp in change_points]

    # Create time axis for metrics (if available)
    if 'detailed_metrics' in detection_results and detection_results['detailed_metrics']:
        try:
            # Extract energy metrics for overlay
            energy_values = []
            energy_thresholds = []
            timestamps = []

            for i, metrics in enumerate(detection_results['detailed_metrics']):
                if isinstance(metrics, dict) and 'raw_metrics' in metrics:
                    energy_val = metrics['raw_metrics'].get('energy', 0)
                    energy_thresh = metrics['thresholds'].get('energy', 0)

                    # Convert to Python float to avoid JSON serialization issues
                    energy_values.append(float(energy_val))
                    energy_thresholds.append(float(energy_thresh))
                    timestamps.append(i * stride_length)

            if energy_values and len(energy_values) > 0:
                # Scale metrics to fit frequency range for visualization
                max_freq = 128
                max_energy = max(energy_values) if max(energy_values) > 0 else 1
                metric_scale_factor = max_freq * 0.8 / max_energy

                scaled_energies = np.array(energy_values) * metric_scale_factor
                scaled_thresholds = np.array(energy_thresholds) * metric_scale_factor

                # Create twin axis for energy overlay
                ax2 = ax.twinx()
                ax2.set_ylim(0, max_freq)
                ax2.set_xlim(0, min(audio_duration, 60))

                # Plot energy line (similar to KL divergence in second code)
                valid_indices = [i for i, t in enumerate(timestamps) if t <= min(audio_duration, 60)]
                if valid_indices:
                    valid_times = [timestamps[i] for i in valid_indices]
                    valid_energies = [scaled_energies[i] for i in valid_indices]
                    valid_thresholds = [scaled_thresholds[i] for i in valid_indices]

                    # Plot energy metric line
                    ax2.plot(valid_times, valid_energies, color='cyan', linewidth=2,
                             alpha=0.9, label='Energy Metric')

                    # Plot threshold line
                    ax2.plot(valid_times, valid_thresholds, color=reds[1], linewidth=2,
                             linestyle='--', alpha=0.8, label='Threshold')

                    # Fill threshold area
                    ax2.fill_between(valid_times, 0, valid_thresholds,
                                     color=reds[1], alpha=0.1)

                    # Set labels for twin axis
                    ax2.set_ylabel('Metric Value (Scaled)', fontsize=22, color='cyan')

        except Exception as e:
            print(f"Warning: Could not create energy overlay: {e}")

    # Add detection markers
    max_freq = 128
    for i, change_time in enumerate(change_times):
        if change_time <= min(audio_duration, 60):
            # Vertical line for detection
            ax.axvline(x=change_time, color=reds[0], linewidth=3, alpha=0.9)

            # Detection marker with timestamp
            ax.text(change_time, max_freq * 0.9, f'{change_time:.0f}s',
                    rotation=0, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=reds[0], alpha=0.8),
                    color='white', fontsize=16, fontweight='bold')

            # Highlight region
            ax.axvspan(change_time - stride_length / 2, change_time + stride_length / 2,
                       color=reds[0], alpha=0.1)

    # Labels and title
    ax.set_xlabel('Time (seconds)', fontsize=22)
    ax.set_ylabel('Mel Frequency Bins', fontsize=22)

    # Create legend if energy overlay exists
    if 'ax2' in locals():
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc='upper right', framealpha=0.9, fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches=None,
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Enhanced spectrogram saved to: {output_path}")

    # Return stats with JSON-serializable values
    total_changes = len([t for t in change_times if t <= min(audio_duration, 60)])
    return {
        'total_detections': int(total_changes),
        'detection_times': [float(t) for t in change_times],
        'visualization_path': output_path
    }


def collect_detection_metrics_during_processing(pattern_detector, window_idx, biooss_result):
    """
    Collect detailed metrics during processing to avoid JSON serialization issues
    """
    detailed_metrics = {
        'raw_metrics': {
            'energy': float(biooss_result.get('energy', 0)) if biooss_result else 0.0,
            'pca': float(biooss_result.get('pca', 0)) if biooss_result else 0.0,
            'cosine': float(biooss_result.get('cosine', 0)) if biooss_result else 0.0
        },
        'thresholds': {
            'energy': 0.0,  # Will be filled by detector
            'pca': 0.0,
            'cosine': 0.0
        },
        'consensus_score': 0.0,
        'persistence_ratio': 0.0
    }

    # Try to extract threshold information from detector if available
    try:
        if hasattr(pattern_detector, 'get_current_thresholds'):
            thresholds = pattern_detector.get_current_thresholds()
            if thresholds:
                for key in detailed_metrics['thresholds']:
                    if key in thresholds:
                        detailed_metrics['thresholds'][key] = float(thresholds[key])
    except:
        pass

    return detailed_metrics


def create_enhanced_mel_spectrogram_with_detection(audio, sr, detection_results, stride_length,
                                                   output_path, title_suffix=""):
    """
    Create enhanced mel spectrogram with energy overlay and detection markers
    Similar to the second code's visualization approach

    Args:
        audio: Audio signal (numpy array)
        sr: Sample rate
        detection_results: Dict containing detection data with metrics
        stride_length: Stride length in seconds
        output_path: Path to save visualization
        title_suffix: Optional title suffix
    """
    print("Generating enhanced mel spectrogram with energy overlay...")

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, fmax=8000, hop_length=512, n_fft=2048
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot mel spectrogram as background
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
                             fmax=8000, cmap='viridis', alpha=0.8, ax=ax)

    # Set x-axis limits
    audio_duration = len(audio) / sr
    ax.set_xlim(0, min(audio_duration, 60))

    # Extract detection data
    change_points = detection_results.get('change_points', [])
    change_times = [cp * stride_length for cp in change_points]

    # Create time axis for metrics (if available)
    if 'detailed_metrics' in detection_results and detection_results['detailed_metrics']:
        try:
            # Extract energy metrics for overlay
            energy_values = []
            energy_thresholds = []
            timestamps = []

            for i, metrics in enumerate(detection_results['detailed_metrics']):
                if isinstance(metrics, dict) and 'raw_metrics' in metrics:
                    energy_val = metrics['raw_metrics'].get('energy', 0)
                    energy_thresh = metrics['thresholds'].get('energy', 0)

                    # Convert to Python float to avoid JSON serialization issues
                    energy_values.append(float(energy_val))
                    energy_thresholds.append(float(energy_thresh))
                    timestamps.append(i * stride_length)

            if energy_values and len(energy_values) > 0:
                # Scale metrics to fit frequency range for visualization
                max_freq = 128
                max_energy = max(energy_values) if max(energy_values) > 0 else 1
                metric_scale_factor = max_freq * 0.8 / max_energy

                scaled_energies = np.array(energy_values) * metric_scale_factor
                scaled_thresholds = np.array(energy_thresholds) * metric_scale_factor

                # Create twin axis for energy overlay
                ax2 = ax.twinx()
                ax2.set_ylim(0, max_freq)
                ax2.set_xlim(0, min(audio_duration, 60))

                # Plot energy line (similar to KL divergence in second code)
                valid_indices = [i for i, t in enumerate(timestamps) if t <= min(audio_duration, 60)]
                if valid_indices:
                    valid_times = [timestamps[i] for i in valid_indices]
                    valid_energies = [scaled_energies[i] for i in valid_indices]
                    valid_thresholds = [scaled_thresholds[i] for i in valid_indices]

                    # Plot energy metric line
                    ax2.plot(valid_times, valid_energies, color='cyan', linewidth=2,
                             alpha=0.9, label='Energy Metric')

                    # Plot the adaptive threshold as a red dashed line.
                    ax2.plot(valid_times, valid_thresholds, color=reds[1], linewidth=3,
                             linestyle='--', alpha=0.9, label='Adaptive Threshold')

                    # Fill the threshold baseline area with a light red shade.
                    ax2.fill_between(valid_times, 0, valid_thresholds,
                                     color=reds[1], alpha=0.1)

                    # Set labels for twin axis
                    ax2.set_ylabel('Metric Value (Scaled)', fontsize=22, color='cyan')

        except Exception as e:
            print(f"Warning: Could not create energy overlay: {e}")

    # Add detection markers
    max_freq = 128
    for i, change_time in enumerate(change_times):
        if change_time <= min(audio_duration, 60):
            # Vertical line for detection
            ax.axvline(x=change_time, color=reds[0], linewidth=3, alpha=0.9)

            # Detection marker with timestamp
            ax.text(change_time, max_freq * 0.9, f'{change_time:.0f}s',
                    rotation=0, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=reds[0], alpha=0.8),
                    color='white', fontsize=16, fontweight='bold')

            # Highlight region
            ax.axvspan(change_time - stride_length / 2, change_time + stride_length / 2,
                       color=reds[0], alpha=0.1)

    # Labels and title
    ax.set_xlabel('Time (seconds)', fontsize=22)
    ax.set_ylabel('Mel Frequency Bins', fontsize=22)

    # Create legend if energy overlay exists
    if 'ax2' in locals():
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc='upper right', framealpha=0.9, fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Enhanced spectrogram saved to: {output_path}")

    # Return stats with JSON-serializable values
    total_changes = len([t for t in change_times if t <= min(audio_duration, 60)])
    return {
        'total_detections': int(total_changes),
        'detection_times': [float(t) for t in change_times],
        'visualization_path': output_path
    }


def collect_detection_metrics_during_processing(pattern_detector, window_idx, biooss_result):
    """
    Collect detailed metrics during processing with proper threshold extraction
    """
    detailed_metrics = {
        'raw_metrics': {
            'energy': 0.0,
            'pca': 0.0,
            'cosine': 0.0
        },
        'thresholds': {
            'energy': 0.0,
            'pca': 0.0,
            'cosine': 0.0
        },
        'consensus_score': 0.0,
        'persistence_ratio': 0.0
    }

    # Extract metrics from biooss_result if available
    if biooss_result:
        # Try to get detailed metrics from the result
        if 'detailed_metrics' in biooss_result:
            detailed_result = biooss_result['detailed_metrics']

            # Extract raw metrics
            if 'raw_metrics' in detailed_result:
                raw_metrics = detailed_result['raw_metrics']
                detailed_metrics['raw_metrics']['energy'] = float(raw_metrics.get('energy', 0))
                detailed_metrics['raw_metrics']['pca'] = float(raw_metrics.get('pca', 0))
                detailed_metrics['raw_metrics']['cosine'] = float(raw_metrics.get('cosine', 0))

            # Extract thresholds
            if 'thresholds' in detailed_result:
                thresholds = detailed_result['thresholds']
                detailed_metrics['thresholds']['energy'] = float(thresholds.get('energy', 0))
                detailed_metrics['thresholds']['pca'] = float(thresholds.get('pca', 0))
                detailed_metrics['thresholds']['cosine'] = float(thresholds.get('cosine', 0))

            # Extract other metrics
            detailed_metrics['consensus_score'] = float(detailed_result.get('consensus_score', 0))
            detailed_metrics['persistence_ratio'] = float(detailed_result.get('persistence_ratio', 0))

        else:
            # Fallback: extract basic metrics
            detailed_metrics['raw_metrics']['energy'] = float(biooss_result.get('energy', 0))
            # Set a simple threshold based on energy (fallback)
            energy_val = detailed_metrics['raw_metrics']['energy']
            detailed_metrics['thresholds']['energy'] = energy_val * 0.7  # Simple heuristic

    return detailed_metrics


def create_enhanced_mel_spectrogram_with_detection(audio, sr, detection_results, stride_length,
                                                   output_path, title_suffix=""):
    """
    Create enhanced mel spectrogram with energy overlay and detection markers
    Similar to the second code's visualization approach

    Args:
        audio: Audio signal (numpy array)
        sr: Sample rate
        detection_results: Dict containing detection data with metrics
        stride_length: Stride length in seconds
        output_path: Path to save visualization
        title_suffix: Optional title suffix
    """
    print("Generating enhanced mel spectrogram with energy overlay...")

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, fmax=8000, hop_length=512, n_fft=2048
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot mel spectrogram as background
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
                             fmax=8000, cmap='viridis', alpha=0.8, ax=ax)

    # Set x-axis limits
    audio_duration = len(audio) / sr
    ax.set_xlim(0, min(audio_duration, 60))

    # Extract detection data
    change_points = detection_results.get('change_points', [])
    change_times = [cp * stride_length for cp in change_points]

    # Create time axis for metrics (if available)
    if 'detailed_metrics' in detection_results and detection_results['detailed_metrics']:
        try:
            # Extract energy metrics for overlay
            energy_values = []
            energy_thresholds = []
            timestamps = []

            for i, metrics in enumerate(detection_results['detailed_metrics']):
                if isinstance(metrics, dict) and 'raw_metrics' in metrics:
                    energy_val = metrics['raw_metrics'].get('energy', 0)
                    energy_thresh = metrics['thresholds'].get('energy', 0)

                    # Convert to Python float to avoid JSON serialization issues
                    energy_values.append(float(energy_val))
                    energy_thresholds.append(float(energy_thresh))
                    timestamps.append(i * stride_length)

            if energy_values and len(energy_values) > 0:
                # Scale metrics to fit frequency range for visualization
                max_freq = 128
                max_energy = max(energy_values) if max(energy_values) > 0 else 1
                metric_scale_factor = max_freq * 0.8 / max_energy

                scaled_energies = np.array(energy_values) * metric_scale_factor
                scaled_thresholds = np.array(energy_thresholds) * metric_scale_factor

                # Create twin axis for energy overlay
                ax2 = ax.twinx()
                ax2.set_ylim(0, max_freq)
                ax2.set_xlim(0, min(audio_duration, 60))

                # Plot energy line (similar to KL divergence in second code)
                valid_indices = [i for i, t in enumerate(timestamps) if t <= min(audio_duration, 60)]
                if valid_indices:
                    valid_times = [timestamps[i] for i in valid_indices]
                    valid_energies = [scaled_energies[i] for i in valid_indices]
                    valid_thresholds = [scaled_thresholds[i] for i in valid_indices]

                    # Plot energy metric line
                    ax2.plot(valid_times, valid_energies, color='cyan', linewidth=2,
                             alpha=0.9, label='Energy Metric')

                    # Plot the adaptive threshold as a red dashed line.
                    ax2.plot(valid_times, valid_thresholds, color=reds[1], linewidth=3,
                             linestyle='--', alpha=0.9, label='Adaptive Threshold')

                    # Fill the threshold baseline area with a light red shade.
                    ax2.fill_between(valid_times, 0, valid_thresholds,
                                     color=reds[1], alpha=0.1)

                    # Set labels for twin axis
                    ax2.set_ylabel('Metric Value (Scaled)', fontsize=22, color='cyan')

        except Exception as e:
            print(f"Warning: Could not create energy overlay: {e}")

    # Add detection markers
    max_freq = 128
    for i, change_time in enumerate(change_times):
        if change_time <= min(audio_duration, 60):
            # Vertical line for detection
            ax.axvline(x=change_time, color=reds[0], linewidth=3, alpha=0.9)

            # Detection marker with timestamp
            ax.text(change_time, max_freq * 0.9, f'{change_time:.0f}s',
                    rotation=0, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=reds[0], alpha=0.8),
                    color='white', fontsize=16, fontweight='bold')

            # Highlight region
            ax.axvspan(change_time - stride_length / 2, change_time + stride_length / 2,
                       color=reds[0], alpha=0.1)

    # Labels and title
    ax.set_xlabel('Time (seconds)', fontsize=22)
    ax.set_ylabel('Mel Frequency Bins', fontsize=22)

    # Create legend if energy overlay exists
    if 'ax2' in locals():
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc='upper right', framealpha=0.9, fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Enhanced spectrogram saved to: {output_path}")

    # Return stats with JSON-serializable values
    total_changes = len([t for t in change_times if t <= min(audio_duration, 60)])
    return {
        'total_detections': int(total_changes),
        'detection_times': [float(t) for t in change_times],
        'visualization_path': output_path
    }


def collect_detection_metrics_during_processing(pattern_detector, window_idx, biooss_result):
    """
    Collect detailed metrics during processing with proper threshold extraction
    """
    detailed_metrics = {
        'raw_metrics': {
            'energy': 0.0,
            'pca': 0.0,
            'cosine': 0.0
        },
        'thresholds': {
            'energy': 0.0,
            'pca': 0.0,
            'cosine': 0.0
        },
        'consensus_score': 0.0,
        'persistence_ratio': 0.0
    }

    # Extract metrics from biooss_result if available
    if biooss_result:
        # Try to get detailed metrics from the result
        if 'detailed_metrics' in biooss_result:
            detailed_result = biooss_result['detailed_metrics']

            # Extract raw metrics
            if 'raw_metrics' in detailed_result:
                raw_metrics = detailed_result['raw_metrics']
                detailed_metrics['raw_metrics']['energy'] = float(raw_metrics.get('energy', 0))
                detailed_metrics['raw_metrics']['pca'] = float(raw_metrics.get('pca', 0))
                detailed_metrics['raw_metrics']['cosine'] = float(raw_metrics.get('cosine', 0))

            # Extract thresholds
            if 'thresholds' in detailed_result:
                thresholds = detailed_result['thresholds']
                detailed_metrics['thresholds']['energy'] = float(thresholds.get('energy', 0))
                detailed_metrics['thresholds']['pca'] = float(thresholds.get('pca', 0))
                detailed_metrics['thresholds']['cosine'] = float(thresholds.get('cosine', 0))

            # Extract other metrics
            detailed_metrics['consensus_score'] = float(detailed_result.get('consensus_score', 0))
            detailed_metrics['persistence_ratio'] = float(detailed_result.get('persistence_ratio', 0))

        else:
            # Fallback: extract basic metrics
            detailed_metrics['raw_metrics']['energy'] = float(biooss_result.get('energy', 0))
            # Set a simple threshold based on energy (fallback)
            energy_val = detailed_metrics['raw_metrics']['energy']
            detailed_metrics['thresholds']['energy'] = energy_val * 0.7  # Simple heuristic

    return detailed_metrics


def create_enhanced_mel_spectrogram_with_detection(audio, sr, detection_results, stride_length,
                                                   output_path, title_suffix=""):
    """
    Create enhanced mel spectrogram with energy overlay and detection markers
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, fmax=8000, hop_length=512, n_fft=2048
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot mel spectrogram as background
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
                             fmax=8000, cmap='viridis', alpha=0.8, ax=ax)

    # Set x-axis limits
    audio_duration = len(audio) / sr
    ax.set_xlim(0, min(audio_duration, 60))

    # Extract detection data
    change_points = detection_results.get('change_points', [])
    change_times = [cp * stride_length for cp in change_points]

    # Create energy overlay if metrics available
    if 'detailed_metrics' in detection_results and detection_results['detailed_metrics']:
        try:
            energy_values = []
            energy_thresholds = []
            timestamps = []

            for i, metrics in enumerate(detection_results['detailed_metrics']):
                if isinstance(metrics, dict) and 'raw_metrics' in metrics:
                    energy_val = metrics['raw_metrics'].get('energy', 0)
                    energy_thresh = metrics['thresholds'].get('energy', 0)

                    energy_values.append(float(energy_val))
                    energy_thresholds.append(float(energy_thresh))
                    timestamps.append(i * stride_length)

            if energy_values and len(energy_values) > 0:
                # Scale metrics to fit frequency range
                max_freq = 128
                max_energy = max(energy_values) if max(energy_values) > 0 else 1
                metric_scale_factor = max_freq * 0.8 / max_energy

                scaled_energies = np.array(energy_values) * metric_scale_factor
                scaled_thresholds = np.array(energy_thresholds) * metric_scale_factor

                # Create twin axis for energy overlay
                ax2 = ax.twinx()
                ax2.set_ylim(0, max_freq)
                ax2.set_xlim(0, min(audio_duration, 60))

                # Plot energy and threshold lines
                valid_indices = [i for i, t in enumerate(timestamps) if t <= min(audio_duration, 60)]
                if valid_indices:
                    valid_times = [timestamps[i] for i in valid_indices]
                    valid_energies = [scaled_energies[i] for i in valid_indices]
                    valid_thresholds = [scaled_thresholds[i] for i in valid_indices]

                    # Plot energy metric line
                    ax2.plot(valid_times, valid_energies, color='cyan', linewidth=2,
                             alpha=0.9, label='Energy Metric')

                    # Plot threshold line as dashed line
                    ax2.plot(valid_times, valid_thresholds, color=reds[1], linewidth=3,
                             linestyle='--', alpha=0.9, label='Adaptive Threshold')

                    # Fill threshold baseline area
                    ax2.fill_between(valid_times, 0, valid_thresholds,
                                     color=reds[1], alpha=0.1)

                    # Set labels for twin axis
                    ax2.set_ylabel('Metric Value (Scaled)', fontsize=22, color='cyan')

        except Exception:
            pass

    # Add detection markers
    max_freq = 128
    for change_time in change_times:
        if change_time <= min(audio_duration, 60):
            # Vertical line for detection
            ax.axvline(x=change_time, color=reds[0], linewidth=3, alpha=0.9)

            # Detection marker with timestamp
            ax.text(change_time, max_freq * 0.9, f'{change_time:.0f}s',
                    rotation=0, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=reds[0], alpha=0.8),
                    color='white', fontsize=16, fontweight='bold')

            # Highlight region
            ax.axvspan(change_time - stride_length / 2, change_time + stride_length / 2,
                       color=reds[0], alpha=0.1)

    # Labels and title
    ax.set_xlabel('Time (seconds)', fontsize=22)
    ax.set_ylabel('Mel Frequency Bins', fontsize=22)

    # Create legend if energy overlay exists
    if 'ax2' in locals():
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc='upper right', framealpha=0.9, fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    # Return stats with JSON-serializable values
    total_changes = len([t for t in change_times if t <= min(audio_duration, 60)])
    return {
        'total_detections': int(total_changes),
        'detection_times': [float(t) for t in change_times],
        'visualization_path': output_path
    }


def collect_detection_metrics_during_processing(pattern_detector, window_idx, biooss_result):
    """
    Collect detailed metrics during processing with proper threshold extraction
    """
    detailed_metrics = {
        'raw_metrics': {
            'energy': 0.0,
            'pca': 0.0,
            'cosine': 0.0
        },
        'thresholds': {
            'energy': 0.0,
            'pca': 0.0,
            'cosine': 0.0
        },
        'consensus_score': 0.0,
        'persistence_ratio': 0.0
    }

    if biooss_result:
        if 'detailed_metrics' in biooss_result:
            detailed_result = biooss_result['detailed_metrics']

            if 'raw_metrics' in detailed_result:
                raw_metrics = detailed_result['raw_metrics']
                detailed_metrics['raw_metrics']['energy'] = float(raw_metrics.get('energy', 0))
                detailed_metrics['raw_metrics']['pca'] = float(raw_metrics.get('pca', 0))
                detailed_metrics['raw_metrics']['cosine'] = float(raw_metrics.get('cosine', 0))

            if 'thresholds' in detailed_result:
                thresholds = detailed_result['thresholds']
                detailed_metrics['thresholds']['energy'] = float(thresholds.get('energy', 0))
                detailed_metrics['thresholds']['pca'] = float(thresholds.get('pca', 0))
                detailed_metrics['thresholds']['cosine'] = float(thresholds.get('cosine', 0))

            detailed_metrics['consensus_score'] = float(detailed_result.get('consensus_score', 0))
            detailed_metrics['persistence_ratio'] = float(detailed_result.get('persistence_ratio', 0))

        else:
            detailed_metrics['raw_metrics']['energy'] = float(biooss_result.get('energy', 0))
            energy_val = detailed_metrics['raw_metrics']['energy']
            detailed_metrics['thresholds']['energy'] = energy_val * 0.7

    return detailed_metrics


def process_audio_completely_realtime_with_metrics(audio, sr, qwen_description, pattern_detector, args):
    """
    Enhanced version that collects metrics for visualization
    """
    change_points = []
    detailed_metrics = []
    temp_dir = "/tmp"

    window_length_samples = int(args.window_length * sr)
    stride_length_samples = int(args.stride_length * sr)

    audio_windows = []
    for start_sample in range(0, len(audio) - window_length_samples + 1, stride_length_samples):
        end_sample = start_sample + window_length_samples
        audio_windows.append(audio[start_sample:end_sample])

    for window_idx in range(len(audio_windows)):
        current_window = audio_windows[window_idx]
        window_path = os.path.join(temp_dir, f"realtime_window_{window_idx}.wav")
        sf.write(window_path, current_window, sr)

        biooss_confidence, biooss_result = pattern_detector.detect_pattern_change(window_path)

        metrics = collect_detection_metrics_during_processing(pattern_detector, window_idx, biooss_result)
        detailed_metrics.append(metrics)

        if biooss_result and biooss_result.get('change_detected', False):
            change_points.append(window_idx)

        try:
            os.remove(window_path)
        except:
            pass

    return change_points, detailed_metrics


def process_audio_with_visualization(test_audio_path):
    """Enhanced audio processing with proper metrics collection and visualization"""
    args = parse_args()
    device = get_device(args)

    figures_dir = "/home/zhyuan/Desktop/Qwen-Audio/Memory_model/results_iclr/figures"
    os.makedirs(figures_dir, exist_ok=True)

    audio_input_queue = Queue()
    recall_queue = Queue()
    description_queue = Queue()

    echoic_memory = EchoicMemory()

    pattern_detector = BioOSSPatternChangeDetector(
        args,
        energy_weight=args.energy_weight,
        pca_weight=args.pca_weight,
        cosine_weight=args.cosine_weight,
        consensus_threshold=args.consensus_threshold,
        min_persistence=args.min_persistence,
        min_confidence_threshold=args.min_confidence_threshold
    )

    if hasattr(pattern_detector, 'biooss_model'):
        pattern_detector.biooss_model.to(device)
    if hasattr(pattern_detector, 'audio_encoder'):
        pattern_detector.audio_encoder.model.to(device)
        pattern_detector.audio_encoder.device = device

    qwen_description = QwenAudioDescription(recall_queue, description_queue, device='cuda:1')

    if not os.path.exists(test_audio_path):
        return None

    try:
        file_id = extract_file_index(test_audio_path)
        audio, sr = load_audio_file_realtime(test_audio_path, args.sample_rate)
        total_duration = len(audio) / sr

        result = {
            "file_id": file_id,
            "audio_path": test_audio_path,
            "total_file_length": float(total_duration),
            "processing_date": datetime.now().isoformat(),
            "detection_method": "BioOSS_Multi_Metric_FDTD_Realtime_with_Visualization",
            "detection_parameters": {
                "energy_weight": float(args.energy_weight),
                "pca_weight": float(args.pca_weight),
                "cosine_weight": float(args.cosine_weight),
                "consensus_threshold": float(args.consensus_threshold),
                "min_persistence": int(args.min_persistence),
                "window_length": float(args.window_length),
                "stride_length": float(args.stride_length),
                "device": device
            },
            "initial_segment": {},
            "pattern_changes": [],
            "change_points": [],
            "time_length_sent_to_qwen": 0,
            "visualization_paths": {}
        }

        if total_duration < 15:
            initial_segment, time_sent = process_short_audio(test_audio_path, qwen_description, total_duration)
            result["initial_segment"] = initial_segment
            result["time_length_sent_to_qwen"] = float(time_sent)

            viz_filename = f"{file_id}_mel_spectrogram_short.pdf"
            viz_path = os.path.join(figures_dir, viz_filename)
            detection_results = {'change_points': []}
            viz_stats = create_enhanced_mel_spectrogram_with_detection(
                audio, sr, detection_results, args.stride_length, viz_path, f"{file_id} (Short)"
            )
            result["visualization_paths"]["spectrogram"] = viz_path

        else:
            pattern_detector.reset_fields()

            all_change_points, detailed_metrics = process_audio_completely_realtime_with_metrics(
                audio, sr, qwen_description, pattern_detector, args
            )

            filtered_change_points = [cp for cp in all_change_points if cp >= 15]

            result["change_points"] = [int(cp) for cp in all_change_points]
            result["qwen_processed_changes"] = [int(cp) for cp in filtered_change_points]

            for timestamp in filtered_change_points:
                pattern_change = process_change_point_realtime(
                    timestamp, audio, sr, qwen_description, args, pattern_detector
                )
                pattern_change["timestamp"] = int(pattern_change["timestamp"])
                pattern_change["change_point"] = int(pattern_change["change_point"])
                for key, value in pattern_change["biooss_metrics"].items():
                    pattern_change["biooss_metrics"][key] = float(value)
                result["pattern_changes"].append(pattern_change)

            initial_audio = audio[:int(15.0 * sr)]
            temp_path = "/tmp/initial_segment_for_description.wav"
            sf.write(temp_path, initial_audio, sr)
            initial_description = qwen_description.generate_description(temp_path)
            os.remove(temp_path)

            result["initial_segment"] = {
                "description": initial_description,
                "timestamp": 0,
                "length": 15
            }

            result["time_length_sent_to_qwen"] = float(
                calculate_time_sent_to_qwen(filtered_change_points, total_duration))

            detection_results = {
                'change_points': all_change_points,
                'detailed_metrics': detailed_metrics
            }

            viz_filename = f"{file_id}_mel_spectrogram_bio.pdf"
            viz_path = os.path.join(figures_dir, viz_filename)
            viz_stats = create_enhanced_mel_spectrogram_with_detection(
                audio, sr, detection_results, args.stride_length, viz_path, file_id
            )
            result["visualization_paths"]["spectrogram"] = viz_path
            result["visualization_stats"] = viz_stats

        return result

    except Exception as e:
        traceback.print_exc()
        return None

    finally:
        try:
            echoic_memory.stop()
            pattern_detector.stop()
            qwen_description.stop()
        except:
            pass

        del echoic_memory, pattern_detector, qwen_description
        del audio_input_queue, recall_queue, description_queue

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main_batch_processing_with_visualization():
    """Batch process all samples with visualization"""
    args = parse_args()
    device = get_device(args)

    SAMPLES = ['R0002', 'R0003', 'R0007', 'R0010', 'R0016', 'R0028',
               'R0030', 'R0031', 'R0037', 'R0056', 'R0078', 'R0130', 'R0131']

    audio_base_path = "/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio"
    results_path = "/home/zhyuan/Desktop/Qwen-Audio/results"

    os.makedirs(results_path, exist_ok=True)

    all_results = []
    processing_summary = {
        'total_samples': len(SAMPLES),
        'successfully_processed': 0,
        'failed_samples': [],
        'total_changes_detected': 0,
        'total_time_sent_to_qwen': 0
    }

    for i, sample_id in enumerate(SAMPLES, 1):
        audio_filename = f"{sample_id}_segment_ambisonics.wav"
        file_path = os.path.join(audio_base_path, audio_filename)

        print(f"[{i}/{len(SAMPLES)}] Processing: {sample_id}")

        if not os.path.exists(file_path):
            processing_summary['failed_samples'].append({
                'sample_id': sample_id,
                'reason': 'File not found'
            })
            continue

        try:
            file_data = process_audio_with_visualization(file_path)

            if file_data:
                all_results.append(file_data)
                processing_summary['successfully_processed'] += 1

                changes_count = len(file_data.get('change_points', []))
                qwen_time = file_data.get('time_length_sent_to_qwen', 0)

                processing_summary['total_changes_detected'] += changes_count
                processing_summary['total_time_sent_to_qwen'] += qwen_time

                print(f"  Complete: {changes_count} changes, {qwen_time:.1f}s to Qwen")

            else:
                processing_summary['failed_samples'].append({
                    'sample_id': sample_id,
                    'reason': 'Processing returned None'
                })

        except Exception as e:
            processing_summary['failed_samples'].append({
                'sample_id': sample_id,
                'reason': f'Exception: {str(e)}'
            })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # print("Batch processing complete!")
    #
    # if all_results:
    #     json_output = {
    #         "dataset_info": {
    #             "dataset_path": audio_base_path,
    #             "total_samples_attempted": len(SAMPLES),
    #             "successfully_processed": processing_summary['successfully_processed'],
    #             "processing_date": datetime.now().isoformat(),
    #             "detection_method": "BioOSS_Multi_Metric_FDTD_Realtime_with_Visualization",
    #             "device": device,
    #             "detection_parameters": {
    #                 "energy_weight": float(args.energy_weight),
    #                 "pca_weight": float(args.pca_weight),
    #                 "cosine_weight": float(args.cosine_weight),
    #                 "consensus_threshold": float(args.consensus_threshold),
    #                 "min_persistence": int(args.min_persistence)
    #             }
    #         },
    #         "processing_summary": processing_summary,
    #         "results": all_results
    #     }
    #
    #     batch_json_path = os.path.join(results_path, "batch_biooss_results_with_visualization.json")
    #     try:
    #         with open(batch_json_path, 'w', encoding='utf-8') as f:
    #             json.dump(json_output, f, indent=2, ensure_ascii=False)
    #         print(f"Results saved to: {batch_json_path}")
    #     except Exception as e:
    #         print(f"Error saving results: {str(e)}")
    #
    # print(f"Processing summary:")
    # print(
    #     f"  Successfully processed: {processing_summary['successfully_processed']}/{processing_summary['total_samples']}")
    # print(f"  Total changes detected: {processing_summary['total_changes_detected']}")
    # print(f"  Total time sent to Qwen: {processing_summary['total_time_sent_to_qwen']:.1f} seconds")
    #
    # if processing_summary['failed_samples']:
    #     print(f"  Failed samples:")
    #     for failed in processing_summary['failed_samples']:
    #         print(f"    {failed['sample_id']}: {failed['reason']}")

    return all_results


def main_single_test_with_visualization():
    """Single sample test function"""
    args = parse_args()
    device = get_device(args)

    # audio_file = {'audio': '/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio/R0002_segment_ambisonics.wav'}
    audio_file = {'audio': '/home/zhyuan/Desktop/Qwen-Audio/esc50_attention_test/composite_trial_1.wav'}
    results_path = "/home/zhyuan/Desktop/Qwen-Audio/results"

    os.makedirs(results_path, exist_ok=True)

    file_path = audio_file['audio']
    file_name = os.path.basename(file_path)

    print(f"Processing {file_name} with visualization...")

    file_data = process_audio_with_visualization(file_path)

    if file_data:
        json_output = {
            "dataset_info": {
                "file_path": file_path,
                "processing_date": datetime.now().isoformat(),
                "detection_method": "BioOSS_Multi_Metric_FDTD_Realtime_with_Visualization",
                "device": device,
                "detection_parameters": {
                    "energy_weight": float(args.energy_weight),
                    "pca_weight": float(args.pca_weight),
                    "cosine_weight": float(args.cosine_weight),
                    "consensus_threshold": float(args.consensus_threshold),
                    "min_persistence": int(args.min_persistence)
                }
            },
            "results": [file_data]
        }

        json_path = os.path.join(results_path, f"{file_name.replace('.wav', '')}_with_visualization.json")

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {json_path}")

            all_changes = file_data['change_points']
            qwen_processed = file_data.get('qwen_processed_changes', [])

            print(f"Summary:")
            print(f"  Total changes detected: {len(all_changes)} at timesteps {all_changes}")
            print(f"  Changes processed with Qwen: {len(qwen_processed)} at timesteps {qwen_processed}")
            print(f"  Time sent to Qwen: {file_data['time_length_sent_to_qwen']} seconds")
            print(f"  Mel spectrogram: {file_data['visualization_paths'].get('spectrogram', 'N/A')}")

        except Exception as e:
            print(f"Error saving JSON: {str(e)}")
            traceback.print_exc()
    else:
        print("No valid data collected")

if __name__ == "__main__":
    # main()
    # main_single_test()
    # main_single_test_with_visualization()
    main_batch_processing_with_visualization()
