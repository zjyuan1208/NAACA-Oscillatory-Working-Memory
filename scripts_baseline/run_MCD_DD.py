import argparse
import os
import sys
import numpy as np
import torch
import time
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

# Import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '/home/zhyuan/Desktop/PCD/baselines'))
from MCD_DD import MCD_DD, smooth_detections

sys.path.append(os.path.dirname(__file__))
from pann_backbone import load_pretrained_pann, create_sliding_windows, load_audio_file


def parse_arguments():
    parser = argparse.ArgumentParser(description='MCD-DD Baseline')

    # Required (with defaults)
    parser.add_argument('--audio_path', type=str,
                        default='/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio/R0056_segment_ambisonics.wav',
                        help='Path to input audio file')
    parser.add_argument('--pann_checkpoint', type=str,
                        default='/home/zhyuan/Desktop/audioset_tagging_cnn/Cnn14_mAP=0.431.pth',
                        help='Path to PANN pretrained checkpoint')
    parser.add_argument('--output_dir', type=str, default='./baseline_results')

    # Audio
    parser.add_argument('--window_length', type=float, default=4.0)
    parser.add_argument('--stride_length', type=float, default=1.0)
    parser.add_argument('--sample_rate', type=int, default=32000)

    # MCD-DD - Balanced sensitivity
    parser.add_argument('--sub_window_num', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--output_size', type=int, default=50)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--eps_small', type=float, default=0.01)      # Moderate noise
    parser.add_argument('--eps_big', type=float, default=0.05)        # Moderate noise
    parser.add_argument('--temperature', type=float, default=0.5)     # Balanced temperature
    parser.add_argument('--percentile', type=float, default=0.88)     # Balanced threshold
    parser.add_argument('--learning_rate', type=float, default=0.005)

    # Other
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def extract_pann_features(audio_segments: List[np.ndarray], pann_model, device: str) -> torch.Tensor:
    """Extract PANN features"""
    print(f"Extracting PANN features from {len(audio_segments)} segments...")

    features = []
    for segment in audio_segments:
        segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = pann_model(segment_tensor)
            features.append(output['clipwise_output'].cpu())

    return torch.cat(features, dim=0)


def run_mcd_dd(pann_features: torch.Tensor, args, stride_length: float) -> Dict:
    """Run MCD-DD detection following original style"""
    print("Running MCD-DD detection...")

    # Initialize detector
    detector = MCD_DD(
        input_dim=527,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        sub_window_num=args.sub_window_num,
        k=args.k,
        eps_small=args.eps_small,
        eps_big=args.eps_big,
        temperature=args.temperature,
        percentile=args.percentile,
        learning_rate=args.learning_rate,
        device=args.device
    )

    # Parameters - balanced window sizing
    T = pann_features.shape[0]

    # Balanced approach for reasonable detection
    min_windows_needed = 8  # Reasonable number of windows
    max_win_size = T // min_windows_needed

    # Balanced window size calculation
    original_win_size = max(args.sub_window_num * 6, T // 12)
    win_size = min(original_win_size, max_win_size)

    # Ensure minimum reasonable size
    win_size = max(win_size, args.sub_window_num * 3)

    # Moderate slide for balanced sensitivity
    slide = max(int(win_size / 4), args.sub_window_num)

    print(f"Audio length: {T} timesteps")
    print(f"Window size: {win_size}, Slide: {slide}")
    print(f"Expected number of windows: {(T - win_size) // slide + 1}")

    # Detection loop
    detections = []
    detected_changes = []
    change_timestamps = []
    confidence_scores = []
    thresholds = []
    threshold = 0.0
    first_window = True
    window_count = 0

    start_time = time.time()

    for i in range(0, T - win_size + 1, slide):
        window_data = pann_features[i:i + win_size]
        window_count += 1

        if args.verbose and window_count % 5 == 0:
            print(f"Processing window {window_count} at position {i}")

        if not first_window:
            # Test phase
            distances = detector.test(window_data)
            if distances:
                max_distance = max(distances)
                is_drift = max_distance > threshold
                detections.append(is_drift)

                # Calculate confidence
                confidence = max(0.0, (max_distance - threshold) / (threshold + 1e-8))
                confidence_scores.append(confidence)
                thresholds.append(threshold)

                if args.verbose:
                    print(
                        f"Window {window_count}: distance={max_distance:.4f}, threshold={threshold:.4f}, drift={is_drift}")

                if is_drift:
                    drift_start = max(0, i + win_size - slide)
                    drift_end = min(T, i + win_size)

                    # Convert to time stamps
                    start_time_sec = drift_start * stride_length
                    end_time_sec = drift_end * stride_length
                    change_timestamps.append(start_time_sec)

                    detected_changes.append({
                        'start': drift_start,
                        'end': drift_end,
                        'start_time': start_time_sec,
                        'end_time': end_time_sec,
                        'distance': max_distance,
                        'threshold': threshold,
                        'confidence': confidence,
                        'window_idx': window_count
                    })

                    if args.verbose:
                        print(f"Drift detected at {start_time_sec:.1f}s-{end_time_sec:.1f}s "
                              f"(distance: {max_distance:.4f} > threshold: {threshold:.4f})")

        # Training phase
        old_threshold = threshold
        threshold = detector.train(window_data)

        if args.verbose:
            print(f"Window {window_count}: threshold updated from {old_threshold:.4f} to {threshold:.4f}")

        first_window = False

    processing_time = time.time() - start_time

    # Apply moderate post-processing
    if len(change_timestamps) > 1:
        # Remove changes that are too close together (within 8 seconds)
        filtered_timestamps = []
        filtered_changes = []
        min_interval = 8.0  # Moderate interval

        for i, (timestamp, change) in enumerate(zip(change_timestamps, detected_changes)):
            if i == 0 or timestamp - filtered_timestamps[-1] >= min_interval:
                filtered_timestamps.append(timestamp)
                filtered_changes.append(change)

        change_timestamps = filtered_timestamps
        detected_changes = filtered_changes

    # Moderate confidence filtering: only keep reasonable confidence changes
    if detected_changes:
        conf_threshold = 0.2  # Lower threshold to keep more changes
        filtered_timestamps = []
        filtered_changes = []

        for timestamp, change in zip(change_timestamps, detected_changes):
            if change['confidence'] > conf_threshold:
                filtered_timestamps.append(timestamp)
                filtered_changes.append(change)

        change_timestamps = filtered_timestamps
        detected_changes = filtered_changes

    # Summary statistics
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
    max_confidence = max(confidence_scores) if confidence_scores else 0.0
    avg_threshold = np.mean(thresholds) if thresholds else 0.0

    print(f"\nDetection completed:")
    print(f"Total windows processed: {window_count}")
    print(f"Windows with detection test: {len(detections)}")
    print(f"Raw detections: {sum(detections)}")
    print(f"Filtered detections: {len(detected_changes)}")
    print(f"Average threshold: {avg_threshold:.4f}")

    return {
        'detections': detections,
        'detected_changes': detected_changes,
        'change_timestamps': change_timestamps,
        'confidence_scores': confidence_scores,
        'thresholds': thresholds,
        'processing_time': processing_time,
        'total_detections': len(detected_changes),  # Use filtered count
        'total_windows': window_count,
        'window_params': {
            'win_size': win_size,
            'slide': slide,
            'audio_length': T
        },
        'statistics': {
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'avg_threshold': avg_threshold
        }
    }


def save_results(results: Dict, output_path: str, args):
    """Save results"""
    # Save numpy data
    np.savez(output_path.replace('.txt', '.npz'), **results)

    # Save text summary
    with open(output_path, 'w') as f:
        f.write("MCD-DD Baseline Results\n")
        f.write("=" * 30 + "\n\n")

        # Basic stats
        f.write("Basic Statistics:\n")
        f.write(f"Total detections: {results['total_detections']}\n")
        f.write(f"Total windows: {results['total_windows']}\n")
        f.write(f"Processing time: {results['processing_time']:.2f}s\n")

        # Window parameters
        wp = results['window_params']
        f.write(f"Audio length: {wp['audio_length']} timesteps\n")
        f.write(f"Window size: {wp['win_size']}\n")
        f.write(f"Slide size: {wp['slide']}\n")

        # Detection statistics
        if 'statistics' in results:
            stats = results['statistics']
            f.write(f"Average confidence: {stats['avg_confidence']:.4f}\n")
            f.write(f"Max confidence: {stats['max_confidence']:.4f}\n")
            f.write(f"Average threshold: {stats['avg_threshold']:.4f}\n")

        f.write("\n")

        # Change timestamps in single line
        if results['change_timestamps']:
            f.write("Change Timestamps: ")
            timestamps_str = [f"{ts:.1f}s" for ts in results['change_timestamps']]
            f.write(" ".join(timestamps_str))
            f.write("\n\n")

            # Detailed changes
            f.write("Detailed Changes:\n")
            f.write("-" * 20 + "\n")
            for i, change in enumerate(results['detected_changes']):
                f.write(f"{i + 1}. {change['start_time']:.1f}s-{change['end_time']:.1f}s ")
                f.write(f"(conf: {change['confidence']:.3f})\n")
        else:
            f.write("No changes detected.\n")

            # Debug info when no changes detected
            if results.get('confidence_scores'):
                f.write("\nDebug Information:\n")
                f.write("-" * 15 + "\n")
                conf_scores = results['confidence_scores']
                f.write(f"Confidence scores range: {min(conf_scores):.4f} to {max(conf_scores):.4f}\n")
                thresholds = results.get('thresholds', [])
                if thresholds:
                    f.write(f"Threshold range: {min(thresholds):.4f} to {max(thresholds):.4f}\n")

    print(f"Results saved to: {output_path}")


def main():
    args = parse_arguments()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load audio
    audio, sr = load_audio_file(args.audio_path, args.sample_rate)
    print(f"Audio loaded: {len(audio) / sr:.2f}s")

    # Create segments and extract features
    audio_segments = create_sliding_windows(audio, sr, args.window_length, args.stride_length)

    pann_model = load_pretrained_pann(
        checkpoint_path=args.pann_checkpoint,
        device=device,
        sample_rate=args.sample_rate,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527
    )

    pann_features = extract_pann_features(audio_segments, pann_model, device)

    # Run detection
    results = run_mcd_dd(pann_features, args, args.stride_length)

    # Save results
    audio_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    output_path = os.path.join(args.output_dir, f"mcd_dd_{audio_name}_results.txt")
    save_results(results, output_path, args)

    # Summary with timestamps in single line
    print(f"\nSummary:")
    print(f"Audio: {args.audio_path}")
    print(f"Detections: {results['total_detections']}")
    print(f"Time: {results['processing_time']:.2f}s")

    if results['change_timestamps']:
        timestamps_str = " ".join([f"{ts:.1f}s" for ts in results['change_timestamps']])
        print(f"Changes: {timestamps_str}")
    else:
        print("Changes: None")


if __name__ == "__main__":
    main()