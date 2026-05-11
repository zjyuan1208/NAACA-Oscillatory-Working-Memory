import argparse
import os
import sys
import numpy as np
import torch
import time
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

# Import modules - simplified approach
sys.path.append(os.path.join(os.path.dirname(__file__), '/home/zhyuan/Desktop/PCD/baselines'))
import kmeans

# Use the classes from the module
AdaptiveOutlierDetector = kmeans.AdaptiveOutlierDetector
OptimalKMeansDetector = kmeans.OptimalKMeansDetector

sys.path.append(os.path.dirname(__file__))
from pann_backbone import load_pretrained_pann, create_sliding_windows, load_audio_file


def parse_arguments():
    parser = argparse.ArgumentParser(description='from the first frame.')

    # Required (with defaults)
    parser.add_argument('--audio_path', type=str,
                        default='/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio/R0056_segment_ambisonics.wav',
                        help='Path to input audio file')
    parser.add_argument('--pann_checkpoint', type=str,
                        default='/home/zhyuan/Desktop/audioset_tagging_cnn/Cnn14_mAP=0.431.pth',
                        help='Path to PANN pretrained checkpoint')
    parser.add_argument('--output_dir', type=str, default='./baseline_results')

    # Audio parameters (matching MCD-DD)
    parser.add_argument('--window_length', type=float, default=4.0)
    parser.add_argument('--stride_length', type=float, default=1.0)
    parser.add_argument('--sample_rate', type=int, default=32000)

    # Parameters.
    parser.add_argument('--k_max', type=int, default=3, # Reduce.
                        help='cluster.')
    parser.add_argument('--distance_metric', type=str, default='euclidean', # distance.
                        choices=['cosine', 'euclidean', 'mahalanobis'],
                        help='distance.')
    parser.add_argument('--initial_threshold_factor', type=float, default=3.0, # Implementation note.
                        help='Threshold.')
    parser.add_argument('--threshold_adaptation_rate', type=float, default=0.03, # Reduce.
                        help='Threshold.')
    parser.add_argument('--consecutive_threshold', type=int, default=4, # Implementation note.
                        help='consecutive outliers.')
    parser.add_argument('--cluster_update_interval', type=int, default=25, # Implementation note.
                        help='cluster.')
    parser.add_argument('--min_samples_for_mahalanobis', type=int, default=20,
                        help='distance.')
    parser.add_argument('--stability_window', type=int, default=10,
                        help='cluster.')

    # Parameters.
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Parameters.')
    parser.add_argument('--init_frames', type=int, default=1,
                        help='Parameters.')
    parser.add_argument('--learning_frames', type=int, default=1,
                        help='Parameters.')
    parser.add_argument('--percentile_threshold', type=float, default=95.0,
                        help='Parameters.')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Parameters.')
    parser.add_argument('--buffer_size', type=int, default=15,
                        help='Parameters.')
    parser.add_argument('--batch_size', type=int, default=7,
                        help='Parameters.')
    parser.add_argument('--change_threshold', type=float, default=0.5,
                        help='Parameters.')
    parser.add_argument('--min_persistence', type=int, default=2,
                        help='Post-processing.')
    parser.add_argument('--use_minibatch', action='store_true', default=True,
                        help='Parameters.')
    parser.add_argument('--max_iter', type=int, default=10,
                        help='Parameters.')

    # Auto-tuning
    parser.add_argument('--use_auto_tune', action='store_true',
                        help='Threshold.')
    parser.add_argument('--tune_sample_ratio', type=float, default=0.2,
                        help='Parameters.')

    # Other parameters
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def extract_pann_features(audio_segments: List[np.ndarray], pann_model, device: str, args) -> torch.Tensor:
    """Extract PANN features"""
    print(f"Extracting PANN features from {len(audio_segments)} segments...")

    features = []
    for i, segment in enumerate(audio_segments):
        if args.verbose and (i + 1) % 100 == 0:
            print(f"Processing segment {i + 1}/{len(audio_segments)}")

        segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = pann_model(segment_tensor)
            features.append(output['clipwise_output'].cpu())

    return torch.cat(features, dim=0)


def run_immediate_adaptive_detection(pann_features: torch.Tensor, args, stride_length: float) -> Dict:
    """ from the first frame. """
    print("from the first frame.")

    # Parameters.
    k_max = args.k_max if hasattr(args, 'k_max') else args.n_clusters

    # Auto-tune parameters if requested
    detector = None
    if args.use_auto_tune:
        print("Parameters.")
        tuner = OptimalKMeansDetector(input_dim=527, device=args.device)

        # Use sample of data for tuning
        T = pann_features.shape[0]
        sample_size = int(T * args.tune_sample_ratio)
        sample_indices = np.random.choice(T, min(sample_size, T), replace=False)
        sample_data = pann_features[sample_indices]

        tuner.tune_parameters(sample_data)
        detector = tuner.create_detector()
    else:
        # Use manual parameters
        detector = AdaptiveOutlierDetector(
            input_dim=527,
            k_max=k_max,
            distance_metric=args.distance_metric,
            initial_threshold_factor=args.initial_threshold_factor,
            threshold_adaptation_rate=args.threshold_adaptation_rate,
            consecutive_threshold=args.consecutive_threshold,
            cluster_update_interval=args.cluster_update_interval,
            min_samples_for_mahalanobis=args.min_samples_for_mahalanobis,
            stability_window=args.stability_window
        )

    print(f"Parameters.")
    print(f"cluster.")
    print(f"distance.")
    print(f"Threshold.")
    print(f"Threshold.")
    print(f"Implementation note.")
    print(f"cluster.")
    print(f"distance.")
    print(f"stability window.")

    # Run detection
    start_time = time.time()
    results = detector.batch_detect(pann_features)
    processing_time = time.time() - start_time

    # Convert positions to timestamps (direct time step to time)
    change_timestamps = []
    detected_changes = []

    for change in results['detected_changes']:
        # Convert time step to timestamp
        timestamp = change['position'] * stride_length
        change_timestamps.append(timestamp)

        # Update change info with correct field names
        change_info = change.copy()
        change_info['timestamp'] = timestamp
        change_info['start_time'] = change['position'] * stride_length
        change_info['end_time'] = (change['position'] + 1) * stride_length
        detected_changes.append(change_info)

    # Implementation note.
    if len(change_timestamps) > 1:
        filtered_timestamps = []
        filtered_changes = []
        min_interval = 5.0 # Implementation note.

        for timestamp, change in zip(change_timestamps, detected_changes):
            if not filtered_timestamps or timestamp - filtered_timestamps[-1] >= min_interval:
                filtered_timestamps.append(timestamp)
                filtered_changes.append(change)

        change_timestamps = filtered_timestamps
        detected_changes = filtered_changes

    # Calculate statistics
    confidence_scores = results['confidence_scores']
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
    max_confidence = max(confidence_scores) if confidence_scores else 0.0
    detection_density = len(detected_changes) / (pann_features.shape[0] * stride_length / 60)  # per minute

    print(f"from the first frame.")
    print(f"Implementation note.")
    print(f"Detection.")
    print(f"Detection.")
    print(f"Implementation note.")

    # Implementation note.
    if 'phase_statistics' in results:
        ps = results['phase_statistics']
        print(f"Implementation note.")
        print(f"cluster.")
        print(f"cluster.")
        print(f"Threshold.")
        print(f"normal sample.")
        print(f"Threshold.")

    print(f"Detection.")

    # Update results
    results.update({
        'change_timestamps': change_timestamps,
        'detected_changes': detected_changes,
        'processing_time': processing_time,
        'total_detections': len(detected_changes),
        'statistics': {
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'detection_density': detection_density,
            'total_audio_duration': pann_features.shape[0] * stride_length,
            'total_consistency_cost': results.get('total_consistency', 0)
        }
    })

    return results


def save_results(results: Dict, output_path: str, args):
    """Save detection results"""
    # Save numpy data
    np.savez(output_path.replace('.txt', '.npz'), **results)

    # Save text summary
    with open(output_path, 'w') as f:
        f.write("from the first frame.")
        f.write("=" * 40 + "\n\n")

        # Parameters
        f.write("Parameters.")
        f.write(f"from the first frame.")
        f.write(f"Audio.")
        f.write(f"Implementation note.")
        f.write(f"Implementation note.")
        f.write(f"cluster.")
        f.write(f"distance.")
        f.write(f"Threshold.")
        f.write(f"Threshold.")
        f.write(f"Implementation note.")
        f.write(f"cluster.")
        f.write(f"Tune.")
        f.write("\n")

        # Implementation note.
        if 'phase_statistics' in results:
            ps = results['phase_statistics']
            f.write("Implementation note.")
            f.write(f"Implementation note.")
            f.write(f"cluster.")
            f.write(f"cluster.")
            f.write(f"Threshold.")
            f.write(f"normal sample.")
            f.write(f"distance.")
            f.write(f"Threshold.")
            f.write("\n")

        # Basic statistics
        f.write("Detection.")
        f.write(f"Detection.")
        f.write(f"Implementation note.")
        f.write(f"Implementation note.")

        # Detection statistics
        if 'statistics' in results:
            stats = results['statistics']
            f.write(f"Implementation note.")
            f.write(f"Implementation note.")
            f.write(f"Detection.")
            f.write(f"Audio.")

        f.write("\n")

        # Change timestamps
        if results['change_timestamps']:
            f.write("Implementation note.")
            timestamps_str = [f"{ts:.1f}s" for ts in results['change_timestamps']]
            f.write(" ".join(timestamps_str))
            f.write("\n\n")

            # Detailed changes
            f.write("Implementation note.")
            f.write("-" * 20 + "\n")
            for i, change in enumerate(results['detected_changes']):
                f.write(f"{i + 1}. {change['timestamp']:.1f}s ")
                f.write(f"Implementation note.")
                f.write(f"distance.")
                f.write(f"Threshold.")
                f.write(f"consecutive outliers.")
                f.write(f"cluster.")
        else:
            f.write("Detection.")

            # Debug info when no changes detected
            if results.get('confidence_scores'):
                f.write("Implementation note.")
                f.write("-" * 15 + "\n")
                conf_scores = results['confidence_scores']
                f.write(f"Implementation note.")
                f.write(f"Implementation note.")

    print(f"Results saved to: {output_path}")


def main():
    args = parse_arguments()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load audio
    print(f"Loading audio from: {args.audio_path}")
    try:
        audio, sr = load_audio_file(args.audio_path, args.sample_rate)
        print(f"Audio loaded: {len(audio) / sr:.2f}s at {sr}Hz")
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    # Create sliding windows
    print(f"Creating sliding windows (length: {args.window_length}s, stride: {args.stride_length}s)")
    audio_segments = create_sliding_windows(audio, sr, args.window_length, args.stride_length)
    print(f"Created {len(audio_segments)} audio segments")

    # Load PANN model
    print("Loading PANN model...")
    try:
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
        print("PANN model loaded successfully")
    except Exception as e:
        print(f"Error loading PANN model: {e}")
        return

    # Extract PANN features
    try:
        pann_features = extract_pann_features(audio_segments, pann_model, device, args)
        print(f"PANN features extracted: {pann_features.shape}")
    except Exception as e:
        print(f"Error extracting PANN features: {e}")
        return

    # Run immediate adaptive detection
    try:
        results = run_immediate_adaptive_detection(pann_features, args, args.stride_length)
    except Exception as e:
        print(f"Error running immediate adaptive detection: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save results
    audio_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    output_path = os.path.join(args.output_dir, f"immediate_adaptive_{audio_name}_results.txt")

    try:
        save_results(results, output_path, args)
    except Exception as e:
        print(f"Error saving results: {e}")
        return

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"from the first frame.")
    print(f"{'=' * 80}")
    print(f"Audio.")
    print(f"Implementation note.")
    print(f"Implementation note.")
    print(f"from the first frame.")
    print(f"distance.")
    print(f"Detection.")
    print(f"Implementation note.")

    # Implementation note.
    if 'phase_statistics' in results:
        ps = results['phase_statistics']
        print(f"Implementation note.")
        print(f"cluster.")
        print(f"Threshold.")
        print(f"normal sample.")

    if results['change_timestamps']:
        timestamps_str = " ".join([f"{ts:.1f}s" for ts in results['change_timestamps']])
        print(f"Implementation note.")

        # Calculate intervals between changes
        if len(results['change_timestamps']) > 1:
            intervals = []
            for i in range(1, len(results['change_timestamps'])):
                interval = results['change_timestamps'][i] - results['change_timestamps'][i - 1]
                intervals.append(interval)
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            min_interval = min(intervals)
            max_interval = max(intervals)
            print(
                f"Implementation note.")
    else:
        print("Detection.")

    # Performance metrics
    if 'statistics' in results:
        stats = results['statistics']
        print(f"Detection.")
        if 'avg_confidence' in stats:
            print(f"Implementation note.")
            print(f"Implementation note.")

    print(f"{'=' * 80}")

    print("Implementation note.")
    print("Implementation note.")
    print("cluster.")
    print("default.")
    print("Adaptive threshold.")
    print("cluster.")
    print("Initialize.")


if __name__ == "__main__":
    main()