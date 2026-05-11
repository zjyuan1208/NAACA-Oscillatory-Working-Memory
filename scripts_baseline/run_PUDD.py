import argparse
import os
import sys
import numpy as np
import torch
import time
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '/home/zhyuan/Desktop/PCD/baselines'))
from PUDD import PUDD, smooth_detections

sys.path.append(os.path.dirname(__file__))
from pann_backbone import load_pretrained_pann, create_sliding_windows, load_audio_file


def parse_arguments():
    parser = argparse.ArgumentParser(description='PUDD Baseline')

    parser.add_argument('--audio_path', type=str,
                        default='/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio/R0090_segment_ambisonics.wav',
                        help='Path to input audio file')
    parser.add_argument('--pann_checkpoint', type=str,
                        default='/home/zhyuan/Desktop/audioset_tagging_cnn/Cnn14_mAP=0.431.pth',
                        help='Path to PANN pretrained checkpoint')
    parser.add_argument('--output_dir', type=str, default='./baseline_results')

    parser.add_argument('--window_length', type=float, default=4.0)
    parser.add_argument('--stride_length', type=float, default=1.0)
    parser.add_argument('--sample_rate', type=int, default=32000)

    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--significance_threshold', type=float, default=0.001)
    parser.add_argument('--output_dim', type=int, default=2)

    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def extract_pann_features(audio_segments: List[np.ndarray], pann_model, device: str) -> torch.Tensor:
    print(f"Extracting PANN features from {len(audio_segments)} segments...")

    features = []
    for segment in audio_segments:
        segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = pann_model(segment_tensor)
            features.append(output['clipwise_output'].cpu())

    return torch.cat(features, dim=0)


def generate_labels(features: torch.Tensor, change_points: List[int] = None) -> torch.Tensor:
    T = features.shape[0]
    labels = torch.zeros(T, dtype=torch.long)

    # Use feature-based clustering to generate more realistic labels
    if change_points is None:
        # Use k-means on PANN features to detect natural clusters
        from sklearn.cluster import KMeans

        # Apply k-means clustering to find natural segments
        n_clusters = min(4, T // 5)  # At most 4 clusters, minimum 5 timesteps per cluster
        if n_clusters < 2:
            n_clusters = 2

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features.numpy())

        # Convert cluster labels to binary change detection
        # Find change points where cluster labels change
        change_points = []
        for t in range(1, T):
            if cluster_labels[t] != cluster_labels[t - 1]:
                change_points.append(t)

    # Convert change points to binary labels
    current_label = 0
    for cp in change_points:
        if cp < T:
            labels[cp:] = 1 - current_label
            current_label = 1 - current_label

    return labels


def run_pudd(pann_features: torch.Tensor, args, stride_length: float) -> Dict:
    print("Running PUDD detection...")

    detector = PUDD(
        significance_threshold=args.significance_threshold,
        k=args.k,
        device=args.device
    )

    T = pann_features.shape[0]
    print(f"Audio length: {T} timesteps")
    print(f"Online processing: 1 timestep = {stride_length}s")

    start_time = time.time()

    detections = []
    confidence_scores = []
    timestamps = []
    detected_changes = []
    pvalues = []

    # Use a more conservative labeling strategy
    for t in range(T):
        features = pann_features[t:t + 1].numpy()

        # More conservative label switching based on larger feature changes
        if t == 0:
            current_label = 0
            prev_feature_norm = np.linalg.norm(features)
        else:
            current_feature_norm = np.linalg.norm(features)
            # Increase threshold for label switching to reduce sensitivity
            magnitude_change = abs(current_feature_norm - prev_feature_norm) / (prev_feature_norm + 1e-8)

            # Only switch label for significant changes (increased from 0.1 to 0.3)
            if magnitude_change > 0.3:
                current_label = 1 - current_label

            prev_feature_norm = current_feature_norm

        current_labels = np.array([current_label])

        action = 0
        if t > 0 and len(detections) > 0 and detections[-1]:
            action = 1

        pvalue_log, is_drift, acc = detector.step(features, current_labels, action)
        pvalue = 10 ** (pvalue_log) if pvalue_log > -100 else 1e-100

        detections.append(is_drift)
        confidence = max(0.0, -pvalue_log)
        confidence_scores.append(confidence)
        pvalues.append(pvalue)

        time_sec = t * stride_length
        timestamps.append(time_sec)

        if args.verbose:
            print(f"t={t}, label={current_label}, acc={acc:.3f}, pvalue={pvalue:.6f}, drift={is_drift}")

        if is_drift:
            detected_changes.append({
                'start': t,
                'end': t + 1,
                'start_time': time_sec,
                'end_time': time_sec + stride_length,
                'pvalue': pvalue,
                'confidence': confidence,
                'timestep': t
            })

    processing_time = time.time() - start_time

    results = {
        'detections': detections,
        'detected_changes': detected_changes,
        'timestamps': timestamps,
        'confidence_scores': confidence_scores,
        'pvalues': pvalues,
        'total_detections': len(detected_changes),
        'total_timesteps': T,
        'processing_time': processing_time
    }

    # Apply more aggressive post-processing filters
    change_timestamps = [change['start_time'] for change in results['detected_changes']]
    detected_changes = results['detected_changes']

    if len(change_timestamps) > 1:
        filtered_timestamps = []
        filtered_changes = []
        min_interval = 8.0  # Increased from 3.0 to 8.0 for more conservative filtering

        for i, (timestamp, change) in enumerate(zip(change_timestamps, detected_changes)):
            if i == 0 or timestamp - filtered_timestamps[-1] >= min_interval:
                filtered_timestamps.append(timestamp)
                filtered_changes.append(change)

        change_timestamps = filtered_timestamps
        detected_changes = filtered_changes

    if detected_changes:
        conf_threshold = 1.0  # Increased from 0.1 to 1.0 for higher confidence requirement
        filtered_timestamps = []
        filtered_changes = []

        for timestamp, change in zip(change_timestamps, detected_changes):
            if change['confidence'] > conf_threshold:
                filtered_timestamps.append(timestamp)
                filtered_changes.append(change)

        change_timestamps = filtered_timestamps
        detected_changes = filtered_changes

    results['change_timestamps'] = change_timestamps
    results['detected_changes'] = detected_changes
    results['total_detections'] = len(detected_changes)

    confidence_scores = results['confidence_scores']
    pvalues = results['pvalues']

    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
    max_confidence = max(confidence_scores) if confidence_scores else 0.0
    avg_pvalue = np.mean(pvalues) if pvalues else 1.0

    results['statistics'] = {
        'avg_confidence': avg_confidence,
        'max_confidence': max_confidence,
        'avg_pvalue': avg_pvalue
    }

    results['window_params'] = {
        'win_size': 1,
        'slide': 1,
        'audio_length': T
    }

    results['total_windows'] = T

    print(f"\nDetection completed:")
    print(f"Total timesteps processed: {T}")
    print(f"Raw detections: {sum(results['detections'])}")
    print(f"Filtered detections: {len(detected_changes)}")
    print(f"Average pvalue: {avg_pvalue:.6f}")

    return results


def save_results(results: Dict, output_path: str, args):
    np.savez(output_path.replace('.txt', '.npz'), **results)

    with open(output_path, 'w') as f:
        f.write("PUDD Baseline Results\n")
        f.write("=" * 30 + "\n\n")

        f.write("Basic Statistics:\n")
        f.write(f"Total detections: {results['total_detections']}\n")
        f.write(f"Total windows: {results['total_windows']}\n")
        f.write(f"Processing time: {results['processing_time']:.2f}s\n")

        wp = results['window_params']
        f.write(f"Audio length: {wp['audio_length']} timesteps\n")
        f.write(f"Window size: {wp['win_size']}\n")
        f.write(f"Slide size: {wp['slide']}\n")

        if 'statistics' in results:
            stats = results['statistics']
            f.write(f"Average confidence: {stats['avg_confidence']:.4f}\n")
            f.write(f"Max confidence: {stats['max_confidence']:.4f}\n")
            f.write(f"Average pvalue: {stats['avg_pvalue']:.6f}\n")

        f.write("\n")

        if results['change_timestamps']:
            f.write("Change Timestamps: ")
            timestamps_str = [f"{ts:.1f}s" for ts in results['change_timestamps']]
            f.write(" ".join(timestamps_str))
            f.write("\n\n")

            f.write("Detailed Changes:\n")
            f.write("-" * 20 + "\n")
            for i, change in enumerate(results['detected_changes']):
                f.write(f"{i + 1}. {change['start_time']:.1f}s-{change['end_time']:.1f}s ")
                f.write(f"(pvalue: {change['pvalue']:.6f})\n")
        else:
            f.write("No changes detected.\n")

            if results.get('confidence_scores'):
                f.write("\nDebug Information:\n")
                f.write("-" * 15 + "\n")
                conf_scores = results['confidence_scores']
                f.write(f"Confidence scores range: {min(conf_scores):.4f} to {max(conf_scores):.4f}\n")
                pvalues = results.get('pvalues', [])
                if pvalues:
                    f.write(f"P-value range: {min(pvalues):.6f} to {max(pvalues):.6f}\n")

    print(f"Results saved to: {output_path}")


def main():
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    audio, sr = load_audio_file(args.audio_path, args.sample_rate)
    print(f"Audio loaded: {len(audio) / sr:.2f}s")

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

    results = run_pudd(pann_features, args, args.stride_length)

    audio_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    output_path = os.path.join(args.output_dir, f"pudd_{audio_name}_results.txt")
    save_results(results, output_path, args)

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