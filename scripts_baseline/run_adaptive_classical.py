import argparse
import os
import sys
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

# Import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '/home/zhyuan/Desktop/PCD/baselines'))
from adaptive_classical_baseline import run_adaptive_classical_detection, smooth_detections

sys.path.append(os.path.dirname(__file__))
from pann_backbone import load_pretrained_pann, create_sliding_windows, load_audio_file


def parse_arguments():
    parser = argparse.ArgumentParser(description='Adaptive Classical Drift Detection Baseline')

    # Required (with defaults)
    parser.add_argument('--audio_path', type=str,
                        default='/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio/R0056_segment_ambisonics.wav',
                        help='Path to input audio file')
    parser.add_argument('--pann_checkpoint', type=str,
                        default='/home/zhyuan/Desktop/audioset_tagging_cnn/Cnn14_mAP=0.431.pth',
                        help='Path to PANN pretrained checkpoint')
    parser.add_argument('--output_dir', type=str, default='./adaptive_baseline_results')

    # Method selection
    parser.add_argument('--method', type=str,
                        choices=['adwin', 'pagehinkley', 'kswin', 'hddm', 'all'],
                        default='adwin',
                        help='Adaptive detection method to use')

    # Audio parameters
    parser.add_argument('--window_length', type=float, default=4.0)
    parser.add_argument('--stride_length', type=float, default=1.0)
    parser.add_argument('--sample_rate', type=int, default=32000)

    # Feature reduction
    parser.add_argument('--feature_reduction', type=str,
                        choices=['mean', 'weighted_mean', 'entropy', 'max', 'dominant_category_prob', 'top_k_sum'],
                        default='weighted_mean',
                        help='Method to reduce 527-dim PANN features to scalar')

    # Adaptive parameters (common)
    parser.add_argument('--target_detection_rate', type=float, default=0.05,
                        help='Target detection rate for adaptive algorithms')

    # ADWIN adaptive parameters
    parser.add_argument('--adwin_initial_delta', type=float, default=0.002,
                        help='Initial ADWIN delta parameter')
    parser.add_argument('--adwin_adaptation_rate', type=float, default=0.02,
                        help='ADWIN adaptation rate')

    # Page-Hinkley adaptive parameters
    parser.add_argument('--ph_initial_threshold', type=float, default=50,
                        help='Initial Page-Hinkley threshold')
    parser.add_argument('--ph_adaptation_rate', type=float, default=0.05,
                        help='Page-Hinkley adaptation rate')
    parser.add_argument('--ph_min_instances', type=int, default=30,
                        help='Page-Hinkley minimum instances')
    parser.add_argument('--ph_delta', type=float, default=0.005,
                        help='Page-Hinkley delta parameter (fixed)')

    # KSWIN adaptive parameters
    parser.add_argument('--kswin_initial_alpha', type=float, default=0.005,
                        help='Initial KSWIN alpha parameter')
    parser.add_argument('--kswin_adaptation_rate', type=float, default=0.03,
                        help='KSWIN adaptation rate')
    parser.add_argument('--kswin_window_size', type=int, default=100,
                        help='KSWIN window size (fixed)')

    # HDDM adaptive parameters
    parser.add_argument('--hddm_initial_drift_conf', type=float, default=0.001,
                        help='Initial HDDM drift confidence')
    parser.add_argument('--hddm_adaptation_rate', type=float, default=0.04,
                        help='HDDM adaptation rate')

    # Post-processing
    parser.add_argument('--apply_smoothing', action='store_true',
                        help='Apply temporal smoothing to reduce false positives')
    parser.add_argument('--min_persistence', type=int, default=2,
                        help='Minimum persistence for smoothing')

    # Visualization
    parser.add_argument('--save_plots', action='store_true',
                        help='Save adaptation plots')

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


def get_adaptive_detector_kwargs(args, method: str) -> Dict:
    """Get adaptive detector-specific kwargs based on method and args"""

    base_kwargs = {
        'feature_reduction': args.feature_reduction,
        'target_detection_rate': args.target_detection_rate
    }

    if method == 'adwin':
        return {
            **base_kwargs,
            'initial_delta': args.adwin_initial_delta,
            'adaptation_rate': args.adwin_adaptation_rate
        }
    elif method == 'pagehinkley':
        return {
            **base_kwargs,
            'initial_threshold': args.ph_initial_threshold,
            'adaptation_rate': args.ph_adaptation_rate,
            'min_instances': args.ph_min_instances,
            'delta': args.ph_delta
        }
    elif method == 'kswin':
        return {
            **base_kwargs,
            'initial_alpha': args.kswin_initial_alpha,
            'adaptation_rate': args.kswin_adaptation_rate,
            'window_size': args.kswin_window_size
        }
    elif method == 'hddm':
        return {
            **base_kwargs,
            'initial_drift_conf': args.hddm_initial_drift_conf,
            'adaptation_rate': args.hddm_adaptation_rate
        }
    else:
        return base_kwargs


def plot_adaptation_results(results: Dict, output_dir: str, audio_name: str):
    """Plot adaptation results for visualization"""
    method = results['method']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{method} Adaptive Detection Results - {audio_name}', fontsize=16)

    timestamps = np.arange(len(results['parameter_evolution'])) * 1.0  # Assuming 1s stride

    # Plot 1: Parameter evolution
    axes[0, 0].plot(timestamps, results['parameter_evolution'], 'b-', linewidth=2)
    axes[0, 0].set_title('Parameter Evolution Over Time')
    axes[0, 0].set_xlabel('Time (s)')

    if 'ADWIN' in method:
        axes[0, 0].set_ylabel('Delta')
    elif 'PageHinkley' in method:
        axes[0, 0].set_ylabel('Threshold')
    elif 'KSWIN' in method:
        axes[0, 0].set_ylabel('Alpha')
    elif 'HDDM' in method:
        axes[0, 0].set_ylabel('Drift Confidence')

    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Detection rate evolution
    axes[0, 1].plot(timestamps, results['detection_rates'], 'g-', linewidth=2, label='Actual Rate')
    target_rate = results['adaptation_statistics']['target_detection_rate']
    axes[0, 1].axhline(y=target_rate, color='r', linestyle='--', linewidth=2, label=f'Target Rate ({target_rate:.3f})')
    axes[0, 1].set_title('Detection Rate Evolution')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Detection Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Confidence scores and detections
    axes[1, 0].plot(timestamps, results['confidence_scores'], 'c-', alpha=0.7, label='Confidence')

    # Mark detections
    detection_times = [i for i, d in enumerate(results['detections']) if d]
    if detection_times:
        axes[1, 0].scatter([timestamps[i] for i in detection_times],
                           [results['confidence_scores'][i] for i in detection_times],
                           color='red', s=50, alpha=0.8, label='Detections')

    axes[1, 0].set_title('Confidence Scores and Detections')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Confidence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Scalar feature values
    axes[1, 1].plot(timestamps, results['scalar_values'], 'm-', alpha=0.7, linewidth=1)
    axes[1, 1].set_title('Scalar Feature Values')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Feature Value')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, f"{method.lower()}_{audio_name}_adaptation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Adaptation plot saved to: {plot_path}")


def run_single_adaptive_method(pann_features: torch.Tensor, method: str, args) -> Dict:
    """Run adaptive detection for a single method"""
    print(f"\n{'=' * 60}")
    print(f"Running Adaptive {method.upper()} Detection")
    print(f"{'=' * 60}")

    # Get method-specific parameters
    detector_kwargs = get_adaptive_detector_kwargs(args, method)

    if args.verbose:
        print(f"Method: Adaptive {method.upper()}")
        print(f"Feature reduction: {args.feature_reduction}")
        print(f"Target detection rate: {args.target_detection_rate}")
        print(f"Detector parameters: {detector_kwargs}")

    start_time = time.time()

    # Run adaptive detection
    results = run_adaptive_classical_detection(
        pann_features=pann_features,
        method=method,
        stride_length=args.stride_length,
        **detector_kwargs
    )

    processing_time = time.time() - start_time
    results['processing_time'] = processing_time

    # Apply smoothing if requested
    if args.apply_smoothing and results['detections']:
        original_detections = results['detections'].copy()
        smoothed_detections = smooth_detections(results['detections'], args.min_persistence)

        # Recalculate change timestamps based on smoothed detections
        smoothed_changes = []
        smoothed_timestamps = []

        for i, (original, smoothed) in enumerate(zip(original_detections, smoothed_detections)):
            if smoothed and original:
                timestamp_sec = i * args.stride_length
                smoothed_timestamps.append(timestamp_sec)
                smoothed_changes.append({
                    'timestamp_idx': i,
                    'timestamp_sec': timestamp_sec,
                    'confidence': results['confidence_scores'][i],
                    'scalar_value': results['scalar_values'][i] if i < len(results['scalar_values']) else 0.0,
                    'parameter_value': results['parameter_evolution'][i] if i < len(
                        results['parameter_evolution']) else 0.0,
                    'smoothed': True
                })

        results['detections'] = smoothed_detections
        results['detected_changes'] = smoothed_changes
        results['change_timestamps'] = smoothed_timestamps
        results['total_detections'] = len(smoothed_changes)
        results['smoothing_applied'] = True

        print(f"Smoothing applied: {sum(original_detections)} -> {sum(smoothed_detections)} detections")

    return results


def save_adaptive_results(results: Dict, output_path: str, args):
    """Save adaptive results with adaptation information"""
    method = results['method']

    # Save numpy data
    np.savez(output_path.replace('.txt', '.npz'), **results)

    # Save detailed text summary
    with open(output_path, 'w') as f:
        f.write(f"{method} Adaptive Baseline Results\n")
        f.write("=" * 50 + "\n\n")

        # Basic stats
        f.write("Basic Statistics:\n")
        f.write(f"Method: {method}\n")
        f.write(f"Total detections: {results['total_detections']}\n")
        f.write(f"Total timesteps: {results['total_timesteps']}\n")
        f.write(f"Final detection rate: {results['final_detection_rate']:.4f}\n")
        f.write(f"Average detection rate: {results['avg_detection_rate']:.4f}\n")
        f.write(f"Processing time: {results['processing_time']:.2f}s\n")

        # Adaptation statistics
        if 'adaptation_statistics' in results:
            adapt_stats = results['adaptation_statistics']
            f.write(f"\nAdaptation Statistics:\n")
            f.write(f"Target detection rate: {adapt_stats['target_detection_rate']:.4f}\n")
            f.write(f"Adaptation rate: {adapt_stats['adaptation_rate']:.4f}\n")
            f.write(f"Total adaptations: {adapt_stats['adaptation_count']}\n")
            f.write(f"Initial parameter: {adapt_stats['initial_parameter']:.6f}\n")
            f.write(f"Final parameter: {adapt_stats['final_parameter']:.6f}\n")
            f.write(f"Parameter change ratio: {adapt_stats['parameter_change_ratio']:.3f}\n")

        # Detector parameters
        if 'detector_params' in results:
            f.write(f"\nDetector Parameters:\n")
            for key, value in results['detector_params'].items():
                f.write(f"{key}: {value}\n")

        # Detection statistics
        if 'statistics' in results:
            stats = results['statistics']
            f.write(f"\nDetection Statistics:\n")
            f.write(f"Average confidence: {stats['avg_confidence']:.4f}\n")
            f.write(f"Max confidence: {stats['max_confidence']:.4f}\n")
            f.write(f"Min confidence: {stats['min_confidence']:.4f}\n")
            f.write(f"Std confidence: {stats['std_confidence']:.4f}\n")

        # Smoothing info
        if results.get('smoothing_applied', False):
            f.write(f"\nSmoothing applied with min_persistence = {args.min_persistence}\n")

        f.write("\n")

        # Change timestamps in single line
        if results['change_timestamps']:
            f.write("Change Timestamps: ")
            timestamps_str = [f"{ts:.1f}s" for ts in results['change_timestamps']]
            f.write(" ".join(timestamps_str))
            f.write("\n\n")

            # Detailed changes with adaptation info
            f.write("Detailed Changes:\n")
            f.write("-" * 30 + "\n")
            for i, change in enumerate(results['detected_changes']):
                f.write(f"{i + 1}. {change['timestamp_sec']:.1f}s ")
                f.write(f"(conf: {change['confidence']:.3f}, ")
                f.write(f"param: {change.get('parameter_value', 0.0):.4f}")
                if 'detection_rate_at_time' in change:
                    f.write(f", rate: {change['detection_rate_at_time']:.3f}")
                if change.get('smoothed', False):
                    f.write(" [smoothed]")
                f.write(")\n")
        else:
            f.write("No changes detected.\n")

            # Enhanced debug info for adaptive methods
            if results.get('confidence_scores'):
                f.write("\nDebug Information:\n")
                f.write("-" * 20 + "\n")
                conf_scores = results['confidence_scores']
                f.write(f"Confidence scores range: {min(conf_scores):.4f} to {max(conf_scores):.4f}\n")

                param_evolution = results.get('parameter_evolution', [])
                if param_evolution:
                    f.write(f"Parameter evolution range: {min(param_evolution):.6f} to {max(param_evolution):.6f}\n")

                detection_rates = results.get('detection_rates', [])
                if detection_rates:
                    f.write(f"Detection rate evolution: {min(detection_rates):.4f} to {max(detection_rates):.4f}\n")

    print(f"Adaptive results saved to: {output_path}")


def run_all_adaptive_methods(pann_features: torch.Tensor, args) -> Dict:
    """Run all adaptive methods and compare results"""
    methods = ['adwin', 'pagehinkley', 'kswin', 'hddm']
    all_results = {}

    print(f"\n{'=' * 70}")
    print(f"Running All Adaptive Classical Methods Comparison")
    print(f"{'=' * 70}")

    for method in methods:
        try:
            results = run_single_adaptive_method(pann_features, method, args)
            all_results[method] = results
        except Exception as e:
            print(f"Error running adaptive {method}: {e}")
            all_results[method] = {'error': str(e)}

    # Summary comparison
    print(f"\n{'=' * 70}")
    print("ADAPTIVE METHODS COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"{'Method':<20} {'Detections':<12} {'Final Rate':<12} {'Adaptations':<12} {'Param Change':<12} {'Time(s)':<8}")
    print("-" * 80)

    for method, results in all_results.items():
        if 'error' in results:
            print(f"{method.upper():<20} {'ERROR':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<8}")
        else:
            detections = results['total_detections']
            final_rate = results['final_detection_rate']
            adaptations = results['adaptation_statistics']['adaptation_count']
            param_change = results['adaptation_statistics']['parameter_change_ratio']
            proc_time = results['processing_time']
            print(
                f"{method.upper():<20} {detections:<12} {final_rate:<12.3f} {adaptations:<12} {param_change:<12.2f} {proc_time:<8.2f}")

    return all_results


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
    print(f"PANN features extracted: {pann_features.shape}")

    # Run adaptive detection based on method selection
    audio_name = os.path.splitext(os.path.basename(args.audio_path))[0]

    if args.method == 'all':
        # Run all adaptive methods
        all_results = run_all_adaptive_methods(pann_features, args)

        # Save individual results and plots
        for method, results in all_results.items():
            if 'error' not in results:
                output_path = os.path.join(args.output_dir, f"adaptive_{method}_{audio_name}_results.txt")
                save_adaptive_results(results, output_path, args)

                # Save adaptation plots
                if args.save_plots:
                    plot_adaptation_results(results, args.output_dir, audio_name)

        # Save comparison summary
        comparison_path = os.path.join(args.output_dir, f"adaptive_comparison_{audio_name}_results.txt")
        with open(comparison_path, 'w') as f:
            f.write("Adaptive Classical Methods Comparison Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Audio: {args.audio_path}\n")
            f.write(f"Feature reduction: {args.feature_reduction}\n")
            f.write(f"Target detection rate: {args.target_detection_rate}\n")
            f.write(f"Smoothing applied: {args.apply_smoothing}\n\n")

            f.write(
                f"{'Method':<20} {'Detections':<12} {'Final Rate':<12} {'Adaptations':<12} {'Param Change':<12} {'Time(s)':<8} {'Timestamps'}\n")
            f.write("-" * 100 + "\n")

            for method, results in all_results.items():
                if 'error' in results:
                    f.write(
                        f"{method.upper():<20} {'ERROR':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<8} {results['error']}\n")
                else:
                    detections = results['total_detections']
                    final_rate = results['final_detection_rate']
                    adaptations = results['adaptation_statistics']['adaptation_count']
                    param_change = results['adaptation_statistics']['parameter_change_ratio']
                    proc_time = results['processing_time']
                    timestamps = " ".join([f"{ts:.1f}s" for ts in results['change_timestamps']]) if results[
                        'change_timestamps'] else "None"
                    f.write(
                        f"{method.upper():<20} {detections:<12} {final_rate:<12.3f} {adaptations:<12} {param_change:<12.2f} {proc_time:<8.2f} {timestamps}\n")

        print(f"\nAdaptive comparison summary saved to: {comparison_path}")

        # Print final summary
        print(f"\nFinal Adaptive Summary:")
        print(f"Audio: {args.audio_path}")
        print(f"Target detection rate: {args.target_detection_rate}")
        for method, results in all_results.items():
            if 'error' not in results:
                timestamps_str = " ".join([f"{ts:.1f}s" for ts in results['change_timestamps']]) if results[
                    'change_timestamps'] else "None"
                adaptations = results['adaptation_statistics']['adaptation_count']
                param_change = results['adaptation_statistics']['parameter_change_ratio']
                print(f"Adaptive {method.upper()}: {results['total_detections']} detections, "
                      f"{adaptations} adaptations, {param_change:.2f}x param change - {timestamps_str}")
            else:
                print(f"Adaptive {method.upper()}: ERROR - {results['error']}")

    else:
        # Run single adaptive method
        results = run_single_adaptive_method(pann_features, args.method, args)

        # Save results
        output_path = os.path.join(args.output_dir, f"adaptive_{args.method}_{audio_name}_results.txt")
        save_adaptive_results(results, output_path, args)

        # Save adaptation plots
        if args.save_plots:
            plot_adaptation_results(results, args.output_dir, audio_name)

        # Summary with adaptation information
        print(f"\nAdaptive Summary:")
        print(f"Audio: {args.audio_path}")
        print(f"Method: Adaptive {args.method.upper()}")
        print(f"Detections: {results['total_detections']}")
        print(f"Target detection rate: {results['adaptation_statistics']['target_detection_rate']:.3f}")
        print(f"Final detection rate: {results['final_detection_rate']:.3f}")
        print(f"Parameter adaptations: {results['adaptation_statistics']['adaptation_count']}")
        print(f"Parameter change: {results['adaptation_statistics']['initial_parameter']:.4f} -> "
              f"{results['adaptation_statistics']['final_parameter']:.4f} "
              f"({results['adaptation_statistics']['parameter_change_ratio']:.2f}x)")
        print(f"Processing time: {results['processing_time']:.2f}s")

        if results['change_timestamps']:
            timestamps_str = " ".join([f"{ts:.1f}s" for ts in results['change_timestamps']])
            print(f"Changes: {timestamps_str}")
        else:
            print("Changes: None")


if __name__ == "__main__":
    main()