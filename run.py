import argparse
import os
import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use('Agg') # Use the non-interactive Matplotlib backend.
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

from pann_backbone import PANNBackbone, load_pretrained_pann, create_sliding_windows, load_audio_file
from biooss_fdtd import BioOSSFDTD2D, BatchOnlineBioOSSProcessor # new processor.
from audio_attention_utils import OnlineMultiMetricChangeDetector, generate_interpretation, smooth_detections

blues = ['#115699', '#0E6DB3', '#5CAAD7', '#95C6DE']
reds = ['#8E0D29', '#BB1E38', '#D35B4D', '#F6BCA9']
yellows = ['#E19D49', '#E8B547', '#EFC99B']
greens = ['#365C3B', '#4A6C4C', '#7DA47C', '#ABD0A7']

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BioOSS Audio Change Detection')

    # Input/Output
    parser.add_argument('--audio_path', type=str,
                        default='/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio/R0056_segment_ambisonics.wav',
                        help='Path to input audio file')
    parser.add_argument('--pann_checkpoint', type=str,
                        default='/home/zhyuan/Desktop/audioset_tagging_cnn/Cnn14_mAP=0.431.pth',
                        help='Path to PANN pretrained checkpoint')
    parser.add_argument('--output_dir', type=str, default='/home/zhyuan/Desktop/Qwen-Audio/Memory_model/plot/figures',
                        help='Output directory for results')

    # Audio processing
    parser.add_argument('--window_length', type=float, default=4.0,
                        help='Sliding window length in seconds')
    parser.add_argument('--stride_length', type=float, default=1.0,
                        help='Stride length in seconds')
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help='Audio sample rate')

    # BioOSS parameters
    parser.add_argument('--freq_min', type=float, default=50.0,
                        help='Minimum frequency for BioOSS initialization')
    parser.add_argument('--freq_max', type=float, default=1200.0,
                        help='Maximum frequency for BioOSS initialization')
    parser.add_argument('--grid_size', type=int, default=64,
                        help='BioOSS grid size')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='BioOSS time step')

    # Detection parameters
    # parser.add_argument('--energy_weight', type=float, default=0.4,
    #                     help='Weight for energy metric')
    # parser.add_argument('--pca_weight', type=float, default=0.35,
    #                     help='Weight for PCA metric')
    # parser.add_argument('--cosine_weight', type=float, default=0.25,
    #                     help='Weight for cosine metric')
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

    # Processing options
    parser.add_argument('--use_batch_processing', action='store_true', default=False,
                        help='Use batch processing for better performance')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing (if batch processing enabled)')

    # Debug options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--verbose_metrics', action='store_true',
                        help='Print detailed metrics for each timestep')

    # Visualization
    parser.add_argument('--plot_spectrogram', default=True, type=bool,
                        help='Generate enhanced spectrogram plot with change markers')
    parser.add_argument('--plot_metrics', default=False, type=bool,
                        help='Generate comprehensive metrics evolution plots')
    parser.add_argument('--plot_biooss', default=True, type=bool,
                        help='Generate BioOSS field evolution plots')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for computation (cpu/cuda)')

    return parser.parse_args()


def load_and_preprocess_audio(audio_path: str, sample_rate: int) -> Tuple[np.ndarray, int]:
    """Load and preprocess audio file"""
    print(f"Loading audio from: {audio_path}")

    # Load audio using the improved function
    audio, sr = load_audio_file(audio_path, sample_rate)

    print(f"Audio loaded: {len(audio) / sr:.2f}s, {sr}Hz")

    return audio, sr


def extract_pann_features(audio_segments: List[np.ndarray], pann_model: PANNBackbone,
                          device: str) -> torch.Tensor:
    """Extract PANN features from audio segments"""
    print(f"Extracting PANN features from {len(audio_segments)} segments...")

    pann_features = []

    for i, segment in enumerate(audio_segments):
        if i % 10 == 0:  # Progress indicator
            print(f"Processing segment {i + 1}/{len(audio_segments)}")

        # Convert to tensor and add batch dimension
        segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)

        # Extract features using the model's forward method
        with torch.no_grad():
            output = pann_model(segment_tensor)
            features = output['clipwise_output']  # Use clipwise_output for 527-dim features

        pann_features.append(features.cpu())

    # Stack all features
    pann_features = torch.cat(pann_features, dim=0)
    print(f"PANN features extracted: {pann_features.shape}")

    return pann_features


def run_biooss_detection_single(pann_features: torch.Tensor, biooss_model: BioOSSFDTD2D,
                               detector: OnlineMultiMetricChangeDetector, device: str,
                               debug: bool = False, verbose_metrics: bool = False) -> Dict:
    """Run BioOSS-based change detection using single timestep processing"""
    print("Running BioOSS change detection (single timestep mode)...")

    # Move to device
    biooss_model.to(device)
    pann_features = pann_features.to(device)

    # Initialize processor with new BatchOnlineBioOSSProcessor
    processor = BatchOnlineBioOSSProcessor(biooss_model)

    # Storage for results
    results = {
        'detections': [],
        'confidence_scores': [],
        'detailed_metrics': [],
        'timestamps': [],
        'interpretations': []
    }

    num_timesteps = pann_features.shape[0]

    # Process each timestep
    for t in range(num_timesteps):
        if t % 20 == 0:
            print(f"Processing timestep {t + 1}/{num_timesteps}")

        # Get current features
        current_features = pann_features[t:t + 1]  # Keep batch dimension

        # Process through BioOSS using the new method
        biooss_result = processor.process_single_timestep(current_features)

        # Update change detector
        is_change, confidence, detailed_metrics = detector.update(
            energy=biooss_result['energy'],
            state_vector=biooss_result['state_vector'],
            timestamp=t
        )

        # Store results
        results['detections'].append(is_change)
        results['confidence_scores'].append(confidence)
        results['detailed_metrics'].append(detailed_metrics)
        results['timestamps'].append(t)

        # Generate interpretation if change detected
        if is_change:
            interpretation = generate_interpretation(detailed_metrics, t)
            results['interpretations'].append(interpretation)
            print(f"Change detected at timestep {t} (confidence: {confidence:.3f})")

    # Apply temporal smoothing
    results['detections_smoothed'] = smooth_detections(results['detections'])

    print(f"Detection complete. Found {sum(results['detections_smoothed'])} significant changes.")

    return results


def run_biooss_detection_batch(pann_features: torch.Tensor, biooss_model: BioOSSFDTD2D,
                              detector: OnlineMultiMetricChangeDetector, device: str,
                              batch_size: int = 32, debug: bool = False, verbose_metrics: bool = False) -> Dict:
    """Run BioOSS-based change detection using batch processing for better performance"""
    print("Running BioOSS change detection (batch mode)...")

    # Move to device
    biooss_model.to(device)
    pann_features = pann_features.to(device)

    # Initialize processor
    processor = BatchOnlineBioOSSProcessor(biooss_model)

    # Storage for results
    results = {
        'detections': [],
        'confidence_scores': [],
        'detailed_metrics': [],
        'timestamps': [],
        'interpretations': []
    }

    num_timesteps = pann_features.shape[0]

    # Process in batches (but still need to maintain temporal order for change detection)
    for t in range(num_timesteps):
        if t % 20 == 0:
            print(f"Processing timestep {t + 1}/{num_timesteps}")

        # Get current features (still process one at a time for temporal consistency)
        current_features = pann_features[t:t + 1]

        # Process through BioOSS
        biooss_result = processor.process_single_timestep(current_features)

        # Update change detector
        is_change, confidence, detailed_metrics = detector.update(
            energy=biooss_result['energy'],
            state_vector=biooss_result['state_vector'],
            timestamp=t
        )

        # Store results
        results['detections'].append(is_change)
        results['confidence_scores'].append(confidence)
        results['detailed_metrics'].append(detailed_metrics)
        results['timestamps'].append(t)

        # Generate interpretation if change detected
        if is_change:
            interpretation = generate_interpretation(detailed_metrics, t)
            results['interpretations'].append(interpretation)
            if debug:
                print(f"Change detected at timestep {t} (confidence: {confidence:.3f})")

    # Apply temporal smoothing
    results['detections_smoothed'] = smooth_detections(results['detections'])

    print(f"Detection complete. Found {sum(results['detections_smoothed'])} significant changes.")

    return results


def create_enhanced_mel_spectrogram_with_detection(audio: np.ndarray, sr: int,
                                                   results: Dict, stride_length: float,
                                                   output_path: str, title_suffix: str = ""):
    """
    Create enhanced mel spectrogram visualization similar to the example image
    with KL divergence overlay, thresholds, and detection markers
    """
    print("Generating enhanced mel spectrogram visualization...")

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, fmax=8000, hop_length=512, n_fft=2048
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Create figure with specific size to match example
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot mel spectrogram as background
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
                             fmax=8000, cmap='viridis', alpha=0.8, ax=ax)

    # Set x-axis limits to 0-60 seconds explicitly
    ax.set_xlim(0, 60)

    # Create time axis for metrics (matching the detection timesteps)
    # Ensure time axis is in seconds and properly scaled
    time_axis = np.array(results['timestamps']) * stride_length

    # Extract metrics for visualization
    # Using energy metric as primary indicator (similar to KL divergence in example)
    primary_metric_values = [m['raw_metrics']['energy'] for m in results['detailed_metrics']]
    primary_thresholds = [m['thresholds']['energy'] for m in results['detailed_metrics']]

    # Scale metrics to fit the frequency range for better visualization
    max_freq = 128  # mel bins
    metric_scale_factor = max_freq / (max(primary_metric_values) if max(primary_metric_values) > 0 else 1)

    scaled_metrics = np.array(primary_metric_values) * metric_scale_factor
    scaled_thresholds = np.array(primary_thresholds) * metric_scale_factor

    # Create twin axis for metrics overlay
    ax2 = ax.twinx()
    ax2.set_ylim(0, max_freq)
    ax2.set_xlim(0, 60)  # Also set x-limits for the twin axis

    # Plot primary metric (energy) line - similar to KL divergence in example
    metric_line = ax2.plot(time_axis, scaled_metrics,
                           color='cyan', linewidth=2, alpha=0.9,
                           label='Energy Metric')

    # Plot adaptive threshold line
    threshold_line = ax2.plot(time_axis, scaled_thresholds,
                              color=reds[1], linewidth=2, linestyle='--', alpha=0.8,
                              label=f'Threshold')
                              # label=f'Threshold ({results.get("consensus_threshold", 0.3):.3f})')

    # Add detection markers
    change_indices = [i for i, d in enumerate(results['detections_smoothed']) if d]

    for i, change_idx in enumerate(change_indices):
        change_time = change_idx * stride_length
        confidence = results['confidence_scores'][change_idx]

        # Add vertical line for detection
        ax.axvline(x=change_time, color=reds[0], linewidth=3, alpha=0.9)

        # Add detection marker with timestamp
        ax.text(change_time, max_freq * 0.9, f'{change_time:.0f}s',
                rotation=0, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=reds[0], alpha=0.8),
                color='white', fontsize=16, fontweight='bold')

    # Set labels and title
    ax.set_xlabel('Time (seconds)', fontsize=22)
    ax.set_ylabel('Mel Frequency Bins', fontsize=22)
    ax2.set_ylabel('Metric Value (Scaled)', fontsize=22, color='cyan')

    # # Create comprehensive title
    total_changes = len(change_indices)
    # title = f'BioOSS Energy-based Detection - {title_suffix}\n' if title_suffix else 'BioOSS Energy-based Detection\n'
    # title += f'Detections: {total_changes} changes found'
    # ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add legend combining both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc='upper right', framealpha=0.9, fontsize=16)

    # Color the threshold line area
    ax2.fill_between(time_axis, 0, scaled_thresholds,
                     color=reds[1], alpha=0.1, label='_nolegend_')

    # Highlight detection regions
    for change_idx in change_indices:
        change_time = change_idx * stride_length
        # Add a subtle highlight region around each detection
        ax.axvspan(change_time - stride_length / 2, change_time + stride_length / 2,
                   color=reds[0], alpha=0.1)

    # Adjust layout
    plt.tight_layout()

    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    print(f"Enhanced spectrogram saved to: {output_path}")

    return {
        'total_detections': total_changes,
        'detection_times': [idx * stride_length for idx in change_indices],
        'avg_confidence': np.mean([results['confidence_scores'][i] for i in change_indices]) if change_indices else 0
    }


def create_multi_metric_visualization(results: Dict, stride_length: float, output_path: str):
    """
    Create a comprehensive multi-metric visualization showing all detection components
    """
    print("Generating multi-metric visualization...")

    fig, axes = plt.subplots(4, 1, figsize=(16, 12))

    # Time axis
    time_axis = np.array(results['timestamps']) * stride_length

    # Extract all metrics
    energy_values = [m['raw_metrics']['energy'] for m in results['detailed_metrics']]
    pca_values = [m['raw_metrics']['pca'] for m in results['detailed_metrics']]
    cosine_values = [m['raw_metrics']['cosine'] for m in results['detailed_metrics']]

    energy_thresholds = [m['thresholds']['energy'] for m in results['detailed_metrics']]
    pca_thresholds = [m['thresholds']['pca'] for m in results['detailed_metrics']]
    cosine_thresholds = [m['thresholds']['cosine'] for m in results['detailed_metrics']]

    confidence_scores = results['confidence_scores']

    # Get detection points
    change_indices = [i for i, d in enumerate(results['detections_smoothed']) if d]

    # Plot 1: Energy Metric
    axes[0].plot(time_axis, energy_values, 'b-', linewidth=2, label='BioOSS Energy')
    axes[0].plot(time_axis, energy_thresholds, 'r--', linewidth=2, label='Adaptive Threshold')
    axes[0].fill_between(time_axis, energy_values, energy_thresholds,
                         where=np.array(energy_values) > np.array(energy_thresholds),
                         color='red', alpha=0.3, label='Above Threshold')

    # Mark detections
    for idx in change_indices:
        axes[0].axvline(x=idx * stride_length, color='red', linewidth=2, alpha=0.8)
        axes[0].text(idx * stride_length, max(energy_values) * 0.9, f'{idx * stride_length:.1f}s',
                     rotation=90, ha='right', va='top', fontsize=8)

    axes[0].set_title('BioOSS Energy Dynamics', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Energy Acceleration')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: PCA Metric
    axes[1].plot(time_axis, pca_values, 'g-', linewidth=2, label='PCA Subspace Change')
    axes[1].plot(time_axis, pca_thresholds, 'r--', linewidth=2, label='Adaptive Threshold')
    axes[1].fill_between(time_axis, pca_values, pca_thresholds,
                         where=np.array(pca_values) > np.array(pca_thresholds),
                         color='red', alpha=0.3)

    for idx in change_indices:
        axes[1].axvline(x=idx * stride_length, color='red', linewidth=2, alpha=0.8)

    axes[1].set_title('PCA Subspace Analysis', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Subspace Angle')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Cosine Metric
    axes[2].plot(time_axis, cosine_values, 'm-', linewidth=2, label='Cosine Distance')
    axes[2].plot(time_axis, cosine_thresholds, 'r--', linewidth=2, label='Adaptive Threshold')
    axes[2].fill_between(time_axis, cosine_values, cosine_thresholds,
                         where=np.array(cosine_values) > np.array(cosine_thresholds),
                         color='red', alpha=0.3)

    for idx in change_indices:
        axes[2].axvline(x=idx * stride_length, color='red', linewidth=2, alpha=0.8)

    axes[2].set_title('Cosine Distance Analysis', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Cosine Distance')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Combined Confidence and Detections
    axes[3].plot(time_axis, confidence_scores, 'orange', linewidth=2, label='Detection Confidence')

    # Add consensus threshold line
    consensus_threshold = results.get('consensus_threshold', 0.3)
    axes[3].axhline(y=consensus_threshold, color='red', linestyle='--', alpha=0.7,
                    label=f'Consensus Threshold')
                    # label=f'Consensus Threshold ({consensus_threshold})')

    # Mark final detections
    detection_binary = [1 if d else 0 for d in results['detections_smoothed']]
    axes[3].fill_between(time_axis, 0, detection_binary,
                         color='red', alpha=0.5, step='mid', label='Final Detections')

    for idx in change_indices:
        axes[3].axvline(x=idx * stride_length, color='red', linewidth=2, alpha=0.8)
        axes[3].text(idx * stride_length, 0.9, f'{idx * stride_length:.1f}s',
                     rotation=90, ha='right', va='top', fontsize=8)

    axes[3].set_title('Final Detection Results', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Confidence / Detection')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Multi-metric visualization saved to: {output_path}")


def plot_metrics_evolution(results: Dict, stride_length: float, output_path: str):
    """Plot evolution of detection metrics (original version)"""
    print("Generating original metrics evolution plots...")

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Time axis
    time_axis = np.array(results['timestamps']) * stride_length

    # Extract metric time series
    energy_values = [m['raw_metrics']['energy'] for m in results['detailed_metrics']]
    pca_values = [m['raw_metrics']['pca'] for m in results['detailed_metrics']]
    cosine_values = [m['raw_metrics']['cosine'] for m in results['detailed_metrics']]

    energy_thresholds = [m['thresholds']['energy'] for m in results['detailed_metrics']]
    pca_thresholds = [m['thresholds']['pca'] for m in results['detailed_metrics']]
    cosine_thresholds = [m['thresholds']['cosine'] for m in results['detailed_metrics']]

    confidence_scores = results['confidence_scores']

    # Plot individual metrics
    axes[0, 0].plot(time_axis, energy_values, 'b-', label='Energy Metric')
    axes[0, 0].plot(time_axis, energy_thresholds, 'r--', label='Adaptive Threshold')
    axes[0, 0].set_title('BioOSS Energy Dynamics')
    axes[0, 0].set_ylabel('Energy Acceleration')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(time_axis, pca_values, 'g-', label='PCA Metric')
    axes[0, 1].plot(time_axis, pca_thresholds, 'r--', label='Adaptive Threshold')
    axes[0, 1].set_title('PCA Subspace Changes')
    axes[0, 1].set_ylabel('Subspace Angle')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(time_axis, cosine_values, 'm-', label='Cosine Metric')
    axes[1, 0].plot(time_axis, cosine_thresholds, 'r--', label='Adaptive Threshold')
    axes[1, 0].set_title('Cosine Distance Changes')
    axes[1, 0].set_ylabel('Cosine Distance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot confidence scores
    axes[1, 1].plot(time_axis, confidence_scores, 'orange', linewidth=2)
    axes[1, 1].axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Detection Threshold')
    axes[1, 1].set_title('Detection Confidence')
    axes[1, 1].set_ylabel('Confidence Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot detection results
    detections_int = [int(d) for d in results['detections_smoothed']]
    axes[2, 0].plot(time_axis, detections_int, 'ro-', markersize=4)
    axes[2, 0].set_title('Change Detection Results')
    axes[2, 0].set_ylabel('Detection (0/1)')
    axes[2, 0].set_ylim(-0.1, 1.1)
    axes[2, 0].grid(True, alpha=0.3)

    # Plot combined view
    axes[2, 1].plot(time_axis, confidence_scores, 'orange', alpha=0.7, label='Confidence')
    ax2 = axes[2, 1].twinx()
    ax2.plot(time_axis, detections_int, 'ro-', markersize=3, label='Detections')
    axes[2, 1].set_title('Combined Detection View')
    axes[2, 1].set_ylabel('Confidence', color='orange')
    ax2.set_ylabel('Detection', color='red')
    axes[2, 1].grid(True, alpha=0.3)

    # Set x-axis labels
    for ax in axes.flat:
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Original metrics plot saved to: {output_path}")


def save_results_summary(results: Dict, output_path: str, stride_length: float):
    """Save detection results summary"""
    change_indices = [i for i, d in enumerate(results['detections_smoothed']) if d]
    change_times = [i * stride_length for i in change_indices]

    summary = {
        'total_segments': len(results['detections']),
        'total_changes_detected': len(change_indices),
        'change_timestamps': change_times,
        'change_indices': change_indices,
        'average_confidence': np.mean(results['confidence_scores']),
        'interpretations': results['interpretations']
    }

    # Write summary to file
    with open(output_path, 'w') as f:
        f.write("BioOSS Audio Change Detection Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total audio segments processed: {summary['total_segments']}\n")
        f.write(f"Total changes detected: {summary['total_changes_detected']}\n")
        f.write(f"Average confidence: {summary['average_confidence']:.3f}\n\n")

        f.write("Detected Changes:\n")
        f.write("-" * 20 + "\n")
        for i, (time, idx) in enumerate(zip(change_times, change_indices)):
            f.write(f"{i + 1:2d}. Time: {time:6.2f}s (segment {idx:3d})\n")

        f.write("\nDetailed Interpretations:\n")
        f.write("-" * 25 + "\n")
        for i, interp in enumerate(results['interpretations']):
            f.write(f"\nChange {i + 1} at {interp['timestamp'] * stride_length:.2f}s:\n")
            f.write(f"  Confidence: {interp['detection_confidence']:.3f}\n")
            f.write(f"  Contributing factors:\n")
            for factor in interp['contributing_factors']:
                f.write(f"    - {factor['metric']}: {factor['strength']:.2f}x threshold\n")

    print(f"Results summary saved to: {output_path}")


def main():
    """Main execution function"""
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load audio
    audio, sr = load_and_preprocess_audio(args.audio_path, args.sample_rate)

    # Create sliding windows
    audio_segments = create_sliding_windows(
        audio, sr, args.window_length, args.stride_length
    )

    # Load PANN model with proper parameters
    print("Loading PANN model...")
    pann_model = load_pretrained_pann(
        checkpoint_path=args.pann_checkpoint,
        device=device,
        sample_rate=args.sample_rate,
        window_size=1024,  # Standard PANN parameters
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527
    )

    # Extract PANN features
    pann_features = extract_pann_features(audio_segments, pann_model, device)

    # Initialize BioOSS model
    print("Initializing BioOSS FDTD model...")
    biooss_model = BioOSSFDTD2D(
        grid_size=args.grid_size,
        dt=args.dt,
        freq_range=(args.freq_min, args.freq_max)
    )

    # Initialize change detector
    detector = OnlineMultiMetricChangeDetector(
        energy_weight=args.energy_weight,
        pca_weight=args.pca_weight,
        cosine_weight=args.cosine_weight,
        consensus_threshold=args.consensus_threshold,
        min_persistence_duration=args.min_persistence
    )

    # Run detection - choose processing mode
    if args.use_batch_processing:
        print("Using batch processing mode for better performance...")
        results = run_biooss_detection_batch(pann_features, biooss_model, detector, device,
                                           batch_size=args.batch_size,
                                           debug=args.debug, verbose_metrics=args.verbose_metrics)
    else:
        print("Using single timestep processing mode...")
        results = run_biooss_detection_single(pann_features, biooss_model, detector, device,
                                            debug=args.debug, verbose_metrics=args.verbose_metrics)

    # Get change timestamps for visualization
    change_indices = [i for i, d in enumerate(results['detections_smoothed']) if d]

    # Extract audio filename for title
    audio_filename = os.path.basename(args.audio_path).replace('.wav', '')

    # Generate enhanced visualizations
    if args.plot_spectrogram:
        spectrogram_path = os.path.join(args.output_dir, f'{audio_filename[:5]}_mel_spectrogram_bio.pdf')
        viz_stats = create_enhanced_mel_spectrogram_with_detection(
            audio, sr, results, args.stride_length, spectrogram_path, audio_filename
        )
        print(f"Enhanced visualization stats: {viz_stats}")

    if args.plot_metrics:
        # New comprehensive metrics visualization
        comprehensive_metrics_path = os.path.join(args.output_dir, 'comprehensive_metrics_analysis.png')
        create_multi_metric_visualization(results, args.stride_length, comprehensive_metrics_path)

        # Keep original metrics plot for comparison
        original_metrics_path = os.path.join(args.output_dir, 'original_metrics_evolution.png')
        plot_metrics_evolution(results, args.stride_length, original_metrics_path)

    # # Save results summary
    # summary_path = os.path.join(args.output_dir, 'detection_summary.txt')
    # save_results_summary(results, summary_path, args.stride_length)
    #
    # # Save raw results as numpy arrays for further analysis
    # np.savez(
    #     os.path.join(args.output_dir, 'raw_results.npz'),
    #     detections=np.array(results['detections']),
    #     detections_smoothed=np.array(results['detections_smoothed']),
    #     confidence_scores=np.array(results['confidence_scores']),
    #     timestamps=np.array(results['timestamps']),
    #     pann_features=pann_features.cpu().numpy()
    # )

    print(f"\nDetection complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Found {sum(results['detections_smoothed'])} significant changes")

    # Print change summary
    if change_indices:
        print(f"\nDetected changes ({len(change_indices)} total):")
        for i, idx in enumerate(change_indices):
            time_stamp = idx * args.stride_length
            confidence = results['confidence_scores'][idx]
            # Only show meaningful detections (confidence > 0)
            if confidence > 0:
                print(f"  {i + 1:2d}. {time_stamp:6.2f}s (timestep {idx}, confidence: {confidence:.3f})")
    else:
        print("No significant changes detected.")

        # If no changes detected, show the raw detections for debugging
        raw_changes = [i for i, d in enumerate(results['detections']) if d]
        if raw_changes:
            print(f"\nHowever, {len(raw_changes)} raw detections were found but filtered out by smoothing:")
            for idx in raw_changes:
                time_stamp = idx * args.stride_length
                confidence = results['confidence_scores'][idx]
                print(f"  Raw detection at {time_stamp:6.2f}s (timestep {idx}, confidence: {confidence:.3f})")

    # Show high confidence detections
    high_confidence_changes = [(i, results['confidence_scores'][i]) for i, d in
                               enumerate(results['detections_smoothed']) if d and results['confidence_scores'][i] > 0.3]
    if high_confidence_changes:
        print(f"\nHigh confidence changes (>0.3):")
        for idx, conf in high_confidence_changes:
            time_stamp = idx * args.stride_length
            print(f"  {time_stamp:6.2f}s (timestep {idx}, confidence: {conf:.3f})")


if __name__ == "__main__":
    main()