import argparse
import os
import json
import numpy as np
import torch
import librosa
from typing import List, Tuple, Dict
import warnings
import concurrent.futures
from tqdm import tqdm
import time
import sys

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '/home/zhyuan/Desktop/PCD'))
from pann_backbone import PANNBackbone, load_pretrained_pann, create_sliding_windows, load_audio_file
from biooss_fdtd import BioOSSFDTD2D, BatchOnlineBioOSSProcessor
from audio_attention_utils import OnlineMultiMetricChangeDetector, generate_interpretation, smooth_detections
sys.path.append(os.path.join(os.path.dirname(__file__), '/home/zhyuan/Desktop/PCD/datasets'))
from dataloader import create_dataloader


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BioOSS Audio Change Detection - Optimized Batch Processing')

    # Input/Output
    parser.add_argument('--audio_dir', type=str,
                        default='/home/zhyuan/Desktop/PCD/data/LU_AVS/audio_dataset',
                        help='Path to audio dataset directory')
    parser.add_argument('--label_file', type=str,
                        default='/home/zhyuan/Desktop/PCD/data/LU_AVS/audio_label.json',
                        help='Path to label JSON file')
    parser.add_argument('--pann_checkpoint', type=str,
                        default='/home/zhyuan/Desktop/audioset_tagging_cnn/Cnn14_mAP=0.431.pth',
                        help='Path to PANN pretrained checkpoint')
    parser.add_argument('--output_dir', type=str, default='./batch_results',
                        help='Output directory for results')
    parser.add_argument('--results_json', type=str, default='detection_results.json',
                        help='JSON file name for results')

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
    parser.add_argument('--freq_max', type=float, default=800.0,
                        help='Maximum frequency for BioOSS initialization')
    parser.add_argument('--grid_size', type=int, default=64,
                        help='BioOSS grid size')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='BioOSS time step')

    # Detection parameters
    parser.add_argument('--energy_weight', type=float, default=0.4,
                        help='Weight for energy metric')
    parser.add_argument('--pca_weight', type=float, default=0.35,
                        help='Weight for PCA metric')
    parser.add_argument('--cosine_weight', type=float, default=0.25,
                        help='Weight for cosine metric')
    parser.add_argument('--consensus_threshold', type=float, default=0.3,
                        help='Consensus threshold for change detection')
    parser.add_argument('--min_persistence', type=int, default=2,
                        help='Minimum persistence duration for detection')

    # Batch optimization parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for PANN feature extraction')
    parser.add_argument('--prefetch_size', type=int, default=4,
                        help='Number of samples to prefetch and process together')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of workers for parallel audio loading')

    # Processing options
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    parser.add_argument('--skip_errors', action='store_true',
                        help='Skip samples that cause errors instead of stopping')

    # Device configuration
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device for computation (auto/cpu/cuda)')

    # Debug options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--verbose_metrics', action='store_true',
                        help='Print detailed metrics for each timestep')

    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup and validate device configuration"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print(f"Using CPU (CUDA not available)")
    else:
        if device_arg == 'cuda':
            if not torch.cuda.is_available():
                print(f"Warning: CUDA not available, falling back to CPU")
                device = torch.device('cpu')
            else:
                device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device(device_arg)
            print(f"Using device: {device}")

    return device


def extract_ground_truth_changes(sample_info: Dict) -> Tuple[List[float], int]:
    """
    Extract ground truth change points from sample info.
    Only extracts start_time of each audio segment as change points.

    Args:
        sample_info: Dictionary containing sample information with audio_segments

    Returns:
        Tuple of (change_points_list, num_segments)
    """
    change_points = []

    if 'audio_segments' in sample_info:
        audio_segments = sample_info['audio_segments']

        # Extract only start_time from each segment
        for segment in audio_segments:
            if 'start_time' in segment:
                change_points.append(float(segment['start_time']))

        # Sort change points to ensure chronological order
        change_points.sort()

        # Remove the first change point if it's 0.0 (beginning of audio is not a "change")
        if change_points and change_points[0] == 0.0:
            change_points = change_points[1:]

        return change_points, len(audio_segments)

    return [], 0


def load_audio_batch(sample_infos: List[Tuple[str, Dict]], args) -> Dict[str, Tuple[np.ndarray, int]]:
    """Load multiple audio files in parallel"""

    def load_single_audio(sample_info):
        sample_id, info = sample_info
        try:
            audio, sr = load_audio_file(info['audio_path'], args.sample_rate)
            return sample_id, (audio, sr), None
        except Exception as e:
            return sample_id, None, str(e)

    audio_data = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(load_single_audio, sample_info) for sample_info in sample_infos]

        for future in concurrent.futures.as_completed(futures):
            sample_id, audio, error = future.result()
            if audio is not None:
                audio_data[sample_id] = audio
            elif args.debug and error:
                print(f"Failed to load {sample_id}: {error}")

    return audio_data


def extract_pann_features_batched(audio_segments: List[np.ndarray], pann_model: PANNBackbone,
                                  device: torch.device, batch_size: int = 16) -> torch.Tensor:
    """Extract PANN features with batch processing for efficiency"""
    if not audio_segments:
        return torch.empty(0, 527)

    pann_features = []
    total_segments = len(audio_segments)

    # Process in batches for efficiency
    for i in range(0, total_segments, batch_size):
        batch_segments = audio_segments[i:i + batch_size]

        # Convert batch to tensor
        batch_tensor = torch.stack([
            torch.tensor(segment, dtype=torch.float32)
            for segment in batch_segments
        ]).to(device)

        # Extract features for the batch
        with torch.no_grad():
            batch_output = pann_model(batch_tensor)
            batch_features = batch_output['clipwise_output'].cpu()

        pann_features.append(batch_features)

    # Concatenate all batch results
    return torch.cat(pann_features, dim=0)


def run_biooss_detection_single(pann_features: torch.Tensor, biooss_model: BioOSSFDTD2D,
                                detector: OnlineMultiMetricChangeDetector, device: torch.device,
                                debug: bool = False, verbose_metrics: bool = False) -> Dict:
    """Run BioOSS-based change detection - optimized version"""

    # Move to device
    biooss_model = biooss_model.to(device)
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

    # Process each timestep
    for t in range(num_timesteps):
        # Get current features
        current_features = pann_features[t:t + 1]  # Keep batch dimension

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

    # Apply temporal smoothing
    results['detections_smoothed'] = smooth_detections(results['detections'])

    return results


def process_sample_batch(sample_batch: List[Tuple[str, Dict]], pann_model: PANNBackbone,
                         biooss_model: BioOSSFDTD2D, args, device: torch.device) -> List[Dict]:
    """Process a batch of samples efficiently"""

    # Load all audio files in parallel
    audio_data = load_audio_batch(sample_batch, args)

    batch_results = []

    for sample_id, sample_info in sample_batch:
        if sample_id not in audio_data:
            # Audio loading failed
            result = {
                'sample_id': sample_id,
                'audio_path': sample_info.get('audio_path', 'unknown'),
                'processing_status': 'error',
                'error_message': 'Failed to load audio',
                'total_changes_detected': 0,
                'change_timestamps': [],
                'ground_truth_changes': [],
                'num_ground_truth_changes': 0,
                'total_segments': 0
            }
            batch_results.append(result)
            continue

        try:
            audio, sr = audio_data[sample_id]

            # Create sliding windows
            audio_segments = create_sliding_windows(
                audio, sr, args.window_length, args.stride_length
            )

            # Extract PANN features with batching
            pann_features = extract_pann_features_batched(
                audio_segments, pann_model, device, args.batch_size
            )

            # Create fresh detector for this sample
            detector = OnlineMultiMetricChangeDetector(
                energy_weight=args.energy_weight,
                pca_weight=args.pca_weight,
                cosine_weight=args.cosine_weight,
                consensus_threshold=args.consensus_threshold,
                min_persistence_duration=args.min_persistence
            )

            # Run detection
            results = run_biooss_detection_single(
                pann_features, biooss_model, detector, device,
                debug=args.debug, verbose_metrics=args.verbose_metrics
            )

            # Calculate results - removed change_indices calculation
            change_timestamps = [i * args.stride_length for i, d in enumerate(results['detections_smoothed']) if d]

            # Extract ground truth change points correctly
            ground_truth_changes, num_segments = extract_ground_truth_changes(sample_info)

            result = {
                'sample_id': sample_id,
                'audio_path': sample_info['audio_path'],
                'audio_duration': len(audio) / sr,
                'total_changes_detected': len(change_timestamps),
                'change_timestamps': change_timestamps,
                'average_confidence': float(np.mean(results['confidence_scores'])) if results[
                    'confidence_scores'] else 0.0,
                'max_confidence': float(np.max(results['confidence_scores'])) if results['confidence_scores'] else 0.0,
                'processing_status': 'success',
                'ground_truth_changes': ground_truth_changes,
                'num_ground_truth_changes': len(ground_truth_changes),
                # This should equal total_segments - 1 (excluding first segment at 0.0)
                'total_segments': num_segments  # This is the actual total number of segments
            }

            if args.debug and len(change_timestamps) > 0:
                print(f"  {sample_id}: {len(change_timestamps)} changes detected")

            batch_results.append(result)

        except Exception as e:
            if args.debug:
                import traceback
                print(f"Error processing {sample_id}: {e}")
                print(traceback.format_exc())

            result = {
                'sample_id': sample_id,
                'audio_path': sample_info.get('audio_path', 'unknown'),
                'processing_status': 'error',
                'error_message': str(e),
                'total_changes_detected': 0,
                'change_timestamps': [],
                'ground_truth_changes': [],
                'num_ground_truth_changes': 0,
                'total_segments': 0
            }
            batch_results.append(result)

    return batch_results


def main():
    """Main execution function with batch optimization"""
    args = parse_arguments()

    start_time = time.time()

    # Setup device
    device = setup_device(args.device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.audio_dir}")
    dataset = create_dataloader(args.audio_dir, args.label_file)

    # Validate dataset
    validation_results = dataset.validate_dataset()
    valid_samples = validation_results['valid']
    missing_samples = validation_results['missing']

    print(f"Dataset: {len(valid_samples)} valid samples, {len(missing_samples)} missing")

    if args.max_samples and args.max_samples < len(valid_samples):
        valid_samples = valid_samples[:args.max_samples]
        print(f"Processing limited to {args.max_samples} samples")

    # Load PANN model
    print("Loading PANN model...")
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

    pann_model = pann_model.to(device)
    pann_model.eval()

    # Initialize BioOSS model
    print("Initializing BioOSS FDTD model...")
    biooss_model = BioOSSFDTD2D(
        grid_size=args.grid_size,
        dt=args.dt,
        freq_range=(args.freq_min, args.freq_max)
    )

    # Prepare sample information
    print("Preparing sample information...")
    sample_infos = []
    for sample_id in valid_samples:
        try:
            info = dataset.get_sample_info(sample_id)
            sample_infos.append((sample_id, info))
        except Exception as e:
            if args.debug:
                print(f"Failed to get info for {sample_id}: {e}")

    # Process in batches for better efficiency
    all_results = []
    successful_processes = 0
    failed_processes = 0

    print(f"Starting batch processing with prefetch_size={args.prefetch_size}...")

    with tqdm(total=len(sample_infos), desc="Processing batches") as pbar:
        for i in range(0, len(sample_infos), args.prefetch_size):
            batch = sample_infos[i:i + args.prefetch_size]

            if args.debug:
                batch_ids = [sample_id for sample_id, _ in batch]
                print(f"\nProcessing batch {i // args.prefetch_size + 1}: {batch_ids}")

            # Process the batch
            batch_results = process_sample_batch(batch, pann_model, biooss_model, args, device)

            # Collect results
            for result in batch_results:
                all_results.append(result)
                if result['processing_status'] == 'success':
                    successful_processes += 1
                else:
                    failed_processes += 1

                pbar.update(1)

            # Optional: Clean up GPU memory periodically
            if device.type == 'cuda' and i % (args.prefetch_size * 4) == 0:
                torch.cuda.empty_cache()

    # Calculate summary statistics
    successful_results = [r for r in all_results if r['processing_status'] == 'success']
    total_changes_detected = sum(r['total_changes_detected'] for r in successful_results)
    samples_with_changes = len([r for r in successful_results if r['total_changes_detected'] > 0])
    total_ground_truth_changes = sum(r['num_ground_truth_changes'] for r in successful_results)
    total_segments = sum(r['total_segments'] for r in successful_results)

    # Calculate processing time
    processing_time = time.time() - start_time
    samples_per_second = len(successful_results) / processing_time if processing_time > 0 else 0

    # Save results
    final_results = {
        'processing_info': {
            'total_samples_attempted': len(valid_samples),
            'successful_processes': successful_processes,
            'failed_processes': failed_processes,
            'device_used': str(device),
            'processing_time_seconds': processing_time,
            'samples_per_second': samples_per_second,
            'processing_parameters': vars(args)
        },
        'summary_statistics': {
            'total_changes_detected': total_changes_detected,
            'samples_with_changes': samples_with_changes,
            'average_changes_per_sample': total_changes_detected / len(successful_results) if successful_results else 0,
            'change_detection_rate': samples_with_changes / len(successful_results) if successful_results else 0,
            'total_ground_truth_changes': total_ground_truth_changes,
            'total_segments': total_segments,
            'average_segments_per_sample': total_segments / len(successful_results) if successful_results else 0
        },
        'sample_results': all_results
    }

    results_path = os.path.join(args.output_dir, args.results_json)
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n" + "=" * 60)
    print(f"BATCH PROCESSING COMPLETE")
    print(f"=" * 60)
    print(f"Total samples: {successful_processes}/{len(valid_samples)} successful")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Processing speed: {samples_per_second:.2f} samples/second")
    print(f"Changes detected: {total_changes_detected}")
    print(f"Ground truth changes: {total_ground_truth_changes}")
    print(f"Total segments: {total_segments}")
    print(f"Samples with changes: {samples_with_changes}")
    print(f"Results saved to: {results_path}")

    # Show examples
    if successful_results:
        print(f"\nExample results:")
        successful_with_changes = [r for r in successful_results if r['total_changes_detected'] > 0]
        for result in successful_with_changes[:5]:
            print(
                f"  {result['sample_id']}: {result['total_changes_detected']} detected, {result['num_ground_truth_changes']} ground truth, {result['total_segments']} segments")

    # Show performance summary
    if len(successful_results) > 0:
        avg_changes = total_changes_detected / len(successful_results)
        detection_rate = samples_with_changes / len(successful_results) * 100
        print(f"\nPerformance Summary:")
        print(f"  Average changes per sample: {avg_changes:.2f}")
        print(f"  Sample detection rate: {detection_rate:.1f}%")
        print(f"  Batch processing speed: {samples_per_second:.2f} samples/second")


if __name__ == "__main__":
    main()