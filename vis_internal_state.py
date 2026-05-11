#!/usr/bin/env python3
"""
Internal State Processing and Visualization Script for BioOSS

This script handles:
1. Processing audio samples and saving internal states
2. Loading existing internal states
3. Visualizing internal states as heatmaps across timesteps
4. Organizing outputs in structured folders
"""

import argparse
import os
import numpy as np
import torch
import librosa

# Configure matplotlib backend before importing pyplot
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
import glob
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Import your existing modules (adjust paths as needed)
from pann_backbone import PANNBackbone, load_pretrained_pann, create_sliding_windows, load_audio_file
from biooss_fdtd import BioOSSFDTD2D, BatchOnlineBioOSSProcessor
from audio_attention_utils import OnlineMultiMetricChangeDetector, generate_interpretation, smooth_detections


def create_custom_colormap():
    """
    Create custom colormap based on provided 6 colors.
    Color progression: deep blue -> blue -> light blue-gray -> light pink -> red -> deep red

    Returns:
        LinearSegmentedColormap: Custom colormap for heatmap visualization
    """
    # Define 6 colors in RGB values (normalized to 0-1 range) - reversed order
    colors = [
        (11 / 255, 66 / 255, 94 / 255),  # deep blue
        (93 / 255, 139 / 255, 157 / 255),  # blue
        (202 / 255, 213 / 255, 219 / 255),  # light blue-gray
        (224 / 255, 198 / 255, 197 / 255),  # light pink
        (144 / 255, 69 / 255, 80 / 255),  # red
        (86 / 255, 19 / 255, 25 / 255)  # deep red
    ]

    # Create custom colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_heatmap', colors, N=256)

    return custom_cmap


def get_colormap(colormap_name):
    """
    Get colormap based on parameter selection.

    Args:
        colormap_name (str): Name of the colormap ('custom' or standard matplotlib names)

    Returns:
        Colormap object or string: Colormap for visualization
    """
    if colormap_name == 'custom':
        return create_custom_colormap()
    else:
        return colormap_name


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BioOSS Internal State Processing and Visualization')

    # Mode selection
    parser.add_argument('--mode', type=str, choices=['process', 'visualize', 'fft', 'both'], default='visualize',
                        help='Mode: process (save states), visualize (load and plot), fft (FFT analysis), or both')
    # Sample selection
    parser.add_argument('--sample_index', type=str, default='R0002',
                        help='Sample identifier (e.g., R0078, R0079)')

    # Input paths (for processing mode)
    parser.add_argument('--audio_dir', type=str,
                        default='/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio',
                        help='Directory containing audio files')
    parser.add_argument('--pann_checkpoint', type=str,
                        default='/home/zhyuan/Desktop/audioset_tagging_cnn/Cnn14_mAP=0.431.pth',
                        help='Path to PANN pretrained checkpoint')

    # Output paths
    parser.add_argument('--internal_state_dir', type=str,
                        default='/home/zhyuan/Desktop/PCD/plot/internal_state',
                        help='Directory to save/load internal states')
    parser.add_argument('--visualization_dir', type=str,
                        default='/home/zhyuan/Desktop/PCD/plot/visualizations',
                        help='Directory to save visualization figures')
    parser.add_argument('--gif_dir', type=str,
                        default='/home/zhyuan/Desktop/PCD/plot/visualizations/gif',
                        help='Directory to save GIF animations')

    # FFT parameters
    parser.add_argument('--perform_fft', action='store_true',
                        help='Perform FFT analysis on pressure neurons')
    parser.add_argument('--fft_dir', type=str,
                        default='/home/zhyuan/Desktop/PCD/plot/visualizations/FFT',
                        help='Directory to save FFT analysis results')
    parser.add_argument('--fft_pdf_dir', type=str,
                        default='/home/zhyuan/Desktop/PCD/plot/visualizations/FFT_pdf',
                        help='Directory to save FFT analysis results')

    # Audio processing parameters
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

    # Visualization parameters
    parser.add_argument('--visualize_component', type=str, choices=['p', 'vx', 'vy', 'all'], default='p',
                        help='Which component to visualize (p=pressure, vx=velocity_x, vy=velocity_y, all=all components)')
    parser.add_argument('--max_timesteps', type=int, default=0,
                        help='Maximum number of timesteps to visualize (0 for all)')
    parser.add_argument('--timestep_step', type=int, default=1,
                        help='Step size for timestep visualization (1 for every timestep)')
    parser.add_argument('--create_gif_from_heatmaps', action='store_true',
                        help='Create GIF from existing heatmap images')
    parser.add_argument('--gif_duration', type=float, default=0.2,
                        help='Duration between frames in GIF (seconds)')

    # Colormap parameter - UPDATED
    parser.add_argument('--colormap', type=str,
                        choices=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'custom'],
                        default='custom',
                        help='Colormap for heatmaps (viridis, plasma, inferno, magma, cividis, custom)')

    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for computation (cpu/cuda)')

    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--verbose_metrics', action='store_true',
                        help='Print detailed metrics for each timestep')

    return parser.parse_args()


def create_sample_directories(sample_id: str, visualization_dir: str, gif_dir: str) -> Tuple[str, str]:
    """
    Create organized directory structure for a sample.

    Args:
        sample_id: Sample identifier (e.g., "R0078")
        visualization_dir: Base visualization directory
        gif_dir: Base GIF directory

    Returns:
        Tuple of (sample_heatmap_dir, gif_dir_path)
    """
    # Create sample-specific heatmap directory
    sample_heatmap_dir = os.path.join(visualization_dir, f"{sample_id}_heatmap")
    os.makedirs(sample_heatmap_dir, exist_ok=True)

    sample_heatmap_pdf_dir = os.path.join(visualization_dir, f"{sample_id}_heatmap_pdf")
    os.makedirs(sample_heatmap_pdf_dir, exist_ok=True)

    # Ensure GIF directory exists
    os.makedirs(gif_dir, exist_ok=True)

    print(f"Created directories:")
    print(f"  Heatmaps: {sample_heatmap_dir}")
    print(f"  GIFs: {gif_dir}")

    return sample_heatmap_dir, sample_heatmap_pdf_dir, gif_dir


def save_internal_states(internal_states: torch.Tensor, sample_id: str,
                         output_dir: str, metadata: Optional[Dict] = None):
    """
    Save internal states from BioOSS processing to a file.

    Args:
        internal_states: Tensor of shape (time_steps, grid_size, grid_size, 3) containing p, vx, vy
        sample_id: Sample identifier (e.g., "R0078")
        output_dir: Directory to save the internal states
        metadata: Optional metadata dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{sample_id}_internal_state.npz"
    filepath = os.path.join(output_dir, filename)

    # Convert to numpy if tensor
    if isinstance(internal_states, torch.Tensor):
        internal_states_np = internal_states.cpu().numpy()
    else:
        internal_states_np = internal_states

    # Prepare save data
    save_data = {
        'internal_states': internal_states_np,
        'sample_id': sample_id,
        'shape_description': 'Shape: (time_steps, grid_size, grid_size, 3) where 3 = [p, vx, vy]',
        'time_steps': internal_states_np.shape[0],
        'grid_size': internal_states_np.shape[1],
        'channels': 3,
        'channel_names': ['pressure', 'velocity_x', 'velocity_y']
    }

    # Add metadata if provided
    if metadata:
        save_data.update(metadata)

    # Save as compressed numpy file
    np.savez_compressed(filepath, **save_data)

    print(f"Internal states saved to: {filepath}")
    print(f"Shape: {internal_states_np.shape}")
    print(f"Size: {internal_states_np.nbytes / (1024 ** 2):.2f} MB")

    return filepath


def load_internal_states(sample_id: str, input_dir: str) -> Tuple[np.ndarray, Dict]:
    """
    Load internal states from saved file.

    Args:
        sample_id: Sample identifier (e.g., "R0078")
        input_dir: Directory containing saved internal states

    Returns:
        Tuple of (internal_states_array, metadata_dict)
    """
    filename = f"{sample_id}_internal_state.npz"
    filepath = os.path.join(input_dir, filename)

    assert os.path.exists(filepath), f"Internal state file not found: {filepath}"

    print(f"Loading internal states from: {filepath}")

    data = np.load(filepath)
    internal_states = data['internal_states']

    # Extract metadata
    metadata = {}
    for key in data.files:
        if key != 'internal_states':
            metadata[key] = data[key].item() if data[key].ndim == 0 else data[key]

    print(f"Loaded internal states with shape: {internal_states.shape}")

    return internal_states, metadata


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


def run_biooss_detection_with_state_collection(pann_features: torch.Tensor, biooss_model: BioOSSFDTD2D,
                                               detector: OnlineMultiMetricChangeDetector, device: str,
                                               args, debug: bool = False, verbose_metrics: bool = False) -> Tuple[
    Dict, torch.Tensor]:
    """Run BioOSS-based change detection using single timestep processing and collect internal states"""
    print("Running BioOSS change detection with internal state collection...")

    # Move to device
    biooss_model.to(device)
    pann_features = pann_features.to(device)

    # Initialize processor with BatchOnlineBioOSSProcessor
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

    # Pre-allocate tensor for internal states (time_steps, grid_size, grid_size, 3)
    internal_states = torch.zeros(num_timesteps, args.grid_size, args.grid_size, 3, device=device)

    # Process each timestep
    for t in range(num_timesteps):
        if t % 20 == 0:
            print(f"Processing timestep {t + 1}/{num_timesteps}")

        # Get current features
        current_features = pann_features[t:t + 1]  # Keep batch dimension

        # Process through BioOSS using the processor
        biooss_result = processor.process_single_timestep(current_features)

        # Collect internal state using the available data from biooss_result
        pressure_field = biooss_result['pressure_field']  # Shape: (grid_size, grid_size)
        state_vector = biooss_result['state_vector']

        # The state_vector contains flattened [p, vx, vy] fields
        # Total size should be grid_size * grid_size * 3
        total_elements = args.grid_size * args.grid_size

        if state_vector.numel() >= total_elements * 3:
            # Extract p, vx, vy from state_vector
            p_flat = state_vector[:total_elements]
            vx_flat = state_vector[total_elements:2 * total_elements]
            vy_flat = state_vector[2 * total_elements:3 * total_elements]

            # Reshape to grid
            p_field = p_flat.reshape(args.grid_size, args.grid_size)
            vx_field = vx_flat.reshape(args.grid_size, args.grid_size)
            vy_field = vy_flat.reshape(args.grid_size, args.grid_size)

            # Stack to create (grid_size, grid_size, 3)
            current_state = torch.stack([p_field, vx_field, vy_field], dim=-1)

            if debug and t < 5:
                print(f"Debug timestep {t}: Extracted full state from state_vector, shape {current_state.shape}")
        else:
            # Fallback: use pressure_field and create zero velocity fields
            vx_field = torch.zeros_like(pressure_field)
            vy_field = torch.zeros_like(pressure_field)
            current_state = torch.stack([pressure_field, vx_field, vy_field], dim=-1)

            if debug and t < 5:
                print(f"Debug timestep {t}: State vector too small, using pressure + zero velocities")

        internal_states[t] = current_state

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

    return results, internal_states


def process_sample_and_save_states(sample_id: str, args) -> str:
    """
    Process audio sample and save internal states following the original script methodology.

    Args:
        sample_id: Sample identifier
        args: Command line arguments

    Returns:
        Path to saved internal states file
    """
    print(f"Processing sample: {sample_id}")

    # Find audio file
    audio_pattern = os.path.join(args.audio_dir, f"{sample_id}*")
    audio_files = glob.glob(audio_pattern)

    assert audio_files, f"No audio file found for sample {sample_id} in {args.audio_dir}"

    audio_path = audio_files[0]
    print(f"Found audio file: {audio_path}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess audio
    print("Loading and preprocessing audio...")
    audio, sr = load_audio_file(audio_path, args.sample_rate)
    print(f"Audio loaded: {len(audio) / sr:.2f}s, {sr}Hz")

    # Create sliding windows
    audio_segments = create_sliding_windows(audio, sr, args.window_length, args.stride_length)
    print(f"Created {len(audio_segments)} audio segments")

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

    # Run detection with internal state collection
    results, internal_states = run_biooss_detection_with_state_collection(
        pann_features, biooss_model, detector, device, args,
        debug=args.debug, verbose_metrics=getattr(args, 'verbose_metrics', False)
    )

    # Prepare metadata
    metadata = {
        'sample_id': sample_id,
        'audio_path': audio_path,
        'window_length': args.window_length,
        'stride_length': args.stride_length,
        'sample_rate': args.sample_rate,
        'grid_size': args.grid_size,
        'freq_range': [args.freq_min, args.freq_max],
        'num_timesteps': len(results['timestamps']),
        'energy_weight': args.energy_weight,
        'pca_weight': args.pca_weight,
        'cosine_weight': args.cosine_weight,
        'consensus_threshold': args.consensus_threshold
    }

    # Save internal states
    filepath = save_internal_states(internal_states, sample_id, args.internal_state_dir, metadata)

    return filepath


def visualize_internal_states(internal_states: np.ndarray, sample_id: str, args, metadata: Dict = None):
    """
    Create heatmap visualizations of internal states.

    Args:
        internal_states: Array of shape (time_steps, grid_size, grid_size, 3)
        sample_id: Sample identifier
        args: Command line arguments
        metadata: Optional metadata dictionary
    """
    print(f"Creating visualizations for sample: {sample_id}")

    # Get colormap based on parameter
    colormap = get_colormap(args.colormap)

    # Create sample-specific directories
    sample_heatmap_dir, sample_heatmap_pdf_dir, gif_dir = create_sample_directories(sample_id, args.visualization_dir, args.gif_dir)


    time_steps, grid_size, _, channels = internal_states.shape
    channel_names = ['pressure', 'velocity_x', 'velocity_y']

    # Determine which components to visualize
    if args.visualize_component == 'all':
        components_to_plot = [0, 1, 2]  # p, vx, vy
        component_names = channel_names
    else:
        component_map = {'p': 0, 'vx': 1, 'vy': 2}
        components_to_plot = [component_map[args.visualize_component]]
        component_names = [channel_names[component_map[args.visualize_component]]]

    # Determine timesteps to visualize
    max_timesteps = args.max_timesteps if args.max_timesteps > 0 else time_steps
    step_size = args.timestep_step
    timesteps_to_plot = list(range(0, min(time_steps, max_timesteps), step_size))

    print(f"Visualizing {len(timesteps_to_plot)} timesteps for {len(components_to_plot)} component(s)")
    print(f"Using colormap: {args.colormap}")

    # Create individual timestep heatmaps
    for comp_idx, comp_name in zip(components_to_plot, component_names):
        print(f"Creating heatmaps for {comp_name} component...")

        # Calculate global min/max for consistent color scale across ALL timesteps
        comp_data = internal_states[:, :, :, comp_idx]

        # Method 1: Use percentiles to avoid outliers
        vmin_percentile, vmax_percentile = np.percentile(comp_data, [1, 99])

        # Method 2: Use absolute min/max for full range
        vmin_absolute, vmax_absolute = np.min(comp_data), np.max(comp_data)

        # Method 3: Use symmetric range around zero (often better for scientific data)
        max_abs = np.percentile(np.abs(comp_data), 99)
        vmin_symmetric, vmax_symmetric = -max_abs, max_abs

        # Choose method (you can change this)
        vmin_global, vmax_global = vmin_percentile, vmax_percentile  # Using percentile method

        print(f"  Global color range for {comp_name}:")
        print(f"    Percentile method: [{vmin_percentile:.4f}, {vmax_percentile:.4f}]")
        print(f"    Absolute method: [{vmin_absolute:.4f}, {vmax_absolute:.4f}]")
        print(f"    Symmetric method: [{vmin_symmetric:.4f}, {vmax_symmetric:.4f}]")
        print(f"    Using: [{vmin_global:.4f}, {vmax_global:.4f}]")

        for t_idx in tqdm(timesteps_to_plot, desc=f"Creating {comp_name} heatmaps"):
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create heatmap with FIXED global color scale
            data = internal_states[t_idx, :, :, comp_idx]
            im = ax.imshow(data, cmap=colormap, vmin=vmin_global, vmax=vmax_global, origin='lower')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f'{comp_name.title()} Value', fontsize=20)

            # Set title and labels
            time_seconds = t_idx * (metadata.get('stride_length', 1.0) if metadata else 1.0)
            ax.set_title(f'{sample_id} - {comp_name.title()} Field\nTimestep {t_idx} ({time_seconds + 3:.1f}s)',
                         fontsize=24, fontweight='bold')
            ax.set_xlabel('Grid X', fontsize=22)
            ax.set_ylabel('Grid Y', fontsize=22)

            # Add grid
            # ax.grid(True, alpha=0.3)

            # Save figure in sample-specific directory
            filename = f"{sample_id}_{comp_name}_timestep_{t_idx:04d}.png"
            filepath = os.path.join(sample_heatmap_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')

            pdf_filename = f"{sample_id}_{comp_name}_timestep_{t_idx:04d}.pdf"
            pdf_filepath = os.path.join(sample_heatmap_pdf_dir, pdf_filename)
            plt.savefig(pdf_filepath, bbox_inches='tight', facecolor='white')
            plt.close()

    # Create GIFs from the generated heatmaps (automatically enabled when creating visualizations)
    print("Creating GIFs from heatmap images...")
    created_gifs = create_all_component_gifs(
        sample_id, sample_heatmap_dir, gif_dir,
        components_to_process=component_names,  # Use the same components as visualization
        duration=args.gif_duration
    )
    print(f"Created {len(created_gifs)} GIF files from heatmaps")

    print(f"Visualization completed. Heatmaps saved to: {sample_heatmap_dir}")



def create_gif_from_heatmaps(sample_id: str, heatmap_dir: str, gif_dir: str, component: str = 'pressure',
                             output_filename: str = None, duration: float = 0.2, loop: int = 0):
    """
    Create a GIF animation from existing heatmap PNG files with improved color handling.

    Args:
        sample_id: Sample identifier (e.g., "R0078")
        heatmap_dir: Directory containing the heatmap PNG files
        gif_dir: Directory to save the GIF file
        component: Component name (pressure, velocity_x, velocity_y)
        output_filename: Output GIF filename (if None, auto-generated)
        duration: Duration between frames in seconds
        loop: Number of loops (0 = infinite loop)
    """
    import glob
    from PIL import Image
    import subprocess
    import os

    print(f"Creating GIF from heatmaps for {sample_id} - {component}...")

    # Find all heatmap files for the specified component
    pattern = os.path.join(heatmap_dir, f"{sample_id}_{component}_timestep_*.png")
    heatmap_files = sorted(glob.glob(pattern))

    if not heatmap_files:
        print(f"No heatmap files found matching pattern: {pattern}")
        return None

    print(f"Found {len(heatmap_files)} heatmap files")

    # Generate output filename if not provided
    if output_filename is None:
        output_filename = f"{sample_id}_{component}_heatmaps.gif"

    output_path = os.path.join(gif_dir, output_filename)

    # Method 1: Try ffmpeg for better quality (no color quantization issues)
    mp4_output = output_path.replace('.gif', '.mp4')
    try:
        # Create file list for ffmpeg
        filelist_path = os.path.join(gif_dir, f"temp_filelist_{component}.txt")
        with open(filelist_path, 'w') as f:
            for filepath in heatmap_files:
                f.write(f"file '{filepath}'\n")
                f.write(f"duration {duration}\n")

        # Use ffmpeg to create high-quality MP4
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', filelist_path,
            '-vf', 'fps=5,scale=1280:-1:flags=lanczos',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            mp4_output
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"High-quality MP4 created: {mp4_output}")

            # Also create GIF from MP4 for compatibility
            palette_file = output_path.replace('.gif', '_palette.png')
            gif_cmd = [
                'ffmpeg', '-y', '-i', mp4_output,
                '-vf', 'fps=5,scale=640:-1:flags=lanczos,palettegen=reserve_transparent=0',
                '-y', palette_file
            ]
            subprocess.run(gif_cmd, capture_output=True)

            gif_cmd2 = [
                'ffmpeg', '-y', '-i', mp4_output, '-i', palette_file,
                '-lavfi', 'fps=5,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3',
                output_path
            ]
            result2 = subprocess.run(gif_cmd2, capture_output=True)

            if result2.returncode == 0:
                print(f"High-quality GIF created: {output_path}")

                # Clean up temporary files
                try:
                    if os.path.exists(filelist_path):
                        os.remove(filelist_path)
                        print(f"Cleaned up temp file: {filelist_path}")
                    if os.path.exists(mp4_output):
                        os.remove(mp4_output)
                        print(f"Cleaned up MP4 file: {mp4_output}")
                    if os.path.exists(palette_file):
                        os.remove(palette_file)
                        print(f"Cleaned up palette file: {palette_file}")
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temporary files: {cleanup_error}")

                return output_path
        else:
            # Clean up temp file if ffmpeg failed
            if os.path.exists(filelist_path):
                os.remove(filelist_path)

    except Exception as e:
        print(f"ffmpeg method failed: {e}")
        # Clean up temp file if it exists
        try:
            if os.path.exists(filelist_path):
                os.remove(filelist_path)
        except:
            pass

    # Method 2: PIL with dithering (fallback)
    print("Using PIL fallback method with improved dithering...")

    try:
        # Load images and convert to RGB
        images = []
        for filepath in heatmap_files:
            img = Image.open(filepath)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)

        # Create a global palette from all images for consistency
        print("Generating optimal color palette...")

        # Combine all images to create a representative palette
        combined_img = Image.new('RGB', (images[0].width * len(images), images[0].height))
        for i, img in enumerate(images):
            combined_img.paste(img, (i * img.width, 0))

        # Generate palette
        palette_img = combined_img.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE)
        palette = palette_img.getpalette()

        # Apply the same palette to all images with dithering
        quantized_images = []
        for img in images:
            # Apply dithering for smoother gradients
            quantized = img.quantize(palette=palette_img, dither=Image.FLOYDSTEINBERG)
            quantized_images.append(quantized)

        # Save GIF with consistent palette
        quantized_images[0].save(
            output_path,
            save_all=True,
            append_images=quantized_images[1:],
            duration=int(duration * 1000),
            loop=loop,
            optimize=False  # Don't optimize to maintain palette consistency
        )

        print(f"GIF saved with improved dithering: {output_path}")
        print(f"Total frames: {len(images)}")
        print(f"Duration per frame: {duration}s")

        return output_path

    except Exception as e:
        print(f"PIL dithering method failed: {e}")

        # Method 3: Simple fallback
        try:
            images = []
            for filepath in heatmap_files:
                img = Image.open(filepath)
                images.append(img)

            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=int(duration * 1000),
                loop=loop,
                optimize=True
            )

            print(f"GIF saved with basic method: {output_path}")
            return output_path

        except Exception as e2:
            print(f"All methods failed: {e2}")
            return None


def create_all_component_gifs(sample_id: str, heatmap_dir: str, gif_dir: str,
                              components_to_process: List[str] = None, duration: float = 0.2):
    """
    Create GIF animations for specified components.

    Args:
        sample_id: Sample identifier
        heatmap_dir: Directory containing heatmap files
        gif_dir: Directory to save GIF files
        components_to_process: List of component names to process (if None, process all available)
        duration: Duration between frames in seconds

    Returns:
        List of created GIF file paths
    """
    # Use provided components or default to all
    if components_to_process is None:
        components_to_process = ['pressure', 'velocity_x', 'velocity_y']

    created_gifs = []

    for component in components_to_process:
        # Check if heatmaps exist for this component
        pattern = os.path.join(heatmap_dir, f"{sample_id}_{component}_timestep_*.png")
        heatmap_files = glob.glob(pattern)

        if heatmap_files:
            gif_path = create_gif_from_heatmaps(
                sample_id=sample_id,
                heatmap_dir=heatmap_dir,
                gif_dir=gif_dir,
                component=component,
                duration=duration
            )
            if gif_path:
                created_gifs.append(gif_path)
        else:
            print(f"No heatmaps found for component: {component}")

    return created_gifs


def analyze_fft_peak_frequencies(internal_states: np.ndarray, sample_id: str, args, metadata: Dict = None):
    """
    Perform FFT analysis on each pressure neuron along the time axis and create a heatmap
    of peak frequencies corresponding to the maximum amplitude in the FFT spectrum.

    Modified to use FDTD physical time scale for accurate frequency analysis.

    Args:
        internal_states: Array of shape (time_steps, grid_size, grid_size, 3) where channel 0 is pressure
        sample_id: Sample identifier (e.g., "R0078")
        args: Command line arguments
        metadata: Optional metadata dictionary (dt can be specified, defaults to 0.01)
    """
    print(f"Performing FFT analysis for sample: {sample_id}")

    # Get colormap based on parameter
    colormap = get_colormap(args.colormap)

    # Create FFT output directory
    fft_dir = args.fft_dir
    os.makedirs(fft_dir, exist_ok=True)
    fft_pdf_dir = args.fft_pdf_dir
    os.makedirs(fft_pdf_dir, exist_ok=True)

    # Extract pressure field (channel 0)
    pressure_data = internal_states[:, :, :, 0]  # Shape: (time_steps, grid_size, grid_size)
    time_steps, grid_size_y, grid_size_x = pressure_data.shape

    print(f"Pressure data shape: {pressure_data.shape}")
    print(f"Analyzing {grid_size_x * grid_size_y} neurons across {time_steps} timesteps")

    # Use FDTD physical time step for accurate frequency analysis
    dt = metadata.get('dt', 0.01) if metadata else 0.01  # FDTD time step in seconds
    sampling_freq = 1.0 / dt  # Hz (FDTD internal sampling rate)

    # Also get stride_length for reference (how often PANN features are updated)
    stride_length = metadata.get('stride_length', 1.0) if metadata else 1.0
    pann_update_freq = 1.0 / stride_length

    print(f"FDTD time step (dt): {dt} s")
    print(f"FDTD sampling frequency: {sampling_freq:.1f} Hz")
    print(f"PANN feature update frequency: {pann_update_freq:.3f} Hz (stride: {stride_length}s)")
    print(f"Using colormap: {args.colormap}")

    # Initialize array to store peak frequencies
    peak_frequencies = np.zeros((grid_size_y, grid_size_x))
    peak_amplitudes = np.zeros((grid_size_y, grid_size_x))

    # Perform FFT analysis for each neuron
    print("Computing FFT for each neuron...")

    for i in tqdm(range(grid_size_y), desc="Processing grid rows"):
        for j in range(grid_size_x):
            # Extract time series for this neuron
            neuron_signal = pressure_data[:, i, j]

            # Remove DC component (mean)
            neuron_signal = neuron_signal - np.mean(neuron_signal)

            # Apply window function to reduce spectral leakage
            windowed_signal = neuron_signal * np.hanning(len(neuron_signal))

            # Compute FFT using FDTD time step
            fft_result = np.fft.fft(windowed_signal)
            fft_magnitude = np.abs(fft_result)

            # Create frequency array using FDTD time step (physical frequencies)
            freqs = np.fft.fftfreq(time_steps, d=dt)
            positive_freq_mask = freqs >= 0

            positive_freqs = freqs[positive_freq_mask]
            positive_magnitudes = fft_magnitude[positive_freq_mask]

            # Find peak frequency (excluding DC component at freq=0)
            if len(positive_freqs) > 1:
                # Skip DC component (first element)
                peak_idx = np.argmax(positive_magnitudes[1:]) + 1
                peak_freq = positive_freqs[peak_idx]
                peak_amp = positive_magnitudes[peak_idx]
            else:
                peak_freq = 0.0
                peak_amp = 0.0

            peak_frequencies[i, j] = peak_freq
            peak_amplitudes[i, j] = peak_amp

    print(f"FFT analysis completed.")
    print(f"Peak frequency range: {np.min(peak_frequencies):.3f} - {np.max(peak_frequencies):.3f} Hz")
    print(f"Peak amplitude range: {np.min(peak_amplitudes):.3f} - {np.max(peak_amplitudes):.3f}")

    # Check if we're in expected gamma band range
    gamma_band_count = np.sum((peak_frequencies >= 30) & (peak_frequencies <= 100))
    total_neurons = grid_size_x * grid_size_y
    gamma_percentage = (gamma_band_count / total_neurons) * 100
    print(f"Neurons with gamma-band activity (30-100 Hz): {gamma_band_count}/{total_neurons} ({gamma_percentage:.1f}%)")

    # Create heatmap visualization - only peak frequencies
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Peak frequencies heatmap with selected colormap
    im = ax.imshow(peak_frequencies, cmap=colormap, origin='lower', aspect='equal')
    ax.set_title(f'{sample_id} - Peak Frequencies (Hz)\nFFT Analysis of Pressure Neurons',
                 fontsize=24, fontweight='bold')
    ax.set_xlabel('Grid X', fontsize=22)
    ax.set_ylabel('Grid Y', fontsize=22)

    # Add colorbar for frequencies
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Peak Frequency (Hz)', fontsize=20)

    # Add grid lines
    # ax.grid(True, alpha=0.3)

    # # Add text with analysis info
    # info_text = (f"Time steps: {time_steps}\n"
    #              f"FDTD dt: {dt} s\n"
    #              f"FDTD sampling: {sampling_freq:.1f} Hz\n"
    #              f"PANN updates: {pann_update_freq:.3f} Hz\n"
    #              f"Grid size: {grid_size_x}×{grid_size_y}\n"
    #              f"Gamma band: {gamma_percentage:.1f}%\n"
    #              f"Colormap: {args.colormap}")
    #
    # fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save the figure
    output_filename = f"{sample_id}_FFT.png"
    output_pdf_filename = f"{sample_id}_FFT.pdf"
    output_path = os.path.join(fft_dir, output_filename)
    output_pdf_path = os.path.join(fft_pdf_dir, output_pdf_filename)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"FFT heatmap saved to: {output_path}")

    # # Save the numerical data as well
    # data_filename = f"{sample_id}_FFT_physical_data.npz"
    # data_path = os.path.join(fft_dir, data_filename)
    #
    # np.savez_compressed(data_path,
    #                     peak_frequencies=peak_frequencies,
    #                     peak_amplitudes=peak_amplitudes,
    #                     sample_id=sample_id,
    #                     fdtd_dt=dt,
    #                     fdtd_sampling_frequency=sampling_freq,
    #                     pann_stride_length=stride_length,
    #                     pann_update_frequency=pann_update_freq,
    #                     time_steps=time_steps,
    #                     grid_size=[grid_size_x, grid_size_y],
    #                     gamma_band_percentage=gamma_percentage,
    #                     colormap=args.colormap)
    #
    # print(f"FFT data saved to: {data_path}")

    return peak_frequencies, peak_amplitudes


def analyze_fft_peak_frequencies_improved(internal_states: np.ndarray, sample_id: str, args, metadata: Dict = None):
    """
    Improved FFT analysis with amplitude thresholding and better handling of low-activity regions
    """
    print(f"Performing improved FFT analysis for sample: {sample_id}")

    # Get colormap and create directories
    colormap = get_colormap(args.colormap)
    fft_dir = args.fft_dir
    os.makedirs(fft_dir, exist_ok=True)

    # Extract pressure field
    pressure_data = internal_states[:, :, :, 0]
    time_steps, grid_size_y, grid_size_x = pressure_data.shape

    # FDTD parameters
    dt = metadata.get('dt', 0.01) if metadata else 0.01
    sampling_freq = 1.0 / dt

    print(f"Pressure data shape: {pressure_data.shape}")
    print(f"FDTD sampling frequency: {sampling_freq:.1f} Hz")

    # Calculate activity statistics for thresholding
    signal_stds = np.std(pressure_data, axis=0)  # (grid_y, grid_x)
    signal_max = np.max(np.abs(pressure_data), axis=0)  # (grid_y, grid_x)

    # Define thresholds
    std_threshold = np.percentile(signal_stds[signal_stds > 0], 75)  # 10th percentile
    max_threshold = np.percentile(signal_max[signal_max > 0], 75)  # 10th percentile

    print(f"Activity thresholds - STD: {std_threshold:.6f}, MAX: {max_threshold:.6f}")

    # Initialize arrays
    peak_frequencies = np.full((grid_size_y, grid_size_x), np.nan)
    peak_amplitudes = np.zeros((grid_size_y, grid_size_x))
    activity_mask = np.zeros((grid_size_y, grid_size_x), dtype=bool)

    # Perform FFT analysis for each neuron
    print("Computing FFT for active neurons...")
    active_count = 0

    for i in tqdm(range(grid_size_y), desc="Processing grid rows"):
        for j in range(grid_size_x):
            neuron_signal = pressure_data[:, i, j]

            # Check if neuron is sufficiently active
            signal_std = signal_stds[i, j]
            signal_amplitude = signal_max[i, j]

            if signal_std < std_threshold or signal_amplitude < max_threshold:
                # Mark as inactive
                peak_frequencies[i, j] = 0.0  # or np.nan
                peak_amplitudes[i, j] = 0.0
                activity_mask[i, j] = False
                continue

            active_count += 1
            activity_mask[i, j] = True

            # Remove DC component
            neuron_signal = neuron_signal - np.mean(neuron_signal)

            # Apply window function
            windowed_signal = neuron_signal * np.hanning(len(neuron_signal))

            # Compute FFT
            fft_result = np.fft.fft(windowed_signal)
            fft_magnitude = np.abs(fft_result)

            # Create frequency array
            freqs = np.fft.fftfreq(time_steps, d=dt)
            positive_freq_mask = freqs >= 0

            positive_freqs = freqs[positive_freq_mask]
            positive_magnitudes = fft_magnitude[positive_freq_mask]

            # Find peak frequency (excluding DC)
            if len(positive_freqs) > 1:
                peak_idx = np.argmax(positive_magnitudes[1:]) + 1
                peak_freq = positive_freqs[peak_idx]
                peak_amp = positive_magnitudes[peak_idx]

                # Additional validation: check if peak is significant
                noise_level = np.mean(positive_magnitudes[1:])  # Average excluding DC
                if peak_amp < 2 * noise_level:  # Peak should be at least 2x noise
                    peak_freq = 0.0
                    peak_amp = 0.0
            else:
                peak_freq = 0.0
                peak_amp = 0.0

            peak_frequencies[i, j] = peak_freq
            peak_amplitudes[i, j] = peak_amp

    print(f"FFT analysis completed.")
    print(
        f"Active neurons: {active_count}/{grid_size_x * grid_size_y} ({100 * active_count / (grid_size_x * grid_size_y):.1f}%)")

    # Calculate statistics only for active neurons
    active_frequencies = peak_frequencies[activity_mask & (peak_frequencies > 0)]
    if len(active_frequencies) > 0:
        print(f"Active neuron frequency range: {np.min(active_frequencies):.3f} - {np.max(active_frequencies):.3f} Hz")

        # Band analysis for active neurons only
        theta_count = np.sum((active_frequencies >= 4) & (active_frequencies <= 8))
        alpha_count = np.sum((active_frequencies >= 8) & (active_frequencies <= 12))
        beta_count = np.sum((active_frequencies >= 13) & (active_frequencies <= 30))
        gamma_count = np.sum((active_frequencies >= 30) & (active_frequencies <= 100))

        print(f"Frequency band distribution (active neurons only):")
        print(f"  Theta (4-8 Hz): {theta_count} ({100 * theta_count / len(active_frequencies):.1f}%)")
        print(f"  Alpha (8-12 Hz): {alpha_count} ({100 * alpha_count / len(active_frequencies):.1f}%)")
        print(f"  Beta (13-30 Hz): {beta_count} ({100 * beta_count / len(active_frequencies):.1f}%)")
        print(f"  Gamma (30-100 Hz): {gamma_count} ({100 * gamma_count / len(active_frequencies):.1f}%)")
    else:
        print("No active neurons with significant oscillations found.")

    # Create visualization with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Left: Peak frequencies (all neurons)
    im1 = ax1.imshow(peak_frequencies, cmap=colormap, origin='lower', aspect='equal')
    ax1.set_title(f'{sample_id} - Peak Frequencies (Hz)\nAll Neurons', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Grid X', fontsize=22)
    ax1.set_ylabel('Grid Y', fontsize=22)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Peak Frequency (Hz)', fontsize=24)

    # Right: Activity mask overlay
    frequencies_masked = np.where(activity_mask, peak_frequencies, np.nan)
    im2 = ax2.imshow(frequencies_masked, cmap=colormap, origin='lower', aspect='equal')
    ax2.set_title(f'{sample_id} - Peak Frequencies (Hz)\nActive Neurons Only', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Grid X', fontsize=22)
    ax2.set_ylabel('Grid Y', fontsize=22)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Peak Frequency (Hz)', fontsize=24)

    plt.tight_layout()

    # Save the figure
    output_filename = f"{sample_id}_FFT_improved.pdf"
    output_path = os.path.join(fft_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Improved FFT analysis saved to: {output_path}")

    return peak_frequencies, peak_amplitudes, activity_mask


def analyze_temporal_frequency_changes(internal_states: np.ndarray, sample_id: str, drift_timepoint: int,
                                       args, metadata: Dict = None, window_size: int = 10):
    """
    Compare frequency patterns before and after a drift detection event

    Args:
        internal_states: Array of shape (time_steps, grid_size, grid_size, 3)
        sample_id: Sample identifier
        drift_timepoint: The timestep where drift was detected
        window_size: Number of timesteps before/after to analyze
        metadata: Optional metadata dictionary
    """
    print(f"Analyzing temporal frequency changes for {sample_id} around timestep {drift_timepoint}")

    # Parameters
    dt = metadata.get('dt', 0.01) if metadata else 0.01
    pressure_data = internal_states[:, :, :, 0]
    time_steps, grid_size_y, grid_size_x = pressure_data.shape

    # Ensure we have enough data before and after
    start_before = max(0, drift_timepoint - window_size)
    end_before = drift_timepoint
    start_after = drift_timepoint
    end_after = min(time_steps, drift_timepoint + window_size)

    print(f"Analyzing periods: [{start_before}:{end_before}] vs [{start_after}:{end_after}]")

    # Extract data periods
    data_before = pressure_data[start_before:end_before, :, :]
    data_after = pressure_data[start_after:end_after, :, :]

    def compute_frequency_map(data_period, period_name=""):
        """Compute frequency map for a time period with independent thresholding"""
        period_steps = data_period.shape[0]
        frequencies = np.zeros((grid_size_y, grid_size_x))
        amplitudes = np.zeros((grid_size_y, grid_size_x))

        # Activity threshold (75th percentile, computed independently for this period)
        signal_stds = np.std(data_period, axis=0)
        valid_stds = signal_stds[signal_stds > 0]

        if len(valid_stds) > 0:
            std_threshold = np.percentile(valid_stds, 75)
            print(f"{period_name} threshold: {std_threshold:.6f}")
        else:
            std_threshold = 0

        active_count = 0
        for i in range(grid_size_y):
            for j in range(grid_size_x):
                neuron_signal = data_period[:, i, j]

                # Check activity threshold
                if np.std(neuron_signal) < std_threshold:
                    frequencies[i, j] = np.nan
                    amplitudes[i, j] = 0
                    continue

                active_count += 1

                # Remove DC and apply window
                neuron_signal = neuron_signal - np.mean(neuron_signal)
                windowed_signal = neuron_signal * np.hanning(len(neuron_signal))

                # FFT analysis
                fft_result = np.fft.fft(windowed_signal)
                fft_magnitude = np.abs(fft_result)
                freqs = np.fft.fftfreq(period_steps, d=dt)

                positive_freq_mask = freqs >= 0
                positive_freqs = freqs[positive_freq_mask]
                positive_magnitudes = fft_magnitude[positive_freq_mask]

                # Find peak (excluding DC)
                if len(positive_freqs) > 1:
                    peak_idx = np.argmax(positive_magnitudes[1:]) + 1
                    frequencies[i, j] = positive_freqs[peak_idx]
                    amplitudes[i, j] = positive_magnitudes[peak_idx]
                else:
                    frequencies[i, j] = 0.0
                    amplitudes[i, j] = 0.0

        print(f"{period_name} active neurons: {active_count}/{grid_size_x * grid_size_y}")
        return frequencies, amplitudes

    # Compute frequency maps with independent thresholding
    freq_before, amp_before = compute_frequency_map(data_before, "Before period")
    freq_after, amp_after = compute_frequency_map(data_after, "After period")

    # Get colormap (assuming get_colormap function exists like in original script)
    colormap = get_colormap(args.colormap)

    # Create visualization - only first two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Color limits for consistency
    vmin, vmax = 0, 50

    # Before drift
    im1 = axes[0].imshow(freq_before, cmap=colormap, vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title(f'Before Drift\n(Steps {int(start_before + 3)}-{int(end_before + 3)}s)', fontsize=23,
                      fontweight='bold')
    axes[0].set_xlabel('Grid X', fontsize=22)
    axes[0].set_ylabel('Grid Y', fontsize=22)

    # After drift
    im2 = axes[1].imshow(freq_after, cmap=colormap, vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title(f'After Drift\n(Steps {int(start_after+3)}-{min(int(end_after+3), 60)}s)', fontsize=23,
                      fontweight='bold')
    axes[1].set_xlabel('Grid X', fontsize=22)
    axes[1].set_ylabel('Grid Y', fontsize=22)

    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Peak Frequency (Hz)', fontsize=20)
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Peak Frequency (Hz)', fontsize=20)

    # plt.suptitle(f'{sample_id} - Frequency Analysis Around Drift Event (Step {drift_timepoint})',
    #              fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    fft_dir = args.fft_dir
    os.makedirs(fft_dir, exist_ok=True)
    output_filename = f"{sample_id}_temporal_freq_comparison_step{drift_timepoint}.pdf"
    output_path = os.path.join(fft_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Statistical analysis
    print("\nStatistical Summary:")
    print(f"Active neurons before: {np.sum(~np.isnan(freq_before))}")
    print(f"Active neurons after: {np.sum(~np.isnan(freq_after))}")

    print(f"Temporal frequency comparison saved to: {output_path}")

    return freq_before, freq_after

def main():
    """Main execution function"""
    args = parse_arguments()

    print(f"Internal State Script - Mode: {args.mode}")
    print(f"Sample: {args.sample_index}")
    print(f"Colormap: {args.colormap}")

    # Process mode or both
    if args.mode in ['process', 'both']:
        print("\n" + "=" * 50)
        print("PROCESSING MODE")
        print("=" * 50)

        filepath = process_sample_and_save_states(args.sample_index, args)
        print(f"Processing completed. States saved to: {filepath}")

    # Visualize mode or both
    if args.mode in ['visualize', 'both']:
        print("\n" + "=" * 50)
        print("VISUALIZATION MODE")
        print("=" * 50)

        # Load internal states
        internal_states, metadata = load_internal_states(args.sample_index, args.internal_state_dir)

        # Create visualizations
        visualize_internal_states(internal_states, args.sample_index, args, metadata)

        # Create sample-specific directories for output paths
        sample_heatmap_dir, sample_heatmap_pdf_dir, gif_dir = create_sample_directories(
            args.sample_index, args.visualization_dir, args.gif_dir
        )

        print(f"Visualization completed.")
        print(f"Heatmaps saved to: {sample_heatmap_dir}")
        print(f"GIFs saved to: {gif_dir}")

    # FFT analysis mode
    if args.mode in ['fft', 'both'] or args.perform_fft:
        print("\n" + "=" * 50)
        print("FFT ANALYSIS MODE")
        print("=" * 50)

        # Load internal states
        internal_states, metadata = load_internal_states(args.sample_index, args.internal_state_dir)

        # Perform FFT analysis
        peak_frequencies, peak_amplitudes = analyze_fft_peak_frequencies(
            internal_states, args.sample_index, args, metadata
        )
        # peak_frequencies, peak_amplitudes, activity_mask = analyze_fft_peak_frequencies_improved(
        #     internal_states, args.sample_index, args, metadata
        # )
        freq_before, freq_after = \
            analyze_temporal_frequency_changes(internal_states, "R0016", 40, args, metadata)
            # analyze_temporal_frequency_changes(internal_states, "R0056", 44, args, metadata)

        print(f"FFT analysis completed.")
        print(f"Results saved to: {args.fft_dir}")

    # Create GIF from existing heatmaps (can be run independently)
    if args.create_gif_from_heatmaps:
        print("\n" + "=" * 50)
        print("GIF CREATION MODE")
        print("=" * 50)

        # Create directories
        sample_heatmap_dir, sample_heatmap_pdf_dir, gif_dir = create_sample_directories(
            args.sample_index, args.visualization_dir, args.gif_dir
        )

        created_gifs = create_all_component_gifs(
            sample_id=args.sample_index,
            heatmap_dir=sample_heatmap_dir,
            gif_dir=gif_dir,
            components_to_process=None,  # Process all available components in independent mode
            duration=args.gif_duration
        )
        print(f"Created {len(created_gifs)} GIF files from existing heatmaps")
        print(f"GIFs saved to: {gif_dir}")

    print("\nScript completed successfully!")


if __name__ == "__main__":
    main()