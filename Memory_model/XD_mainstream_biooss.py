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
import re
import json
import traceback
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side runs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from echoic_memory import EchoicMemory
from biodetector import BioOSSPatternChangeDetector
from Qwen_processor import QwenAudioDescription

# Color palette
reds = ['#8E0D29', '#BB1E38', '#D35B4D', '#F6BCA9']

def parse_args():
    """Parse command line arguments for BioOSSPatternChangeDetector and AudioEncoder."""
    parser = argparse.ArgumentParser(description='BioOSS Dataset Processing for XD-Violence')

    # ==========================
    # 1. Runtime environment and path arguments
    # ==========================
    parser.add_argument("--cuda", action="store_true", default=True, 
                        help="Use CUDA; AudioEncoder depends on this flag")
    parser.add_argument("--device", type=str, default="cuda:1", 
                        help="Runtime device, e.g. cuda:0 or cuda:1")
    parser.add_argument("--dataset_path", type=str, 
                        default="/home/zhyuan/Desktop/PCD/data/audios", 
                        help="Directory containing XD-Violence audio files")
    parser.add_argument("--results_dir", type=str, 
                        default="/home/zhyuan/Desktop/Qwen-Audio/results", 
                        help="Output directory for JSON results and visualizations")
    parser.add_argument("--checkpoint_path", type=str, 
                        default='/home/zhyuan/Desktop/audioset_tagging_cnn/Cnn14_mAP=0.431.pth',
                        help="Path to the pretrained PANNs Cnn14 checkpoint")

    # ==========================
    # 2. PANNs audio encoder arguments used by AudioEncoder
    # ==========================
    parser.add_argument("--sample_rate", type=int, default=32000, 
                        help="Sample rate; 32000 Hz is recommended for XD-Violence")
    parser.add_argument("--window_size", type=int, default=1024, 
                        help="STFT window size; PANNs default is 1024")
    parser.add_argument("--hop_size", type=int, default=320, 
                        help="STFT hop size; PANNs default is 320")
    parser.add_argument("--mel_bins", type=int, default=64, 
                        help="Number of mel bins")
    parser.add_argument("--fmin", type=int, default=50, 
                        help="Minimum frequency")
    parser.add_argument("--fmax", type=int, default=14000, 
                        help="Maximum frequency")
    parser.add_argument("--output_size", type=int, default=527, 
                        help="Number of PANNs semantic output classes; AudioSet uses 527")

    # ==========================
    # 3. BioOSS multi-metric detection weights tuned for violence detection
    # ==========================
    # Recommended setting: emphasize energy for explosions/impacts and cosine distance for vocal shifts such as screams.
    parser.add_argument('--energy_weight', type=float, default=0.6, 
                        help='Energy metric weight; higher values help capture explosions')
    parser.add_argument('--pca_weight', type=float, default=0.0, 
                        help='PCA weight; set to 0 by default to reduce compute')
    parser.add_argument('--cosine_weight', type=float, default=0.4, 
                        help='Cosine distance weight for semantic pattern shifts such as screams')
    parser.add_argument('--consensus_threshold', type=float, default=0.45, 
                        help='Consensus trigger threshold; increase for strong movie background music')
    parser.add_argument('--min_persistence', type=int, default=2, 
                        help='Minimum persistence frames for filtering transient noise')
    parser.add_argument('--min_confidence_threshold', type=float, default=0.3, 
                        help='Minimum confidence for emitted detections')

    # ==========================
    # 4. BioOSS FDTD physical model arguments
    # ==========================
    parser.add_argument('--freq_min', type=float, default=50.0, 
                        help='Minimum BioWM neuron resonance frequency')
    parser.add_argument('--freq_max', type=float, default=1200.0, 
                        help='Maximum BioWM neuron resonance frequency')
    parser.add_argument('--grid_size', type=int, default=64, 
                        help='FDTD simulation grid size, e.g. 64x64')
    parser.add_argument('--dt', type=float, default=0.01, 
                        help='Simulation time step')

    # ==========================
    # 5. Sliding-window and inference strategy
    # ==========================
    parser.add_argument('--window_length', type=float, default=4.0, 
                        help='BioWM processing window length in seconds')
    parser.add_argument('--stride_length', type=float, default=1.0, 
                        help='Stride in seconds between pattern-shift decisions')

    # ==========================
    # 6. Debugging and visualization
    # ==========================
    parser.add_argument('--debug', action='store_true', default=False, 
                        help='Enable debug output')
    parser.add_argument('--verbose_metrics', action='store_true', default=False, 
                        help='Print detailed metrics for every frame')

    return parser.parse_args()

def extract_file_index(file_path):
    # Preserve the full XD-Violence filename stem.
    name = os.path.basename(file_path)
    return name.replace('.wav', '')

def calculate_time_sent_to_qwen(change_points, total_length):
    """Calculate deduplicated total duration sent to Qwen."""
    if not change_points: return 15.0 # Initial 15 seconds
    processed_intervals = [(0, 15)]
    for cp in change_points:
        start = max(0, int(cp - 3))
        end = min(int(cp + 12), int(total_length))
        processed_intervals.append((start, end))
    
    # Merge overlapping intervals.
    processed_intervals.sort()
    merged = []
    if not processed_intervals: return 0
    curr_start, curr_end = processed_intervals[0]
    for next_start, next_end in processed_intervals[1:]:
        if next_start <= curr_end:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))
    
    return sum([end - start for start, end in merged])

def collect_metrics(biooss_result):
    """Extract JSON-compatible metric data."""
    res = {
        'raw_metrics': {'energy': 0.0, 'pca': 0.0, 'cosine': 0.0},
        'thresholds': {'energy': 0.0, 'pca': 0.0, 'cosine': 0.0}
    }
    if biooss_result and 'detailed_metrics' in biooss_result:
        dm = biooss_result['detailed_metrics']
        for k in ['energy', 'pca', 'cosine']:
            res['raw_metrics'][k] = float(dm.get('raw_metrics', {}).get(k, 0.0))
            res['thresholds'][k] = float(dm.get('thresholds', {}).get(k, 0.0))
    return res

def process_single_audio(file_path, args, qwen_description, pattern_detector):
    """Process one audio file and collect curves plus Qwen responses."""
    audio, sr = librosa.load(file_path, sr=args.sample_rate, mono=True)
    total_duration = len(audio) / sr
    temp_dir = "/tmp"
    
    pattern_detector.reset_fields()
    
    # 1. Split into sliding windows.
    win_samples = int(args.window_length * sr)
    stride_samples = int(args.stride_length * sr)
    windows = [audio[s:s+win_samples] for s in range(0, len(audio)-win_samples+1, stride_samples)]
    
    change_points = []
    full_metrics = []

    # 2. Run detection over windows.
    for idx, win in enumerate(windows):
        tmp_path = os.path.join(temp_dir, f"win_{idx}.wav")
        sf.write(tmp_path, win, sr)
        
        conf, bio_res = pattern_detector.detect_pattern_change(tmp_path)
        
        # Record per-second metrics.
        full_metrics.append(collect_metrics(bio_res))
        
        if bio_res and bio_res.get('change_detected', False):
            change_points.append(idx)
        
        os.remove(tmp_path)

    # 3. Trigger Qwen for detections at or after 15 seconds.
    qwen_processed = [cp for cp in change_points if cp >= 15]
    responses = []
    
    # Process the initial 15 seconds.
    init_audio = audio[:int(15*sr)]
    init_path = os.path.join(temp_dir, "init.wav")
    sf.write(init_path, init_audio, sr)
    responses.append({"time": 0, "description": qwen_description.generate_description(init_path)})
    os.remove(init_path)

    # Process detected segments.
    for cp in qwen_processed:
        start_s = max(0, int(cp - 3))
        end_s = min(int(cp + 12), int(total_duration))
        seg = audio[int(start_s*sr):int(end_s*sr)]
        seg_path = os.path.join(temp_dir, f"seg_{cp}.wav")
        sf.write(seg_path, seg, sr)
        responses.append({"time": cp, "description": qwen_description.generate_description(seg_path)})
        os.remove(seg_path)

    # # 4. Visualization
    # figures_dir = os.path.join(args.results_dir, "figures")
    # os.makedirs(figures_dir, exist_ok=True)
    # viz_path = os.path.join(figures_dir, f"{extract_file_index(file_path)}_spec.pdf")
    
    # A create_enhanced_mel_spectrogram_with_detection call can be inserted here.

    return {
        "filename": os.path.basename(file_path),
        "total_length": total_duration,
        "change_points": change_points,
        "energy_curve": [m['raw_metrics']['energy'] for m in full_metrics],
        "threshold_curve": [m['thresholds']['energy'] for m in full_metrics],
        "qwen_responses": responses,
        "time_sent_to_qwen": calculate_time_sent_to_qwen(qwen_processed, total_duration)
    }

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize components.
    pattern_detector = BioOSSPatternChangeDetector(
        args, energy_weight=args.energy_weight, pca_weight=args.pca_weight,
        cosine_weight=args.cosine_weight, consensus_threshold=args.consensus_threshold,
        min_persistence=args.min_persistence, min_confidence_threshold=args.min_confidence_threshold
    )
    
    # Move modules to the requested device.
    if hasattr(pattern_detector, 'biooss_model'): pattern_detector.biooss_model.to(args.device)
    if hasattr(pattern_detector, 'audio_encoder'):
        pattern_detector.audio_encoder.model.to(args.device)
        pattern_detector.audio_encoder.device = args.device

    qwen_description = QwenAudioDescription(Queue(), Queue(), device='cuda:1')

    all_results = []
    audio_files = [f for f in os.listdir(args.dataset_path) if f.endswith(".wav")]

    for i, fname in enumerate(audio_files, 1):
        fpath = os.path.join(args.dataset_path, fname)
        print(f"[{i}/{len(audio_files)}] Processing {fname}...")
        
        try:
            res = process_single_audio(fpath, args, qwen_description, pattern_detector)
            if res:
                # Compute the processing-time saving ratio.
                res["saved_ratio"] = 1.0 - (res["time_sent_to_qwen"] / res["total_length"])
                all_results.append(res)
                print(f"    Done. Saved: {res['saved_ratio']*100:.1f}%")
        except Exception as e:
            print(f"    Error: {e}")
            traceback.print_exc()

        gc.collect()
        torch.cuda.empty_cache()

    # Export JSON.
    # out_json = os.path.join(args.results_dir, "dataset_analysis_full.json")
    out_json = os.path.join(args.results_dir, "dataset_xd_cls.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    print(f"✨ All finished. Results: {out_json}")

if __name__ == "__main__":
    main()
