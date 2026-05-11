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
import random
import traceback
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from Qwen_processor import QwenAudioDescription


def parse_args():
    parser = argparse.ArgumentParser(description='Random Baseline for XD-Violence Dataset')

    parser.add_argument("--device", type=str, default="cuda:1",
                        help="Runtime device")
    parser.add_argument("--dataset_path", type=str,
                        default="/home/zhyuan/Desktop/PCD/data/audios",
                        help="Directory containing XD-Violence audio files")
    parser.add_argument("--results_dir", type=str,
                        default="/home/zhyuan/Desktop/Qwen-Audio/results",
                        help="Output directory for JSON results")
    parser.add_argument("--sample_rate", type=int, default=32000,
                        help="Sample rate")
    parser.add_argument("--biooss_json", type=str,
                        default="/home/zhyuan/Desktop/PCD/results_iclr/dataset_xd_cls.json",
                        help="BioOSS result JSON used to match the number of change points per file")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def extract_file_index(file_path):
    name = os.path.basename(file_path)
    return name.replace('.wav', '')


def calculate_time_sent_to_qwen(change_points, total_length):
    """Calculate deduplicated total duration sent to Qwen, matching the main script."""
    if not change_points:
        return 15.0  # Initial 15 seconds
    processed_intervals = [(0, 15)]
    for cp in change_points:
        start = max(0, int(cp - 3))
        end = min(int(cp + 12), int(total_length))
        processed_intervals.append((start, end))

    processed_intervals.sort()
    merged = []
    if not processed_intervals:
        return 0
    curr_start, curr_end = processed_intervals[0]
    for next_start, next_end in processed_intervals[1:]:
        if next_start <= curr_end:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))

    return sum([end - start for start, end in merged])


def generate_random_change_points(n_points, total_duration, window_length=4.0, stride_length=1.0):
    """
    Randomly generate n_points change points as window indices.
    The valid index range is [0, n_windows - 1], where n_windows is determined by total_duration.
    """
    if n_points == 0:
        return []

    # Match the main script: n_windows = floor((total_samples - win_samples) / stride_samples) + 1.
    # This approximation uses seconds.
    max_window_idx = max(0, int((total_duration - window_length) / stride_length))

    if max_window_idx < n_points:
        # Allow repeated indices when too few windows are available.
        return sorted(random.choices(range(max_window_idx + 1), k=n_points))

    return sorted(random.sample(range(max_window_idx + 1), n_points))


def process_single_audio_random(file_path, args, qwen_description, random_change_points):
    """
    Single-file processing for the random baseline:
    - Do not call BioOSSPatternChangeDetector.
    - Use externally generated random change_points.
    - Keep Qwen invocation logic aligned with the main script.
    """
    audio, sr = librosa.load(file_path, sr=args.sample_rate, mono=True)
    total_duration = len(audio) / sr
    temp_dir = "/tmp"

    change_points = random_change_points  # Use the generated random change points directly.

    # Trigger Qwen for detections at or after 15 seconds, matching the main script.
    qwen_processed = [cp for cp in change_points if cp >= 15]
    responses = []

    # Process the initial 15 seconds.
    init_audio = audio[:int(15 * sr)]
    init_path = os.path.join(temp_dir, "init_rand.wav")
    sf.write(init_path, init_audio, sr)
    responses.append({
        "time": 0,
        "description": qwen_description.generate_description(init_path)
    })
    os.remove(init_path)

    # Process randomly selected change-point segments.
    for cp in qwen_processed:
        start_s = max(0, int(cp - 3))
        end_s = min(int(cp + 12), int(total_duration))
        seg = audio[int(start_s * sr):int(end_s * sr)]
        seg_path = os.path.join(temp_dir, f"seg_rand_{cp}.wav")
        sf.write(seg_path, seg, sr)
        responses.append({
            "time": cp,
            "description": qwen_description.generate_description(seg_path)
        })
        os.remove(seg_path)

    return {
        "filename": os.path.basename(file_path),
        "total_length": total_duration,
        "change_points": change_points,        # Randomly generated change points
        "qwen_responses": responses,
        "time_sent_to_qwen": calculate_time_sent_to_qwen(qwen_processed, total_duration)
    }


def main():
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.makedirs(args.results_dir, exist_ok=True)

    # Load BioOSS results and build a filename -> n_change_points map.
    print(f"Loading BioOSS results from: {args.biooss_json}")
    with open(args.biooss_json, 'r', encoding='utf-8') as f:
        biooss_results = json.load(f)

    biooss_map = {
        item["filename"]: len(item.get("change_points", []))
        for item in biooss_results
    }
    print(f"  Loaded {len(biooss_map)} entries from BioOSS JSON.")

    # Initialize Qwen.
    qwen_description = QwenAudioDescription(Queue(), Queue(), device=args.device)

    all_results = []
    audio_files = [f for f in os.listdir(args.dataset_path) if f.endswith(".wav")]

    for i, fname in enumerate(audio_files, 1):
        fpath = os.path.join(args.dataset_path, fname)
        print(f"[{i}/{len(audio_files)}] Processing {fname}...")

        # Look up the corresponding number of BioOSS change points.
        n_cp = biooss_map.get(fname, 0)
        if fname not in biooss_map:
            print(f"    Warning: {fname} not found in BioOSS JSON, defaulting to 0 change points.")

        try:
            # Load duration first so random window indices stay valid.
            audio_info = librosa.get_duration(path=fpath)
            rand_cps = generate_random_change_points(
                n_points=n_cp,
                total_duration=audio_info,
                window_length=4.0,
                stride_length=1.0
            )
            print(f"    BioOSS had {n_cp} change_points → random: {rand_cps}")

            res = process_single_audio_random(fpath, args, qwen_description, rand_cps)
            if res:
                res["saved_ratio"] = 1.0 - (res["time_sent_to_qwen"] / res["total_length"])
                all_results.append(res)
                print(f"    Done. Saved: {res['saved_ratio']*100:.1f}%")

        except Exception as e:
            print(f"    Error: {e}")
            traceback.print_exc()

        gc.collect()
        torch.cuda.empty_cache()

    # Export JSON.
    out_json = os.path.join(args.results_dir, "dataset_xd_cls_random.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"✨ All finished. Results: {out_json}")


if __name__ == "__main__":
    main()
