import argparse
import os
import sys
import time
import numpy as np
import torch
from typing import Dict, List

# Path settings.
sys.path.append('/home/zhyuan/Desktop/PCD/baselines')
from DriftLens import DriftLensProbDetector

sys.path.append(os.path.dirname(__file__))
from pann_backbone import load_pretrained_pann, create_sliding_windows, load_audio_file


def parse_arguments():
    p = argparse.ArgumentParser(description='DriftLens Probability+EMA (4s window, 1s stride)')
    # IO
    p.add_argument('--audio_path', type=str, default='/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio/R0031_segment_ambisonics.wav')
    p.add_argument('--pann_checkpoint', type=str, default='/home/zhyuan/Desktop/audioset_tagging_cnn/Cnn14_mAP=0.431.pth')
    p.add_argument('--output_dir', type=str, default='./baseline_results')

    # Windowing
    p.add_argument('--window_length', type=float, default=4.0)
    p.add_argument('--stride_length', type=float, default=1.0)
    p.add_argument('--sample_rate', type=int, default=32000)

    # default.
    p.add_argument('Implementation note.')
    p.add_argument('--metric', type=str, choices=['kl', 'js', 'bhattacharyya'], default='js')
    p.add_argument('--threshold_percentile', type=float, default=0.995)
    p.add_argument('--init_trim', type=float, default=0.1)
    p.add_argument('--temp_scale', type=float, default=1.0)

    # Online.
    p.add_argument('--ema_alpha', type=float, default=0.0005)
    p.add_argument('--ema_only_if_no_drift', action='store_true', default=True)
    p.add_argument('--ema_warmup', type=int, default=10)

    # Implementation note.
    p.add_argument('Implementation note.')
    p.add_argument('Implementation note.')
    p.add_argument('Implementation note.')

    # Post-processing.
    p.add_argument('--min_change_interval', type=float, default=6.0)

    # System
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def extract_pann_probs(segments: List[np.ndarray], pann_model, device: str) -> np.ndarray:
    feats = []
    for seg in segments:
        x = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = pann_model(x)
            feats.append(out['clipwise_output'].detach().cpu().numpy())
    return np.concatenate(feats, axis=0)  # [T, 527]


def timestamps_for_window(idx: int, win_len: float, stride: float) -> Dict[str, float]:
    start = idx * stride
    end = start + win_len
    return {"start_time": start, "end_time": end}


def run_pipeline(pann_probs: np.ndarray, args) -> Dict:
    T = pann_probs.shape[0]
    if args.verbose:
        print(f"Windows: {T} (win={args.window_length}s, stride={args.stride_length}s)")

    # Implementation note.
    baseline_last_idx = int(np.floor((args.baseline_seconds - args.window_length) / args.stride_length)) + 1
    if baseline_last_idx <= 0:
        baseline_last_idx = 1
    baseline_idx = np.arange(min(baseline_last_idx, T))
    if len(baseline_idx) == 0:
        raise RuntimeError("No baseline windows selected.")

    Xb = pann_probs[baseline_idx]

    # Initialize.
    det = DriftLensProbDetector(
        metric=args.metric,
        threshold_percentile=args.threshold_percentile,
        init_trim=args.init_trim,
        ema_alpha=args.ema_alpha,
        ema_only_if_no_drift=args.ema_only_if_no_drift,
        ema_warmup=args.ema_warmup,
        temp_scale=args.temp_scale,
    )
    det.fit_baseline(Xb)

    # Online.
    scores = np.zeros(T, dtype=np.float64)
    flags  = np.zeros(T, dtype=bool)
    raw_changes = []

    start_i = len(baseline_idx)
    consec = 0
    for i in range(T):
        if i < start_i:
            # Implementation note.
            scores[i] = det.metric_fn(pann_probs[i], det.p_baseline_)
            continue

        step = det.predict_step(pann_probs[i]) # Implementation note.
        scores[i] = step["score"]
        over = scores[i] > step["threshold"] * args.margin
        consec = consec + 1 if over else 0

        # Implementation note.
        if over and consec >= args.n_consecutive:
            left = max(start_i, i - args.peak_neighbor)
            right = min(T - 1, i + args.peak_neighbor)
            if scores[i] >= scores[left:right + 1].max():
                flags[i] = True
                ts = timestamps_for_window(i, args.window_length, args.stride_length)
                raw_changes.append({
                    "index": i,
                    "start_time": ts["start_time"],
                    "end_time": ts["end_time"],
                    "score": float(scores[i]),
                    "threshold": float(step["threshold"]),
                })
                consec = 0 # avoid.

    # Implementation note.
    filtered_changes = []
    last_keep = -1e9
    for ch in raw_changes:
        if ch["start_time"] - last_keep >= args.min_change_interval:
            filtered_changes.append(ch)
            last_keep = ch["start_time"]

    return {
        "scores": scores,
        "threshold": float(det.threshold_),
        "baseline_count": int(len(baseline_idx)),
        "raw_changes": raw_changes,
        "changes": filtered_changes,
        "total_windows": int(T),
        "params": det.get_params(),
    }


def save_results(results: Dict, output_path: str, args) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # npz
    np.savez(
        output_path.replace('.txt', '.npz'),
        scores=results["scores"],
        threshold=results["threshold"],
        total_windows=results["total_windows"],
        baseline_count=results["baseline_count"]
    )
    # txt
    with open(output_path, 'w') as f:
        f.write("DriftLens Probability+EMA Results\n")
        f.write("=" * 36 + "\n\n")
        f.write(f"Metric: {args.metric.upper()}, Percentile: {args.threshold_percentile:.3f}\n")
        f.write(f"Window: {args.window_length:.1f}s, stride: {args.stride_length:.1f}s\n")
        f.write(f"Baseline seconds: {args.baseline_seconds:.1f}s -> windows: {results['baseline_count']}\n")
        f.write(f"EMA alpha: {args.ema_alpha:.4g}, only_if_no_drift={args.ema_only_if_no_drift}, warmup={args.ema_warmup}\n")
        f.write(f"Temp scale: {args.temp_scale:.2f}\n")
        f.write(f"Trigger: n_consecutive={args.n_consecutive}, margin={args.margin:.2f}, peak_neighbor={args.peak_neighbor}\n")
        f.write(f"Min change interval: {args.min_change_interval:.1f}s\n")
        f.write(f"Threshold: {results['threshold']:.6f}\n")
        f.write(f"Total windows: {results['total_windows']}\n")
        f.write(f"Detections (filtered): {len(results['changes'])}\n\n")
        if results["changes"]:
            f.write("Change Timestamps:\n")
            f.write("-" * 20 + "\n")
            for i, ch in enumerate(results["changes"], 1):
                f.write(f"{i}. {ch['start_time']:.1f}s - {ch['end_time']:.1f}s "
                        f"(score={ch['score']:.6f} > thr={ch['threshold']:.6f})\n")
        else:
            f.write("No changes detected.\n")


def main():
    args = parse_arguments()

    # Device.
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Audio.
    audio, sr = load_audio_file(args.audio_path, args.sample_rate)
    print(f"Audio loaded: {len(audio) / sr:.2f}s @ {sr} Hz")
    segments = create_sliding_windows(audio, sr, args.window_length, args.stride_length)
    print(f"Segments: {len(segments)} windows (win={args.window_length}s, stride={args.stride_length}s)")

    # PANN
    pann_model = load_pretrained_pann(
        checkpoint_path=args.pann_checkpoint,
        device=device,
        sample_rate=args.sample_rate,
        window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527
    )
    pann_probs = extract_pann_probs(segments, pann_model, device)  # [T, 527]

    # Detection.
    t0 = time.time()
    results = run_pipeline(pann_probs, args)
    elapsed = time.time() - t0

    # Save.
    audio_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    out_txt = os.path.join(args.output_dir, f"driftlens_probEMA_{args.metric}_{audio_name}.txt")
    save_results(results, out_txt, args)

    # Console summary.
    print("\nSummary")
    print("-" * 30)
    print(f"Audio: {args.audio_path}")
    print(f"Windows: {results['total_windows']}, Baseline windows: {results['baseline_count']}")
    print(f"Threshold: {results['threshold']:.6f}, Time: {elapsed:.2f}s")
    if results["changes"]:
        times = " ".join([f"{c['start_time']:.1f}s" for c in results['changes']])
        print(f"Detections: {len(results['changes'])} at {times}")
    else:
        print("Detections: None")


if __name__ == "__main__":
    main()