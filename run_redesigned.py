# main.py
import argparse
import os
import math
import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Implementation note.
from pann_backbone import PANNBackbone, load_pretrained_pann, create_sliding_windows, load_audio_file
from audio_attention_utils import generate_interpretation, smooth_detections
# Implementation note.
from biooss_pipeline import (
    BioOSSFDTD2D,
    PANNToRegionModulator,
    BatchOnlineBioOSSProcessor,
    build_biooss_pipeline
)

# ===========================
# Detection.
# ===========================

class SimpleEnergyChangeDetector:
    """
    log-energy note + note/MAD note
    """
    def __init__(self, k: float = 6.0, min_hist: int = 20, debug_print: bool = False):
        self.k = k
        self.min_hist = min_hist
        self.prev_logE = None
        self.grad_hist = []
        self.debug_print = debug_print
        self._dbg_counter = 0

    def update(self, energy: float, state_vector: torch.Tensor, timestamp: int):
        logE = math.log1p(max(energy, 0.0))
        if self.prev_logE is None:
            self.prev_logE = logE
            return False, 0.0, {
                'raw_metrics': {'energy': 0.0, 'pca': 0.0, 'cosine': 0.0},
                'thresholds': {'energy': None, 'pca': None, 'cosine': None}
            }

        grad = abs(logE - self.prev_logE)
        self.prev_logE = logE

        self.grad_hist.append(grad)
        if len(self.grad_hist) > 500:
            self.grad_hist = self.grad_hist[-500:]

        if len(self.grad_hist) < self.min_hist:
            thr = None
            is_change, conf = False, 0.0
        else:
            arr = np.array(self.grad_hist, dtype=np.float32)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) + 1e-6
            thr = med + self.k * mad
            is_change = grad > thr
            conf = 0.0 if not is_change else min(1.0, (grad - thr) / (thr + 1e-6) + 0.2)

        if self.debug_print and self._dbg_counter < 60:
            print(f"[DBG] t#{timestamp:05d} grad={grad:.4g} thr={thr if thr is not None else -1} det={is_change}")
            self._dbg_counter += 1

        return is_change, conf, {
            'raw_metrics': {'energy': grad, 'pca': 0.0, 'cosine': 0.0},
            'thresholds': {'energy': thr, 'pca': None, 'cosine': None}
        }


class AbsoluteEnergyDetector:
    """
    note log-energy note（note）
    """
    def __init__(self, k: float = 3.0, min_hist: int = 20):
        self.k = k
        self.min_hist = min_hist
        self.logE_hist = []

    def feed(self, logE: float) -> Dict[str, float]:
        self.logE_hist.append(logE)
        if len(self.logE_hist) > 1000:
            self.logE_hist = self.logE_hist[-1000:]
        out = {'hit': False, 'thr': None}
        if len(self.logE_hist) >= self.min_hist:
            arr = np.array(self.logE_hist[-300:], dtype=np.float32)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) + 1e-6
            thr = med + self.k * mad
            out['thr'] = thr
            out['hit'] = (logE >= thr)
        return out

# ===========================
# Parameters.
# ===========================

def parse_arguments():
    parser = argparse.ArgumentParser(description='BioOSS Audio Change Detection (Region-based, Causal OLA)')

    # IO
    parser.add_argument('--audio_path', type=str,
                        default='/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio/R0056_segment_ambisonics.wav')
    parser.add_argument('--pann_checkpoint', type=str,
                        default='/home/zhyuan/Desktop/audioset_tagging_cnn/Cnn14_mAP=0.431.pth')
    parser.add_argument('--output_dir', type=str, default='./results')

    # Audio
    parser.add_argument('--window_length', type=float, default=4.0)
    parser.add_argument('--stride_length', type=float, default=1.0)
    parser.add_argument('--sample_rate', type=int, default=32000)

    # BioOSS core
    parser.add_argument('--freq_min', type=float, default=40.0)
    parser.add_argument('--freq_max', type=float, default=80.0)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--region_rows', type=int, default=8)
    parser.add_argument('--region_cols', type=int, default=8)
    parser.add_argument('--kp', type=float, default=0.01)
    parser.add_argument('--ko', type=float, default=0.01)
    parser.add_argument('Implementation note.')

    # Detector sampling and gating.
    parser.add_argument('--detect_hop_s', type=float, default=0.05, help='Detection interval in seconds.')
    parser.add_argument('--min_persist_s', type=float, default=0.25, help='Minimum event duration in seconds.')
    parser.add_argument('--cooldown_s', type=float, default=0.8, help='Event-level cooldown in seconds.')
    parser.add_argument('--warmup_s', type=float, default=1.0, help='Skip detection during the initial warmup interval.')
    parser.add_argument('--ema_tau_s', type=float, default=0.10, help='Energy EMA time constant in seconds; set to 0 to disable.')
    parser.add_argument('--nms_sep_s', type=float, default=1.0, help='Minimum separation for event-level NMS merging.')
    parser.add_argument('--tail_flush_s', type=float, default=0.30, help='Tail flushing duration in seconds.')

    # Detector type and sensitivity.
    parser.add_argument('--detector', type=str, default='simple',
                        choices=['simple'], help='Unsupervised detector core to use.')
    parser.add_argument('--simple_k', type=float, default=5.0,
                        help='MAD multiplier for SimpleEnergyChangeDetector; lower is more sensitive.')
    parser.add_argument('--persist_frac', type=float, default=0.5,
                        help='Majority-vote persistence ratio.')
    parser.add_argument('--peak_gate', type=float, default=3.0,
                        help='Peak gate on robust_ratio = metric / median_lag(threshold).')

    # Implementation note.
    parser.add_argument('Implementation note.')
    parser.add_argument('Implementation note.')

    # Parameters.
    parser.add_argument('--profile', type=str, default='default',
                        choices=['conservative', 'default', 'sensitive'],
                        help='Implementation note.')

    # Viz / debug
    parser.add_argument('--plot_spectrogram', default=True, type=bool)
    parser.add_argument('--plot_metrics', default=True, type=bool)
    parser.add_argument('Implementation note.')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


def apply_profile(args):
    """ Parameters. """
    if args.profile == 'conservative':
        args.detect_hop_s = 0.06
        args.min_persist_s = 0.30
        args.cooldown_s = 1.2
        args.ema_tau_s = 0.12
        args.mod_alpha = max(args.mod_alpha, 4.0)
        args.kp = 0.015
        args.ko = 0.015
        args.simple_k = max(args.simple_k, 5.5)
        args.persist_frac = max(args.persist_frac, 0.6)
        args.peak_gate = max(args.peak_gate, 3.5)
    elif args.profile == 'sensitive':
        args.detect_hop_s = 0.04
        args.min_persist_s = 0.12 # Implementation note.
        args.cooldown_s = 0.6
        args.ema_tau_s = 0.00 # Implementation note.
        args.mod_alpha = max(args.mod_alpha, 8.0)
        args.kp = 0.008
        args.ko = 0.008
        args.simple_k = min(args.simple_k, 4.5)
        args.persist_frac = min(args.persist_frac, 0.5)
        args.peak_gate = min(args.peak_gate, 2.6)
    # Parameters.
    return args

# ===========================
# Implementation note.
# ===========================

def load_and_preprocess_audio(audio_path: str, sample_rate: int) -> Tuple[np.ndarray, int]:
    print(f"Loading audio from: {audio_path}")
    audio, sr = load_audio_file(audio_path, sample_rate)
    print(f"Audio loaded: {len(audio) / sr:.2f}s, {sr}Hz")
    return audio, sr

def extract_pann_features(audio_segments: List[np.ndarray], pann_model: PANNBackbone, device: str) -> torch.Tensor:
    print(f"Extracting PANN features from {len(audio_segments)} segments...")
    feats = []
    for i, segment in enumerate(audio_segments):
        if i % 10 == 0:
            print(f"  segment {i + 1}/{len(audio_segments)}")
        seg = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = pann_model(seg)
            clipwise = out['clipwise_output']  # (1,527)
        feats.append(clipwise.cpu())
    feats = torch.cat(feats, dim=0)  # (M,527)
    print(f"PANN features extracted: {feats.shape}")
    return feats

def merge_events(timestamps: List[float], detections: List[bool], confidences: List[float], min_sep_s: float):
    idxs = [i for i, d in enumerate(detections) if d]
    if not idxs:
        return []
    merged = [idxs[0]]
    for i in idxs[1:]:
        if timestamps[i] - timestamps[merged[-1]] >= min_sep_s:
            merged.append(i)
        else:
            if confidences[i] > confidences[merged[-1]]:
                merged[-1] = i
    return merged

# ===========================
# Implementation note.
# Threshold.
# ===========================

def run_biooss_stream(
    pann_frames: torch.Tensor,
    processor: BatchOnlineBioOSSProcessor,
    detector: SimpleEnergyChangeDetector,
    dt: float,
    stride_s: float,
    tail_flush_s: float = 0.30,
    max_time_s: float = None,
    debug: bool = False,
    detect_hop_s: float = 0.05,
    min_persist_s: float = 0.25,
    cooldown_s: float = 0.8,
    warmup_s: float = 1.0,
    ema_tau_s: float = 0.10,
    trace_energy: bool = False,
    persist_frac: float = 0.5, # default.
    peak_gate: float = 3.0, # Threshold.
    abs_gate: bool = False, # Implementation note.
    abs_gate_k: float = 3.0
) -> Dict:
    M = pann_frames.shape[0]
    stride_steps = max(1, int(round(stride_s / dt)))
    tail_steps = max(0, int(round(tail_flush_s / dt)))
    hop_steps = max(1, int(round(detect_hop_s / dt)))
    persist_steps = max(1, int(round(min_persist_s / detect_hop_s)))
    cooldown_steps = max(1, int(round(cooldown_s / detect_hop_s)))
    warmup_steps = int(round(warmup_s / detect_hop_s))
    need_votes = max(1, math.ceil(persist_frac * persist_steps))

    results = {'detections': [], 'confidence_scores': [], 'detailed_metrics': [],
               'timestamps': [], 'interpretations': []}

    pooled_energies, pooled_state_vecs = [], []
    ema_energy = None
    ema_alpha = 0.0 if ema_tau_s <= 0 else 1.0 - math.exp(-detect_hop_s / max(ema_tau_s, 1e-6))

    last_event_hop = -10**9
    hop_counter = 0
    t_seconds = 0.0

    det_window = deque(maxlen=persist_steps) # Implementation note.
    ratio_window = deque(maxlen=persist_steps) # Implementation note.
    thr_window   = deque(maxlen=24) # Threshold.
    robust_lag   = 1 # Threshold.

    candidate_conf = 0.0
    det_total_debug = 0
    fired_indices = []

    abs_gate_det = AbsoluteEnergyDetector(k=abs_gate_k, min_hist=max(10, int(2.0 / detect_hop_s))) if abs_gate else None
    absE_buf_thr = None

    def clamp_time(t):
        return min(t, max_time_s) if max_time_s is not None else t

    def push_output(is_det, conf, metrics, t_sec):
        tt = clamp_time(t_sec)
        results['detections'].append(is_det)
        results['confidence_scores'].append(conf)
        results['detailed_metrics'].append(metrics)
        results['timestamps'].append(tt)

    # circular.
    for m in range(M):
        processor.push_pann_frame(pann_frames[m])
        for s in range(stride_steps):
            out = processor.step()
            energy = float(out['energy'])
            field = out['pressure_field'].reshape(-1)
            state_vec = torch.cat([field])

            pooled_energies.append(energy)
            pooled_state_vecs.append(state_vec)

            is_hop_boundary = ((s + 1) % hop_steps == 0)
            if not is_hop_boundary:
                t_seconds += dt
                continue

            # Implementation note.
            E = float(np.mean(pooled_energies)); pooled_energies.clear()
            S = torch.stack(pooled_state_vecs, dim=0).mean(dim=0); pooled_state_vecs.clear()

            # Implementation note.
            E_for_det = E
            if ema_tau_s > 0:
                ema_energy = E if ema_energy is None else (1 - ema_alpha) * ema_energy + ema_alpha * E
                E_for_det = ema_energy

            if trace_energy and hop_counter < 120:
                print(f"[E-Trace] t={t_seconds:.3f}s E_pool={E:.6g} E_det={E_for_det:.6g}")

            hop_counter += 1

            # Implementation note.
            if hop_counter <= warmup_steps:
                push_output(False, 0.0, {
                    'raw_metrics': {'energy': 0.0, 'pca': 0.0, 'cosine': 0.0},
                    'thresholds': {'energy': None, 'pca': None, 'cosine': None}
                }, t_seconds)
                det_window.clear(); ratio_window.clear(); thr_window.clear()
                candidate_conf = 0.0
                t_seconds += dt
                continue

            # Threshold.
            abs_hit = False
            if abs_gate and abs_gate_det is not None:
                logE = math.log1p(max(E_for_det, 0.0))
                res = abs_gate_det.feed(logE)
                absE_buf_thr = res['thr']
                abs_hit = bool(res['hit'])

            # Implementation note.
            is_change, conf, metrics = detector.update(
                energy=E_for_det, state_vector=S, timestamp=len(results['timestamps'])
            )

            # Threshold.
            val = float(metrics['raw_metrics'].get('energy', 0.0))
            thr = metrics['thresholds'].get('energy', None)
            if thr is not None and np.isfinite(thr) and thr > 0:
                thr_window.append(float(thr))

            # Implementation note.
            inst_ratio = 0.0
            if thr is not None and np.isfinite(thr) and thr > 0:
                inst_ratio = float(val / (thr + 1e-12))
            ratio_window.append(inst_ratio)

            # Implementation note.
            if (hop_counter - last_event_hop) <= cooldown_steps:
                det_window.clear(); ratio_window.clear() # Implementation note.
                candidate_conf = 0.0
                push_output(False, 0.0, metrics, t_seconds)
            else:
                # Threshold.
                det_window.append(bool(is_change))
                candidate_conf = max(candidate_conf, conf)

                # Threshold.
                median_thr = np.median(list(thr_window)[:-1]) if len(thr_window) > robust_lag else None
                robust_ratio = 0.0
                if median_thr is not None and np.isfinite(median_thr) and median_thr > 0:
                    robust_ratio = float(val / (median_thr + 1e-12))

                peak_hit = (robust_ratio >= peak_gate) or (max(ratio_window) >= (peak_gate + 0.5))
                votes = sum(det_window)
                pass_votes = (len(det_window) >= persist_steps and votes >= need_votes)

                if abs_hit or peak_hit or pass_votes:
                    last_event_hop = hop_counter
                    inst_max = max(ratio_window) if len(ratio_window) > 0 else 0.0
                    final_conf = max(candidate_conf, min(1.0, 0.5 + 0.25 * (max(robust_ratio, inst_max) - peak_gate)))
                    push_output(True, final_conf, metrics, t_seconds)
                    det_total_debug += 1
                    fired_indices.append(len(results['detections']) - 1)
                    if debug:
                        mode = "ABS" if abs_hit else ("PEAK" if peak_hit else "VOTE")
                        print(f"[EVENT-{mode}] t={clamp_time(t_seconds):.2f}s conf={final_conf:.3f} "
                              f"inst_max={inst_max:.2f} robust={robust_ratio:.2f} votes={votes}/{persist_steps} "
                              f"abs_thr={absE_buf_thr if absE_buf_thr is not None else -1:.3g}")
                    det_window.clear(); ratio_window.clear()
                    candidate_conf = 0.0
                else:
                    push_output(False, 0.0, metrics, t_seconds)

            t_seconds += dt

    # Implementation note.
    for _ in range(tail_steps):
        out = processor.step()
        energy = float(out['energy'])
        field = out['pressure_field'].reshape(-1)
        state_vec = torch.cat([field])
        pooled_energies.append(energy)
        pooled_state_vecs.append(state_vec)
        is_hop_boundary = (len(pooled_energies) % hop_steps == 0)
        if not is_hop_boundary:
            t_seconds += dt
            continue

        E = float(np.mean(pooled_energies)); pooled_energies.clear()
        S = torch.stack(pooled_state_vecs, dim=0).mean(dim=0); pooled_state_vecs.clear()
        E_for_det = E
        if ema_tau_s > 0:
            ema_energy = E if ema_energy is None else (1 - ema_alpha) * ema_energy + ema_alpha * E
            E_for_det = ema_energy

        hop_counter += 1
        if hop_counter <= warmup_steps:
            push_output(False, 0.0, {
                'raw_metrics': {'energy': 0.0, 'pca': 0.0, 'cosine': 0.0},
                'thresholds': {'energy': None, 'pca': None, 'cosine': None}
            }, t_seconds)
            det_window.clear(); ratio_window.clear(); thr_window.clear()
            candidate_conf = 0.0
            t_seconds += dt
            continue

        abs_hit = False
        if abs_gate and abs_gate_det is not None:
            logE = math.log1p(max(E_for_det, 0.0))
            res = abs_gate_det.feed(logE)
            absE_buf_thr = res['thr']
            abs_hit = bool(res['hit'])

        is_change, conf, metrics = detector.update(
            energy=E_for_det, state_vector=S, timestamp=len(results['timestamps'])
        )

        val = float(metrics['raw_metrics'].get('energy', 0.0))
        thr = metrics['thresholds'].get('energy', None)
        if thr is not None and np.isfinite(thr) and thr > 0:
            thr_window.append(float(thr))
        inst_ratio = 0.0
        if thr is not None and np.isfinite(thr) and thr > 0:
            inst_ratio = float(val / (thr + 1e-12))
        ratio_window.append(inst_ratio)

        if (hop_counter - last_event_hop) <= cooldown_steps:
            det_window.clear(); ratio_window.clear()
            candidate_conf = 0.0
            push_output(False, 0.0, metrics, t_seconds)
        else:
            det_window.append(bool(is_change))
            candidate_conf = max(candidate_conf, conf)
            median_thr = np.median(list(thr_window)[:-1]) if len(thr_window) > robust_lag else None
            robust_ratio = 0.0
            if median_thr is not None and np.isfinite(median_thr) and median_thr > 0:
                robust_ratio = float(val / (median_thr + 1e-12))
            peak_hit = (robust_ratio >= peak_gate) or (max(ratio_window) >= (peak_gate + 0.5))
            votes = sum(det_window)
            pass_votes = (len(det_window) >= persist_steps and votes >= need_votes)
            if abs_hit or peak_hit or pass_votes:
                last_event_hop = hop_counter
                inst_max = max(ratio_window) if len(ratio_window) > 0 else 0.0
                final_conf = max(candidate_conf, min(1.0, 0.5 + 0.25 * (max(robust_ratio, inst_max) - peak_gate)))
                push_output(True, final_conf, metrics, t_seconds)
                det_total_debug += 1
                fired_indices.append(len(results['detections']) - 1)
                if debug:
                    mode = "ABS" if abs_hit else ("PEAK" if peak_hit else "VOTE")
                    print(f"[EVENT-{mode}] t={clamp_time(t_seconds):.2f}s conf={final_conf:.3f} "
                          f"inst_max={inst_max:.2f} robust={robust_ratio:.2f} votes={votes}/{persist_steps} "
                          f"abs_thr={absE_buf_thr if absE_buf_thr is not None else -1:.3g}")
                det_window.clear(); ratio_window.clear()
                candidate_conf = 0.0
            else:
                push_output(False, 0.0, metrics, t_seconds)

        t_seconds += dt

    # Implementation note.
    results['detections_smoothed'] = smooth_detections(results['detections'])
    smoothed = list(results['detections_smoothed'])
    for idx in fired_indices:
        if 0 <= idx < len(smoothed):
            smoothed[idx] = True
    results['detections_smoothed'] = smoothed

    if debug and results['timestamps']:
        print(f"[Sanity] events_printed={det_total_debug}  fired_indices={len(fired_indices)}  "
              f"events_in_array={int(sum(results['detections_smoothed']))}  "
              f"t_first={results['timestamps'][0]:.2f}s  t_last={results['timestamps'][-1]:.2f}s")
    return results

# ===========================
# Threshold.
# ===========================

def create_enhanced_mel_spectrogram_with_detection(audio: np.ndarray, sr: int,
                                                   results: Dict, output_path: str,
                                                   title_suffix: str = "", fmax: int = 8000):
    import librosa.display
    print("Generating enhanced mel spectrogram visualization...")
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=fmax, hop_length=512, n_fft=2048)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    fig, ax = plt.subplots(figsize=(16, 6))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', fmax=fmax, cmap='viridis', alpha=0.85, ax=ax)

    time_axis = np.array(results['timestamps'])

    # Threshold.
    energy_vals, thr_vals, time_vals = [], [], []
    for t, m in zip(time_axis, results['detailed_metrics']):
        e = m['raw_metrics']['energy']
        thr = m['thresholds']['energy']
        if thr is None:
            continue
        energy_vals.append(e); thr_vals.append(thr); time_vals.append(t)

    if len(time_vals) == 0:
        time_vals = [0.0]; energy_vals = [0.0]; thr_vals = [0.0]

    max_freq = 128
    sf = max_freq / (max(energy_vals) if max(energy_vals) > 0 else 1.0)
    ax2 = ax.twinx(); ax2.set_ylim(0, max_freq)
    ax2.plot(time_vals, np.array(energy_vals) * sf, color='cyan', linewidth=1.8, label='Energy metric')
    ax2.plot(time_vals, np.array(thr_vals) * sf, color='red', linestyle='--', linewidth=1.6, label='Adaptive Thr')

    change_idx = [i for i, d in enumerate(results['detections_smoothed']) if d]
    for idx in change_idx:
        ax.axvline(x=results['timestamps'][idx], color='red', linewidth=2.0, alpha=0.9)

    total_changes = len(change_idx)
    title = f'BioOSS Detection — {title_suffix} (changes={total_changes})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Mel bins')
    ax2.set_ylabel('Scaled Energy', color='cyan')
    ax.grid(True, alpha=0.25)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Enhanced spectrogram saved to: {output_path}")

def create_multi_metric_visualization(results: Dict, output_path: str):
    print("Generating multi-metric visualization...")
    time_axis = np.array(results['timestamps'])
    if len(time_axis) == 0:
        print("No data to plot."); return

    energy_values, energy_thresholds, time_vals = [], [], []
    for t, m in zip(time_axis, results['detailed_metrics']):
        thr = m['thresholds']['energy']
        if thr is None:
            continue
        energy_values.append(m['raw_metrics']['energy'])
        energy_thresholds.append(thr)
        time_vals.append(t)

    if not time_vals:
        print("No valid (non-warmup) points to plot."); return

    confidence_scores = results['confidence_scores']
    change_indices = [i for i, d in enumerate(results['detections_smoothed']) if d]

    fig, axes = plt.subplots(3, 1, figsize=(16, 9))
    axes[0].plot(time_vals, energy_values, 'b-', linewidth=1.6, label='Energy metric')
    axes[0].plot(time_vals, energy_thresholds, 'r--', linewidth=1.4, label='Adaptive Thr')
    for idx in change_indices:
        axes[0].axvline(x=results['timestamps'][idx], color='red', linewidth=1.6, alpha=0.8)
    axes[0].set_title('Energy & Threshold'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, confidence_scores, color='orange', linewidth=1.8, label='Confidence')
    axes[1].grid(True, alpha=0.3); axes[1].legend(); axes[1].set_title('Confidence')

    det_bin = [1 if d else 0 for d in results['detections_smoothed']]
    axes[2].fill_between(time_axis, 0, det_bin, color='red', alpha=0.5, step='mid', label='Detections')
    axes[2].set_ylim(0, 1.1); axes[2].grid(True, alpha=0.3); axes[2].legend(); axes[2].set_title('Detections')
    axes[2].set_xlabel('Time (s)')
    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Multi-metric visualization saved to: {output_path}")

def save_results_summary(results: Dict, output_path: str, grid_shape: Tuple[int, int], freq_range: Tuple[float, float]):
    change_indices = [i for i, d in enumerate(results['detections_smoothed']) if d]
    change_times = [results['timestamps'][i] for i in change_indices]
    with open(output_path, 'w') as f:
        f.write("BioOSS Audio Change Detection (Region-based, Causal OLA)\n")
        f.write("=" * 60 + "\n\n")
        f.write("System Configuration:\n")
        f.write(f"  Grid: {grid_shape[0]}x{grid_shape[1]} regions\n")
        f.write(f"  Frequency range: {freq_range[0]}–{freq_range[1]} Hz\n")
        f.write("\nDetection Results:\n")
        f.write(f"  Total steps (hops): {len(results['detections'])}\n")
        f.write(f"  Total changes detected: {len(change_indices)}\n")
        for i, t in enumerate(change_times):
            f.write(f"  {i+1:2d}. t = {t:.2f}s\n")
    print(f"Results summary saved to: {output_path}")

# ===========================
# Implementation note.
# ===========================

def main():
    args = parse_arguments()
    args = apply_profile(args) # Implementation note.
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load audio & windows
    audio, sr = load_and_preprocess_audio(args.audio_path, args.sample_rate)
    segments = create_sliding_windows(audio, sr, args.window_length, args.stride_length)

    # Load PANN
    print("Loading PANN model...")
    pann_model = load_pretrained_pann(
        checkpoint_path=args.pann_checkpoint, device=device,
        sample_rate=args.sample_rate, window_size=1024, hop_size=320,
        mel_bins=64, fmin=50, fmax=14000, classes_num=527
    )
    pann_frames = extract_pann_features(segments, pann_model, device)  # (M,527)

    # Build BioOSS pipeline
    proc = build_biooss_pipeline(
        grid_h=64, grid_w=64,
        dt=args.dt,
        freq_range=(args.freq_min, args.freq_max),
        region_rows=args.region_rows, region_cols=args.region_cols,
        kp=args.kp, ko=args.ko,
        pann_dim=527
    )
    proc.mod.alpha = args.mod_alpha
    print(f"[Pipeline] kp={args.kp}, ko={args.ko}, mod.alpha={args.mod_alpha}")
    print("Running BioOSS stream (dt resolution with hop gating + majority persistence)...")

    # Detector
    detector = SimpleEnergyChangeDetector(k=args.simple_k, min_hist=20, debug_print=False)

    # Implementation note.
    results = run_biooss_stream(
        pann_frames=pann_frames,
        processor=proc,
        detector=detector,
        dt=args.dt,
        stride_s=args.stride_length,
        tail_flush_s=args.tail_flush_s,
        max_time_s=len(audio)/sr,
        debug=True,
        detect_hop_s=args.detect_hop_s,
        min_persist_s=args.min_persist_s,
        cooldown_s=args.cooldown_s,
        warmup_s=args.warmup_s,
        ema_tau_s=args.ema_tau_s,
        trace_energy=args.trace_energy,
        persist_frac=args.persist_frac,
        peak_gate=args.peak_gate,
        abs_gate=args.abs_gate,
        abs_gate_k=args.abs_gate_k
    )

    print(f"Raw detections: {sum(results['detections_smoothed'])}")

    # Implementation note.
    merged_idx = merge_events(results['timestamps'], results['detections_smoothed'],
                              results['confidence_scores'], args.nms_sep_s)
    if merged_idx:
        print(f"\nDetected changes (merged {len(merged_idx)}):")
        for i, idx in enumerate(merged_idx):
            print(f"  {i+1:2d}. t = {results['timestamps'][idx]:6.2f}s  (conf={results['confidence_scores'][idx]:.3f})")
    else:
        print("No significant changes detected (after NMS).")

    # Threshold.
    E, T = [], []
    for m in results['detailed_metrics']:
        if m['thresholds']['energy'] is None:
            continue
        E.append(m['raw_metrics']['energy'])
        T.append(m['thresholds']['energy'])
    if E and T and np.isfinite(np.mean(T)):
        print(f"[Diag] metric mean/std/max = {np.mean(E):.4g}/{np.std(E):.4g}/{np.max(E):.4g}")
        print(f"[Diag] thr    mean/std/min = {np.mean(T):.4g}/{np.std(T):.4g}/{np.min(T):.4g}")
        print(f"[Diag] max(metric)/mean(thr) = { (np.max(E) / (np.mean(T)+1e-8)) :.3f}")

    # Save / Viz
    audio_name = os.path.basename(args.audio_path).replace('.wav', '')
    if args.plot_spectrogram:
        spec_path = os.path.join(args.output_dir, 'enhanced_mel_spectrogram_detection.png')
        create_enhanced_mel_spectrogram_with_detection(audio, sr, results, spec_path, title_suffix=audio_name)
    if args.plot_metrics:
        metrics_path = os.path.join(args.output_dir, 'comprehensive_metrics_analysis.png')
        create_multi_metric_visualization(results, metrics_path)

    summary_path = os.path.join(args.output_dir, 'detection_summary.txt')
    save_results_summary(results, summary_path,
                         grid_shape=(args.region_rows, args.region_cols),
                         freq_range=(args.freq_min, args.freq_max))

    # Save.
    np.savez(
        os.path.join(args.output_dir, 'raw_results_region_pipeline.npz'),
        detections=np.array(results['detections']),
        detections_smoothed=np.array(results['detections_smoothed']),
        confidence_scores=np.array(results['confidence_scores']),
        timestamps=np.array(results['timestamps']),
        pann_features=pann_frames.cpu().numpy()
    )

    print("\nSystem Summary:")
    print(f"  Regions: {args.region_rows} x {args.region_cols}")
    print(f"  Frequency pool: {args.freq_min}-{args.freq_max} Hz")
    print(f"  Time step dt: {args.dt}s")
    print(f"  PANN mapping: 527 -> {args.region_rows*args.region_cols} region amplitudes (causal OLA upsample)")
    print(f"  Input: u_r[n] = a_r[n] * sin(2π f_r n dt) (per region)")
    print("Done.")

if __name__ == "__main__":
    main()