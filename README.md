# NAACA: Training-Free NeuroAuditory Attentive Cognitive Architecture

Official code repository for the ICML 2026 paper:

**NAACA: Training-Free NeuroAuditory Attentive Cognitive Architecture with Oscillatory Working Memory for Salience-Driven Attention Gating**

NAACA is a training-free audio attention-gating pipeline for long-form audio understanding. It uses an Oscillatory Working Memory (OWM) module to detect salient acoustic transitions and selectively route those segments to an audio language model, reducing unnecessary inference while improving recall of rare or late events.

## Highlights

- **Training-free salience gating**: no task-specific training is required for the OWM detector.
- **Oscillatory Working Memory**: a 2D damped-wave recurrent field driven by pretrained audio object probabilities.
- **Audio language model routing**: salient windows are forwarded to Qwen-Audio for higher-level semantic interpretation.
- **XD-Violence and USoW workflows**: scripts for salience-driven processing, random baselines, and qualitative urban soundscape analysis.

## Repository Structure

```text
Memory_model/
  biodetector.py              # OWM/BioOSS FDTD dynamics and multi-metric salience detector
  XD_mainstream_biooss.py     # XD-Violence NAACA pipeline
  XD_mainstream_random.py     # Random-gating baseline for XD-Violence
  mainstream_bio.py           # USoW / general long-audio NAACA pipeline
  mainstream_clipwise.py      # PANN clipwise baseline
  Qwen_processor.py           # Qwen-Audio wrapper for segment descriptions
  PANN_backbone.py            # PANNs Cnn14 audio encoder backbone
results/
  biooss_multimetric_pattern_change_results.json
  biooss_multimetric_pattern_change_results_realtime.json
  pattern_change_results.json
```

Large audio files, generated chunks, checkpoints, caches, and the local review PDF are intentionally excluded from Git.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
pip install librosa soundfile scikit-learn pandas openpyxl tqdm
```

You also need:

- A pretrained PANNs Cnn14 checkpoint, passed with `--checkpoint_path`.
- Qwen-Audio model weights if you use the higher-cognition description module.
- Dataset paths for XD-Violence or USoW-style audio, passed with `--dataset_path`.

## Example Usage

Run NAACA on XD-Violence-style audio:

```bash
python Memory_model/XD_mainstream_biooss.py \
  --device cuda:0 \
  --dataset_path /path/to/xd_violence/audios \
  --checkpoint_path /path/to/Cnn14_mAP=0.431.pth \
  --results_dir results
```

Run the random-gating baseline with the same number of selected windows:

```bash
python Memory_model/XD_mainstream_random.py \
  --device cuda:0 \
  --dataset_path /path/to/xd_violence/audios \
  --biooss_json results/dataset_xd_cls.json \
  --results_dir results
```

Run the USoW/general long-audio NAACA pipeline:

```bash
python Memory_model/mainstream_bio.py \
  --device cuda:0 \
  --checkpoint_path /path/to/Cnn14_mAP=0.431.pth
```

## Core Method

The detector maps PANN clipwise probability vectors into frequency-specific oscillatory drives on a 2D OWM lattice. The recurrent field evolves pressure and velocity states with damped wave dynamics. Salience is detected from adaptive energy and state-change metrics, then persistent detections open the attention gate for higher-level audio-language reasoning.

The main implementation lives in `Memory_model/biodetector.py`:

- `BioOSSFDTD2D`: OWM recurrent field and FDTD update.
- `OnlineMultiMetricChangeDetector`: adaptive salience decision logic.
- `BioOSSPatternChangeDetector`: end-to-end audio feature extraction plus OWM gating.

## Citation

```bibtex
@inproceedings{naaca2026,
  title={NAACA: Training-Free NeuroAuditory Attentive Cognitive Architecture with Oscillatory Working Memory for Salience-Driven Attention Gating},
  author={Anonymous},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## Acknowledgements

This repository builds on Qwen-Audio components for audio-language reasoning and PANNs-style audio tagging features for the encoder-driven OWM input.

## License

The NAACA repository is released under the MIT License. Qwen-Audio-derived components retain their original Tongyi Qianwen license; see `LICENSE_QWEN` for those terms.

## NAACA Audio Attention Code

This repository also includes the audio-attention and BioOSS working-memory code used for the NAACA audio experiments. The added components include:

- `biooss_fdtd.py` and `biooss_pipeline.py`: BioOSS/FDTD dynamics and online audio-attention processing.
- `run.py` and `run_redesigned.py`: audio change-detection entry points.
- `audio_attention_utils.py`: online thresholding, detection post-processing, and metric helpers for the audio-attention pipeline.
- `baselines/`: DriftLens, MCD-DD, PUDD, adaptive classical, and K-Means baselines.
- `datasets/`: LU-AVS/XD-style dataset preparation and dataloading utilities.
- `scripts_LU/`, `scripts_baseline/`, and `script/`: reproducibility scripts.
- `plot/`: paper plotting scripts and lightweight key result artifacts.

Large raw audio/video datasets, internal-state arrays, checkpoints, and generated visualization folders are excluded from Git.

