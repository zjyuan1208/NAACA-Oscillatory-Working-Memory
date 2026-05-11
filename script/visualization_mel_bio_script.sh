#!/bin/bash

# BioOSS Audio Change Detection - Batch Processing Script

# Define sample names
SAMPLES=('R0002' 'R0003' 'R0007' 'R0010' 'R0016' 'R0028' 'R0030' 'R0031' 'R0037' 'R0056' 'R0078' 'R0130' 'R0131')

# Configuration paths
PYTHON_EXEC="/home/zhyuan/anaconda3/envs/QWEN/bin/python3.9"
PYTHON_SCRIPT="/home/zhyuan/Desktop/PCD/run.py"
AUDIO_DIR="/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio"
PANN_CHECKPOINT="/home/zhyuan/Desktop/audioset_tagging_cnn/Cnn14_mAP=0.431.pth"
OUTPUT_DIR="/home/zhyuan/Desktop/Qwen-Audio/Memory_model/plot/figures"

# Script parameters
WINDOW_LENGTH=4.0
STRIDE_LENGTH=1.0
SAMPLE_RATE=32000
FREQ_MIN=50.0
FREQ_MAX=1200.0
GRID_SIZE=64
ENERGY_WEIGHT=0.4
PCA_WEIGHT=0.0
COSINE_WEIGHT=0.2
CONSENSUS_THRESHOLD=0.35
MIN_PERSISTENCE=2

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each sample
for sample in "${SAMPLES[@]}"; do
    echo "Processing sample: $sample"

    # Construct audio file path
    audio_file="$AUDIO_DIR/${sample}_segment_ambisonics.wav"

    # Run the Python script
    "$PYTHON_EXEC" -u "$PYTHON_SCRIPT" \
        --audio_path "$audio_file" \
        --pann_checkpoint "$PANN_CHECKPOINT" \
        --output_dir "$OUTPUT_DIR" \
        --window_length "$WINDOW_LENGTH" \
        --stride_length "$STRIDE_LENGTH" \
        --sample_rate "$SAMPLE_RATE" \
        --freq_min "$FREQ_MIN" \
        --freq_max "$FREQ_MAX" \
        --grid_size "$GRID_SIZE" \
        --energy_weight "$ENERGY_WEIGHT" \
        --pca_weight "$PCA_WEIGHT" \
        --cosine_weight "$COSINE_WEIGHT" \
        --consensus_threshold "$CONSENSUS_THRESHOLD" \
        --min_persistence "$MIN_PERSISTENCE" \
        --plot_spectrogram true \
        --plot_metrics false \
        --plot_biooss true \
        --device cpu

    echo ""
done

echo "Batch processing complete."