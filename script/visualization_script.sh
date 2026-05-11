#!/bin/bash

# Internal State Visualization - Batch Processing Script

# Define sample names
SAMPLES=('R0002' 'R0003' 'R0007' 'R0010' 'R0016' 'R0028' 'R0030' 'R0031' 'R0037' 'R0056' 'R0078' 'R0130' 'R0131')

# Configuration paths
PYTHON_EXEC="/home/zhyuan/anaconda3/envs/QWEN/bin/python3.9"
PYTHON_SCRIPT="/home/zhyuan/Desktop/PCD/vis_internal_state.py"

# Process each sample with both modes
for sample in "${SAMPLES[@]}"; do
    echo "Processing sample: $sample"

    # First run: both mode (process and visualize)
    echo "Running both mode (process + visualize)..."
    "$PYTHON_EXEC" -u "$PYTHON_SCRIPT" --mode both --sample_index "$sample"

    # Second run: fft mode
    echo "Running FFT analysis..."
    "$PYTHON_EXEC" -u "$PYTHON_SCRIPT" --mode fft --sample_index "$sample"

    echo ""
done

echo "Batch processing complete."