#!/usr/bin/env python3
"""
Qwen Audio Processing Script
Processes all audio samples in the dataset using Qwen-Audio-Chat and saves responses to JSON.
"""

import os
import json
import glob
from datetime import datetime
import traceback
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
DATASET_PATH = "/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio"
RESULTS_PATH = "/home/zhyuan/Desktop/audioset_tagging_cnn/results"
MODEL_ID = 'qwen/Qwen-Audio-Chat'
REVISION = 'master'

# Create results directory if it doesn't exist
os.makedirs(RESULTS_PATH, exist_ok=True)

# Initialize logging
summary_file = os.path.join(RESULTS_PATH, "qwen_processing_summary.txt")


def log_message(message, print_to_console=True):
    """Log message to both console and summary file"""
    if print_to_console:
        print(message)
    with open(summary_file, 'a') as f:
        f.write(f"{message}\n")


def initialize_model():
    """Initialize Qwen model and tokenizer"""
    log_message("Initializing Qwen-Audio-Chat model...")

    try:
        # Download model checkpoint to local dir
        model_dir = snapshot_download(MODEL_ID, revision=REVISION)

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="cuda",
            trust_remote_code=True
        ).eval()

        log_message("✓ Model initialized successfully")
        return model, tokenizer

    except Exception as e:
        log_message(f"✗ Failed to initialize model: {str(e)}")
        raise


def is_ai_response(response):
    """Check if the response contains AI model disclaimer"""
    ai_keywords = [
        "I am an AI",
        "I'm an AI",
        "As an AI",
        "I am an artificial intelligence",
        "I'm an artificial intelligence",
        "I cannot actually hear",
        "I don't have the ability to hear",
        "I cannot experience sounds"
    ]

    response_lower = response.lower()
    return any(keyword.lower() in response_lower for keyword in ai_keywords)


def process_audio_file(model, tokenizer, audio_path, max_retries=3):
    """Process a single audio file with Qwen, with retry mechanism for AI responses"""
    filename = os.path.splitext(os.path.basename(audio_path))[0]

    for attempt in range(max_retries):
        try:
            # Prepare the query
            query = tokenizer.from_list_format([
                {'audio': audio_path},
                {
                    'text': """Assume that you are a human in a public place, answer the following question: Can you describe what sounds do you hear?"""}
            ])

            # Get response from model
            response, history = model.chat(tokenizer, query=query, history=None)

            # Check if response contains AI disclaimer
            if is_ai_response(response):
                log_message(f"  ⚠ Attempt {attempt + 1}: AI disclaimer detected, retrying...")
                if attempt < max_retries - 1:
                    continue
                else:
                    log_message(f"  ✗ All {max_retries} attempts returned AI disclaimer")
                    return {
                        "filename": filename,
                        "audio_path": audio_path,
                        "response": response,
                        "status": "ai_disclaimer",
                        "attempts": max_retries,
                        "note": "All attempts returned AI model disclaimer"
                    }

            # Success - got a valid human-like response
            if attempt > 0:
                log_message(f"  ✓ Success on attempt {attempt + 1}")

            return {
                "filename": filename,
                "audio_path": audio_path,
                "response": response,
                "status": "success",
                "attempts": attempt + 1
            }

        except Exception as e:
            error_msg = f"Error processing {filename} (attempt {attempt + 1}): {str(e)}"
            log_message(f"  ✗ {error_msg}")

            if attempt == max_retries - 1:
                return {
                    "filename": filename,
                    "audio_path": audio_path,
                    "response": "",
                    "status": "failed",
                    "error": str(e),
                    "attempts": max_retries
                }


def main():
    """Main processing function"""
    # Initialize summary file
    with open(summary_file, 'w') as f:
        f.write(f"Qwen Audio Processing Summary - {datetime.now()}\n")
        f.write(f"Dataset: {DATASET_PATH}\n")
        f.write(f"Results: {RESULTS_PATH}\n")
        f.write("=" * 50 + "\n")

    # Initialize counters
    total_files = 0
    processed_files = 0
    failed_files = 0
    ai_disclaimer_files = 0

    # Initialize model
    try:
        model, tokenizer = initialize_model()
    except Exception as e:
        log_message(f"Failed to initialize model. Exiting.")
        return

    # Find all wav files
    audio_pattern = os.path.join(DATASET_PATH, "*.wav")
    audio_files = glob.glob(audio_pattern)

    if not audio_files:
        log_message(f"No .wav files found in {DATASET_PATH}")
        return

    total_files = len(audio_files)
    log_message(f"Found {total_files} audio files to process")
    log_message("=" * 50)

    # Initialize results list
    all_results = []

    # Process each audio file
    for i, audio_file in enumerate(audio_files, 1):
        log_message(f"Processing ({i}/{total_files}): {os.path.basename(audio_file)}")

        result = process_audio_file(model, tokenizer, audio_file)
        all_results.append(result)

        if result["status"] == "success":
            processed_files += 1
            log_message(f"✓ Successfully processed: {result['filename']}")
        elif result["status"] == "ai_disclaimer":
            ai_disclaimer_files += 1
            log_message(f"⚠ AI disclaimer response: {result['filename']}")
        else:
            failed_files += 1
            log_message(f"✗ Failed to process: {result['filename']}")

        # Progress indicator
        log_message(f"Progress: {i}/{total_files} files processed")
        log_message("-" * 30)

    # Create final JSON output
    output_data = {
        "dataset_info": {
            "dataset_path": DATASET_PATH,
            "total_files": total_files,
            "processed_files": processed_files,
            "failed_files": failed_files,
            "ai_disclaimer_files": ai_disclaimer_files,
            "processing_date": datetime.now().isoformat(),
            "model_info": {
                "model_id": MODEL_ID,
                "revision": REVISION
            }
        },
        "results": all_results
    }

    # Save results to JSON file
    output_file = os.path.join(RESULTS_PATH, "qwen_results.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        log_message("=" * 50)
        log_message("PROCESSING COMPLETE!")
        log_message("=" * 50)
        log_message(f"Total files found: {total_files}")
        log_message(f"Successfully processed: {processed_files}")
        log_message(f"AI disclaimer responses: {ai_disclaimer_files}")
        log_message(f"Failed: {failed_files}")
        log_message(f"Results saved to: {output_file}")
        log_message(f"Summary saved to: {summary_file}")

    except Exception as e:
        log_message(f"✗ Failed to save results: {str(e)}")


if __name__ == "__main__":
    main()