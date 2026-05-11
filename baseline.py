import os
import time
import gc
import torch
import traceback
import pandas as pd
import re
import sys

MEMORY_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Memory_model")
if MEMORY_MODEL_DIR not in sys.path:
    sys.path.insert(0, MEMORY_MODEL_DIR)
from Qwen_processor import QwenAudioDescription
from queue import Queue


def extract_file_index(file_path):
    """Extract the R00XX file index from file path"""
    match = re.search(r'R\d{4}', file_path)
    if match:
        return match.group(0)
    return "Unknown"


def extract_classification(description):
    """Extract classification (calming/tranquil, lively/active, or neither) from description"""
    lower_desc = description.lower()

    # Pattern 1: Match the exact format in the example with two styles of question numbering
    q3_pattern = re.search(
        r'(?:3\.|3\)|3:)\s+In general,\s+how would you categorize the environment you just experienced\?.*?\n(.*?)(?:\n|$)',
        description, re.IGNORECASE | re.DOTALL)
    if q3_pattern:
        answer = q3_pattern.group(1).strip().lower()
        # Check for exact classifications in the direct answer
        if "lively/active" in answer:
            return "lively/active"
        elif "calming/tranquil" in answer:
            return "calming/tranquil"
        elif "neither" in answer:
            return "neither"

    # Pattern 2: Try to extract using the exact format layout
    format_pattern = re.search(
        r'\d+\.\s+What sounds do you hear\?.*?\n.*?\n\s*\d+\.\s+What type of public place.*?\n.*?\n\s*\d+\.\s+In general,.*?\n(.*?)(?:\n|$)',
        description, re.IGNORECASE | re.DOTALL)
    if format_pattern:
        answer = format_pattern.group(1).strip().lower()
        # Check for exact classifications in the direct answer
        if "lively/active" in answer:
            return "lively/active"
        elif "calming/tranquil" in answer:
            return "calming/tranquil"
        elif "neither" in answer:
            return "neither"

    # Try a more general pattern for question 3
    q3_alt_pattern = re.search(r'(?:3\.|\bthird|\bthree\b).*categorize.*environment.*?\n(.*?)(?:\n|$)', description,
                               re.IGNORECASE | re.DOTALL)
    if q3_alt_pattern:
        answer = q3_alt_pattern.group(1).strip().lower()
        # Check for exact classifications in the direct answer
        if "lively/active" in answer or "lively" in answer:
            return "lively/active"
        elif "calming/tranquil" in answer or "tranquil" in answer or "calming" in answer:
            return "calming/tranquil"
        elif "neither" in answer:
            return "neither"

    # Try another pattern for differently formatted descriptions
    q3_alt_pattern = re.search(
        r'(?:categorize|classify).*environment.*?\s+(lively/active|calming/tranquil|neither)\.?', description,
        re.IGNORECASE)
    if q3_alt_pattern:
        return q3_alt_pattern.group(1).lower()

    # If we can't find the question format, fall back to the keyword approach
    # Define expanded keywords for classification
    lively_keywords = ["lively/active", "lively", "active", "car", "busy", "traffic",
                       "noisy", "loud", "bustling", "crowded", "vehicle", "engine", "train", "airport"]

    tranquil_keywords = ["calming/tranquil", "tranquil", "calming", "quiet", "peaceful",
                         "serene", "gentle", "relaxing"]

    # First check for explicit 'neither' classification
    if re.search(r'\bneither\b', lower_desc):
        return "neither"

    # Count occurrences of keywords from each category
    lively_count = 0
    tranquil_count = 0

    for keyword in lively_keywords:
        if re.search(rf'\b{keyword}\b', lower_desc):
            lively_count += 1

    for keyword in tranquil_keywords:
        if re.search(rf'\b{keyword}\b', lower_desc):
            tranquil_count += 1

    # Determine classification based on keyword frequency
    if lively_count > tranquil_count:
        return "lively/active"
    elif tranquil_count > lively_count:
        return "calming/tranquil"
    elif lively_count > 0 and tranquil_count > 0:
        # If both types of keywords are present but equal count, look for explicit classification
        match = re.search(r'(calming/tranquil|lively/active|neither)', lower_desc)
        if match:
            return match.group(1)
        else:
            return "neither"  # If ambiguous, default to neither
    else:
        # Try to find an explicit answer (backup)
        match = re.search(r'(calming/tranquil|lively/active|neither)', lower_desc)
        if match:
            return match.group(1)

        # Default case if we couldn't extract
        return "unclassified"


def process_audio_baseline(audio_file_path):
    """Process a single audio file, return description and classification"""
    # Create fresh queue for QwenAudioDescription
    recall_queue = Queue()
    description_queue = Queue()

    # Initialize Qwen model
    qwen_description = QwenAudioDescription(recall_queue, description_queue, device='cuda:0')

    try:
        # Generate description for the entire audio file
        description = qwen_description.generate_description(audio_file_path)

        # Extract classification from the description
        classification = extract_classification(description)

        return {
            'description': description,
            'classification': classification
        }
    except Exception as e:
        print(f"Error processing {audio_file_path}: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        # Clean up resources
        try:
            qwen_description.stop()
        except:
            pass

        # Force Python garbage collection
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    # Define paths
    dataset_path = "/home/zhyuan/Desktop/Qwen-Audio/ambisonics_audio"
    results_path = "/home/zhyuan/Desktop/Qwen-Audio/results/TXT"
    excel_results_path = "/home/zhyuan/Desktop/Qwen-Audio/results/excel"

    # Create results directories if they don't exist
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(excel_results_path, exist_ok=True)

    # List to collect all file results for Excel
    all_file_data = []

    # Process each audio file in the dataset
    audio_files = [f for f in os.listdir(dataset_path) if f.endswith(".wav")]
    print(f"Found {len(audio_files)} audio files to process.")

    for i, file_name in enumerate(audio_files):
        file_path = os.path.join(dataset_path, file_name)
        file_id = extract_file_index(file_path)

        print(f"[{i + 1}/{len(audio_files)}] Processing {file_name} (ID: {file_id})...")

        # Define output paths
        txt_file_name = file_name.replace(".wav", ".txt")
        txt_file_path = os.path.join(results_path, txt_file_name)

        # Process the audio file
        result = process_audio_baseline(file_path)

        if result:
            # Write to text file
            with open(txt_file_path, 'w') as txt_file:
                txt_file.write(f"File: {file_id} ({file_name})\n")
                txt_file.write(f"Classification: {result['classification']}\n")
                txt_file.write(f"Description: {result['description']}\n")

            # Add to Excel data collection
            all_file_data.append({
                'File_Index': file_id,
                'Classification': result['classification'],
                'Description': result['description'],
                'File_Name': file_name
            })

            print(f"  Classification: {result['classification']}")
            print(f"  Results saved to {txt_file_path}")
        else:
            print(f"  Failed to process {file_name}")

        # Give the system a short break between files
        time.sleep(1)

    # Create combined Excel file with all results
    if all_file_data:
        print(f"Creating combined Excel file with {len(all_file_data)} entries...")
        try:
            df = pd.DataFrame(all_file_data)
            excel_path = os.path.join(excel_results_path, "baseline_audio_classifications_1to133.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"All results written to {excel_path}")

            # Also save individual Excel files for each audio file
            for data in all_file_data:
                file_id = data['File_Index']
                single_df = pd.DataFrame([data])  # Convert single entry to DataFrame
                single_excel_path = os.path.join(excel_results_path, f"{file_id}_classification.xlsx")
                single_df.to_excel(single_excel_path, index=False)
        except Exception as e:
            print(f"Error creating Excel file: {str(e)}")
            traceback.print_exc()
    else:
        print("No valid data collected, skipping Excel file creation")


if __name__ == "__main__":
    main()
