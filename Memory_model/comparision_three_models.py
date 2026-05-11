#!/usr/bin/env python3
"""
JSON to Excel Comparison Script
Loads PANN, Qwen, and Cognitive Framework results and creates comparison Excel file.
"""

import json
import re
import pandas as pd
import os
from datetime import datetime

# File paths
PANN_PATH = "/home/zhyuan/Desktop/audioset_tagging_cnn/results/pann_results.json"
QWEN_PATH = "/home/zhyuan/Desktop/audioset_tagging_cnn/results/qwen_results.json"
COGNITIVE_PATH = "/home/zhyuan/Desktop/Qwen-Audio/results/pattern_change_results.json"
OUTPUT_PATH = "/home/zhyuan/Desktop/Qwen-Audio/results/comparison_results.xlsx"


def load_json_file(file_path):
    """Load JSON file and return data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_file_index(filename):
    """Extract R00XX index from filename"""
    match = re.search(r'R\d{4}', filename)
    if match:
        return match.group(0)
    return filename


def extract_sound_events_from_text(text):
    """Extract sound event names from natural language description"""
    if not text or not isinstance(text, str):
        return []

    # Convert to lowercase for matching
    text_lower = text.lower()

    # Comprehensive list of sound event keywords to detect
    sound_keywords = {
        # Human sounds
        'speech': ['speech', 'talking', 'conversation', 'voices', 'people talking', 'discussion', 'man speaking',
                   'woman speaking', 'voice', 'narrating', 'announcement'],
        'laughter': ['laughing', 'laughter', 'laugh'],
        'crying': ['crying', 'cry', 'sobbing', 'baby crying'],
        'shouting': ['shouting', 'yelling'],
        'screaming': ['screaming', 'children screaming'],
        'footsteps': ['footsteps', 'walking', 'steps'],
        'children': ['children playing', 'kids playing', 'children', 'kids'],
        'cheering': ['cheering', 'crowd cheering'],
        'clapping': ['clapping', 'applause'],

        # Transportation
        'traffic': ['traffic', 'cars', 'vehicles', 'automobile', 'cars passing by'],
        'car': ['car engine', 'car', 'vehicle engine', 'car door', 'door slamming'],
        'truck': ['truck', 'heavy vehicle'],
        'motorcycle': ['motorcycle', 'motorbike', 'motorcycle engine'],
        'train': ['train', 'railway', 'locomotive', 'train station'],
        'train_whistle': ['train whistle', 'train horn'],
        'airplane': ['airplane', 'aircraft', 'plane'],
        'helicopter': ['helicopter'],
        'bus': ['bus'],
        'ambulance': ['ambulance', 'siren', 'emergency siren'],
        'bicycle': ['bicycle', 'bike'],
        'bicycle_bell': ['bicycle bell'],
        'boat': ['boat engine'],
        'horn': ['horn honking', 'car horn', 'honking'],

        # Nature sounds
        'birds': ['birds', 'bird', 'chirping', 'bird calls', 'bird song'],
        'wind': ['wind', 'breeze', 'windy', 'wind blowing'],
        'rain': ['rain', 'rainfall', 'precipitation', 'drizzle', 'rain falling'],
        'thunder': ['thunder', 'thunderstorm'],
        'water': ['water', 'river', 'stream', 'flowing water', 'fountain', 'water flowing'],
        'ocean': ['ocean', 'sea', 'surf'],
        'waves': ['waves', 'waves crashing'],
        'insects': ['insects', 'buzzing', 'crickets'],
        'leaves': ['leaves rustling', 'rustling'],

        # Music and instruments
        'music': ['music', 'musical', 'music playing'],
        'piano': ['piano'],
        'guitar': ['guitar'],
        'drums': ['drums', 'drumming'],
        'bagpipe': ['bagpipe', 'bagpipes'],
        'violin': ['violin'],
        'trumpet': ['trumpet'],
        'singing': ['singing', 'song'],

        # Mechanical/Urban sounds
        'construction': ['construction', 'drilling', 'hammering', 'building', 'construction equipment'],
        'machinery': ['machinery', 'machine', 'mechanical', 'lawnmower'],
        'alarm': ['alarm', 'alert', 'warning'],
        'bell': ['bell', 'ringing', 'chime', 'church bells', 'church bell'],
        'telephone': ['telephone', 'phone', 'ring'],
        'door': ['door', 'opening', 'closing', 'slam'],
        'engine': ['engine', 'engine idling'],
        'whistle': ['whistle', 'whistle blowing'],
        'metal': ['metal', 'metal clanging'],
        'camera': ['camera', 'camera shutter', 'shutter clicking'],
        'ventilation': ['ventilation', 'ventilation system'],
        'ice_cream_truck': ['ice cream truck'],

        # Sports and activities
        'basketball': ['basketball', 'basketball bouncing', 'ball bouncing'],
        'sports': ['sports', 'game'],

        # Animals
        'dog': ['dog', 'barking', 'bark'],
        'cat': ['cat', 'meowing', 'purr'],
        'animals': ['animals', 'animal sounds'],

        # Environment
        'crowd': ['crowd', 'crowded', 'busy'],
        'quiet': ['quiet', 'peaceful', 'calm', 'silent'],
        'indoor': ['indoor', 'inside'],
        'outdoor': ['outdoor', 'outside'],
        'urban': ['urban', 'city'],
        'park': ['park'],
        'restaurant': ['restaurant', 'cafe'],
        'street': ['street'],

        # Activities
        'cooking': ['cooking', 'kitchen'],
        'typing': ['typing', 'keyboard'],
        'cleaning': ['cleaning', 'vacuum']
    }

    detected_events = set()

    # Search for each sound category
    for category, keywords in sound_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected_events.add(category)
                break  # Found one keyword for this category, move to next category

    return sorted(list(detected_events))


def process_pann_data(pann_data):
    """Process PANN results and return dictionary indexed by file ID"""
    pann_results = {}

    if 'results' in pann_data:
        for result in pann_data['results']:
            file_id = extract_file_index(result.get('filename', ''))
            detected_events = result.get('detected_events', {})

            # Format detected events as "event_name: probability"
            event_list = []
            for event, prob in detected_events.items():
                event_list.append(f"{event}: {prob}")

            pann_results[file_id] = '; '.join(event_list) if event_list else ''

    return pann_results


def process_qwen_data(qwen_data):
    """Process Qwen results and return dictionary indexed by file ID"""
    qwen_results = {}

    if 'results' in qwen_data:
        for result in qwen_data['results']:
            file_id = extract_file_index(result.get('filename', ''))
            response = result.get('response', '')

            # Extract sound events from response
            events = extract_sound_events_from_text(response)
            qwen_results[file_id] = '; '.join(events) if events else ''

    return qwen_results


def process_cognitive_data(cognitive_data):
    """Process Cognitive Framework results and return dictionary indexed by file ID"""
    cognitive_results = {}

    if 'results' in cognitive_data:
        for result in cognitive_data['results']:
            file_id = result.get('file_id', '')

            all_events = set()

            # Extract from initial segment
            initial_desc = result.get('initial_segment', {}).get('description', '')
            if initial_desc:
                events = extract_sound_events_from_text(initial_desc)
                all_events.update(events)

            # Extract from pattern changes
            pattern_changes = result.get('pattern_changes', [])
            for change in pattern_changes:
                desc = change.get('description', '')
                if desc:
                    events = extract_sound_events_from_text(desc)
                    all_events.update(events)

            cognitive_results[file_id] = '; '.join(sorted(all_events)) if all_events else ''

    return cognitive_results


def create_comparison_excel():
    """Main function to create comparison Excel file"""
    print("Loading JSON files...")

    # Load all JSON files
    pann_data = load_json_file(PANN_PATH)
    qwen_data = load_json_file(QWEN_PATH)
    cognitive_data = load_json_file(COGNITIVE_PATH)

    if not all([pann_data, qwen_data, cognitive_data]):
        print("Failed to load one or more JSON files. Exiting.")
        return

    print("Processing data...")

    # Process each dataset
    pann_results = process_pann_data(pann_data)
    qwen_results = process_qwen_data(qwen_data)
    cognitive_results = process_cognitive_data(cognitive_data)

    # Get all file IDs (should be R0001 to R0133)
    all_file_ids = set()
    all_file_ids.update(pann_results.keys())
    all_file_ids.update(qwen_results.keys())
    all_file_ids.update(cognitive_results.keys())

    # Sort file IDs numerically
    def sort_key(file_id):
        match = re.search(r'R(\d+)', file_id)
        return int(match.group(1)) if match else 0

    sorted_file_ids = sorted(all_file_ids, key=sort_key)

    print(f"Found {len(sorted_file_ids)} files to process")

    # Create comparison data
    comparison_data = []

    for file_id in sorted_file_ids:
        row = {
            'File_Index': file_id,
            'PANN_Detected_Events': pann_results.get(file_id, ''),
            'Qwen_Events': qwen_results.get(file_id, ''),
            'Cognitive_Events': cognitive_results.get(file_id, '')
        }
        comparison_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Save to Excel
    print(f"Saving to Excel file: {OUTPUT_PATH}")

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        # Save with formatting
        with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Audio_Analysis_Comparison', index=False)

            # Get the workbook and worksheet for formatting
            workbook = writer.book
            worksheet = writer.sheets['Audio_Analysis_Comparison']

            # Adjust column widths
            worksheet.column_dimensions['A'].width = 12  # File_Index
            worksheet.column_dimensions['B'].width = 50  # PANN_Detected_Events
            worksheet.column_dimensions['C'].width = 40  # Qwen_Events
            worksheet.column_dimensions['D'].width = 40  # Cognitive_Events

            # Format header row
            from openpyxl.styles import Font, PatternFill

            header_font = Font(bold=True)
            header_fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")

            for cell in worksheet[1]:
                cell.font = header_font
                cell.fill = header_fill

        print("✓ Excel file created successfully!")

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total files processed: {len(df)}")
        print(f"PANN results: {len([x for x in df['PANN_Detected_Events'] if x])}")
        print(f"Qwen results: {len([x for x in df['Qwen_Events'] if x])}")
        print(f"Cognitive results: {len([x for x in df['Cognitive_Events'] if x])}")

        # Show first few rows as preview
        print("\nPreview of first 5 rows:")
        print(df.head().to_string(max_colwidth=30))

    except Exception as e:
        print(f"Error saving Excel file: {e}")


def main():
    """Main execution function"""
    print("Audio Analysis Comparison - JSON to Excel Converter")
    print("=" * 55)
    print(f"PANN data: {PANN_PATH}")
    print(f"Qwen data: {QWEN_PATH}")
    print(f"Cognitive data: {COGNITIVE_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 55)

    create_comparison_excel()


if __name__ == "__main__":
    main()