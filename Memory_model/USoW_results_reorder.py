# #!/usr/bin/env python3
# """
# JSON File Reorder Script
# Reorders the results in JSON files based on the numeric index in filenames (R0001 to R0133).
# """
#
# import json
# import re
# import os
# from datetime import datetime
#
#
# def extract_index_from_filename(filename):
#     """Extract numeric index from filename like 'R0077_segment_ambisonics'"""
#     match = re.search(r'R(\d+)_', filename)
#     if match:
#         return int(match.group(1))
#     else:
#         # Fallback: try to extract any number from filename
#         numbers = re.findall(r'\d+', filename)
#         if numbers:
#             return int(numbers[0])
#         else:
#             # If no number found, return a high number to put it at the end
#             return 9999
#
#
# def reorder_json_results(input_file, output_file=None):
#     """Reorder JSON results based on filename index"""
#
#     # Read the input JSON file
#     try:
#         with open(input_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except Exception as e:
#         print(f"Error reading input file {input_file}: {e}")
#         return False
#
#     # Check if the file has the expected structure
#     if 'results' not in data or not isinstance(data['results'], list):
#         print("Error: JSON file doesn't have the expected 'results' array structure")
#         return False
#
#     original_count = len(data['results'])
#     print(f"Original file contains {original_count} results")
#
#     # Sort the results based on the numeric index extracted from filename
#     print("Sorting results by filename index...")
#     data['results'].sort(key=lambda x: extract_index_from_filename(x.get('filename', '')))
#
#     # Add reorder information to dataset_info
#     if 'dataset_info' in data:
#         data['dataset_info']['reordered'] = True
#         data['dataset_info']['reorder_date'] = datetime.now().isoformat()
#         data['dataset_info']['original_order'] = False
#
#     # Determine output filename
#     if output_file is None:
#         # Create output filename based on input filename
#         base_name = os.path.splitext(input_file)[0]
#         output_file = f"{base_name}_reordered.json"
#
#     # Save the reordered JSON
#     try:
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)
#
#         print(f"✓ Successfully reordered and saved to: {output_file}")
#
#         # Print some statistics
#         print("\nReorder Summary:")
#         print(f"Total results: {len(data['results'])}")
#
#         # Show first and last few entries
#         print("\nFirst 5 entries:")
#         for i in range(min(5, len(data['results']))):
#             filename = data['results'][i].get('filename', 'N/A')
#             index = extract_index_from_filename(filename)
#             print(f"  {i + 1}. {filename} (index: R{index:04d})")
#
#         print("\nLast 5 entries:")
#         start_idx = max(0, len(data['results']) - 5)
#         for i in range(start_idx, len(data['results'])):
#             filename = data['results'][i].get('filename', 'N/A')
#             index = extract_index_from_filename(filename)
#             print(f"  {i + 1}. {filename} (index: R{index:04d})")
#
#         # Check for any missing indices in the expected range
#         print("\nMissing indices check:")
#         found_indices = set()
#         for result in data['results']:
#             filename = result.get('filename', '')
#             index = extract_index_from_filename(filename)
#             if index <= 133:  # Only count indices in expected range
#                 found_indices.add(index)
#
#         expected_indices = set(range(1, 134))  # R0001 to R0133
#         missing_indices = expected_indices - found_indices
#
#         if missing_indices:
#             missing_sorted = sorted(missing_indices)
#             print(f"Missing indices: {missing_sorted}")
#         else:
#             print("✓ All indices from R0001 to R0133 are present")
#
#         return True
#
#     except Exception as e:
#         print(f"Error saving output file {output_file}: {e}")
#         return False
#
#
# def main():
#     # """Main function with simple command line usage"""
#     # import sys
#     #
#     # if len(sys.argv) != 2:
#     #     print("Usage: python3 reorder_json.py <input_json_file>")
#     #     print("The script will overwrite the input file with the reordered version.")
#     #     return
#
#     # input_file = sys.argv[1]
#     input_file = "/home/zhyuan/Desktop/audioset_tagging_cnn/results/qwen_results.json"
#
#     # Check if input file exists
#     if not os.path.exists(input_file):
#         print(f"Error: Input file '{input_file}' does not exist")
#         return
#
#     print(f"Reordering JSON file: {input_file}")
#     print("Note: The original file will be overwritten")
#     print("=" * 50)
#
#     # Perform the reordering (overwrite the original file)
#     success = reorder_json_results(input_file, input_file)
#
#     if success:
#         print("=" * 50)
#         print("Reordering completed successfully!")
#     else:
#         print("=" * 50)
#         print("Reordering failed!")
#
#
# if __name__ == "__main__":
#     main()

# !/usr/bin/env python3
"""
Script to remove "all_cosine_similarities" field from the saved JSON file and
reorder results based on file names (R0001, R0002, etc.).
This will clean up the JSON file by removing unnecessary data while preserving
the cosine similarities that are associated with detected pattern changes.
"""

import json
import os
import shutil
import re
from datetime import datetime


def extract_file_number(file_id):
    """
    Extract numeric part from file ID (e.g., 'R0077' -> 77).

    Args:
        file_id (str): File ID like 'R0077' or 'Unknown'

    Returns:
        int: Numeric part of the file ID, or 9999 for unknown files
    """
    if not file_id or file_id == "Unknown":
        return 9999  # Put unknown files at the end

    match = re.search(r'R(\d+)', str(file_id))
    if match:
        return int(match.group(1))

    return 9999  # Fallback for unexpected formats


def clean_json_file(json_path):
    """
    Remove 'all_cosine_similarities' field from each result in the JSON file.

    Args:
        json_path (str): Path to the JSON file to clean
    """

    # Check if file exists
    if not os.path.exists(json_path):
        print(f"❌ Error: File {json_path} does not exist!")
        return False

    try:
        # Create backup first
        backup_path = json_path.replace('.json', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        shutil.copy2(json_path, backup_path)
        print(f"✅ Backup created: {backup_path}")

        # Load the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"📄 Loaded JSON file with {len(data.get('results', []))} results")

        # Track changes
        removed_count = 0
        total_similarities_removed = 0

        # Process each result
        if 'results' in data:
            for i, result in enumerate(data['results']):
                if 'all_cosine_similarities' in result:
                    similarities_count = len(result['all_cosine_similarities'])
                    total_similarities_removed += similarities_count

                    # Remove the field
                    del result['all_cosine_similarities']
                    removed_count += 1

                    print(
                        f"  🧹 Cleaned result {i + 1} (file: {result.get('file_id', 'unknown')}): removed {similarities_count} similarity entries")

            # Sort results by file name (R0001, R0002, etc.)
            print(f"\n📋 Sorting {len(data['results'])} results by file name...")
            original_order = [result.get('file_id', 'Unknown') for result in data['results']]

            data['results'].sort(key=lambda x: extract_file_number(x.get('file_id', 'Unknown')))

            sorted_order = [result.get('file_id', 'Unknown') for result in data['results']]

            print(f"  ✅ Results reordered:")
            print(f"     Original order: {', '.join(original_order)}")
            print(f"     New order: {', '.join(sorted_order)}")

        # Save the cleaned JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Successfully cleaned and sorted JSON file!")
        print(f"📊 Summary:")
        print(f"   - Cleaned {removed_count} result entries")
        print(f"   - Removed {total_similarities_removed} total similarity entries")
        print(f"   - Sorted {len(data.get('results', []))} results by file name")
        print(f"   - Original file backed up to: {backup_path}")
        print(f"   - Cleaned file saved to: {json_path}")

        # Calculate file size reduction
        original_size = os.path.getsize(backup_path)
        new_size = os.path.getsize(json_path)
        size_reduction = original_size - new_size
        reduction_percent = (size_reduction / original_size) * 100 if original_size > 0 else 0

        print(f"💾 File size reduction: {size_reduction:,} bytes ({reduction_percent:.1f}%)")
        print(f"   - Original: {original_size:,} bytes")
        print(f"   - New: {new_size:,} bytes")

        return True

    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format - {e}")
        return False
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return False


def main():
    # json_path = "/home/zhyuan/Desktop/Qwen-Audio/results/pattern_change_results.json"
    json_path = "/home/zhyuan/Desktop/Qwen-Audio/results/biooss_multimetric_pattern_change_results_realtime.json"

    print("🧹 JSON Cleaner - Removing 'all_cosine_similarities' and Sorting by File Name")
    print("=" * 80)
    print(f"Target file: {json_path}")
    print()

    success = clean_json_file(json_path)

    if success:
        print("\n🎉 Cleaning completed successfully!")
    else:
        print("\n💥 Cleaning failed!")

    print("=" * 80)


if __name__ == "__main__":
    main()