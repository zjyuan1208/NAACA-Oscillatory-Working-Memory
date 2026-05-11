#!/usr/bin/env python3
"""
Attention Gate Open Times Calculator
Processes pattern_change_results.json to calculate attention gate open times for each sample.
Attention gate open times = len(change_points) + 1 (including initial 15s segment)
"""

import json
import pandas as pd
import os
from datetime import datetime
import re

# File paths
INPUT_PATH = "/home/zhyuan/Desktop/Qwen-Audio/results/pattern_change_results.json"
OUTPUT_PATH = "/home/zhyuan/Desktop/Qwen-Audio/results/attention_gate_analysis.xlsx"


def extract_file_index(file_id):
    """Extract numeric index from file ID like 'R0077' -> 77"""
    match = re.search(r'R(\d+)', file_id)
    if match:
        return int(match.group(1))
    return 0


def calculate_attention_gates():
    """Calculate attention gate open times for each sample"""

    print("Loading pattern change results...")

    # Load the JSON data
    try:
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file {INPUT_PATH}: {e}")
        return

    if 'results' not in data:
        print("Error: JSON file doesn't have expected 'results' structure")
        return

    print(f"Processing {len(data['results'])} samples...")

    # Process each sample
    analysis_data = []

    for result in data['results']:
        file_id = result.get('file_id', 'Unknown')
        total_length = result.get('total_file_length', 0)
        change_points = result.get('change_points', [])
        time_sent_to_qwen = result.get('time_length_sent_to_qwen', 0)

        # Calculate attention gate open times
        # Formula: len(change_points) + 1 (for initial 15s segment)
        attention_gate_opens = len(change_points) + 1

        # Calculate processing efficiency
        processing_efficiency = (time_sent_to_qwen / total_length * 100) if total_length > 0 else 0

        # Extract file index for sorting
        file_index = extract_file_index(file_id)

        # Collect pattern change timestamps and similarities
        pattern_changes = result.get('pattern_changes', [])

        # Calculate average similarity if available
        similarities = [pc.get('cosine_similarity', 0) for pc in pattern_changes]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        min_similarity = min(similarities) if similarities else 0
        max_similarity = max(similarities) if similarities else 0

        analysis_data.append({
            'File_ID': file_id,
            'File_Index': file_index,
            'Total_Length_Seconds': total_length,
            'Change_Points_Count': len(change_points),
            'Attention_Gate_Opens': attention_gate_opens,
            'Time_Sent_to_Qwen': time_sent_to_qwen,
            'Processing_Efficiency_Percent': round(processing_efficiency, 2),
            'Change_Points': change_points,
            'Avg_Cosine_Similarity': round(avg_similarity, 4) if avg_similarity > 0 else None,
            'Min_Cosine_Similarity': round(min_similarity, 4) if min_similarity > 0 else None,
            'Max_Cosine_Similarity': round(max_similarity, 4) if max_similarity > 0 else None,
            'Pattern_Changes_Detail': len(pattern_changes)
        })

    # Convert to DataFrame and sort by file index
    df = pd.DataFrame(analysis_data)
    df = df.sort_values('File_Index')

    # Calculate summary statistics
    total_samples = len(df)
    total_attention_gates = df['Attention_Gate_Opens'].sum()
    avg_attention_gates = df['Attention_Gate_Opens'].mean()
    avg_processing_efficiency = df['Processing_Efficiency_Percent'].mean()

    # Samples with different numbers of attention gates
    static_samples = len(df[df['Attention_Gate_Opens'] == 1])  # Only initial segment
    low_dynamic = len(df[df['Attention_Gate_Opens'].between(2, 3)])
    medium_dynamic = len(df[df['Attention_Gate_Opens'].between(4, 6)])
    high_dynamic = len(df[df['Attention_Gate_Opens'] > 6])

    print(f"\n=== ATTENTION GATE ANALYSIS SUMMARY ===")
    print(f"Total samples: {total_samples}")
    print(f"Total attention gate opens: {total_attention_gates}")
    print(f"Average attention gates per sample: {avg_attention_gates:.2f}")
    print(f"Average processing efficiency: {avg_processing_efficiency:.2f}%")
    print(f"\nSample distribution:")
    print(f"  Static (1 gate): {static_samples} samples ({static_samples / total_samples * 100:.1f}%)")
    print(f"  Low dynamic (2-3 gates): {low_dynamic} samples ({low_dynamic / total_samples * 100:.1f}%)")
    print(f"  Medium dynamic (4-6 gates): {medium_dynamic} samples ({medium_dynamic / total_samples * 100:.1f}%)")
    print(f"  High dynamic (>6 gates): {high_dynamic} samples ({high_dynamic / total_samples * 100:.1f}%)")

    # Find extremes
    max_gates_sample = df.loc[df['Attention_Gate_Opens'].idxmax()]
    min_gates_sample = df.loc[df['Attention_Gate_Opens'].idxmin()]

    print(f"\nMost dynamic sample: {max_gates_sample['File_ID']} with {max_gates_sample['Attention_Gate_Opens']} gates")
    print(f"Least dynamic sample: {min_gates_sample['File_ID']} with {min_gates_sample['Attention_Gate_Opens']} gates")

    # Save to Excel with multiple sheets
    print(f"\nSaving results to {OUTPUT_PATH}...")

    try:
        with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
            # Main analysis sheet
            df.to_excel(writer, sheet_name='Attention_Gate_Analysis', index=False)

            # Summary statistics sheet
            summary_data = {
                'Metric': [
                    'Total Samples',
                    'Total Attention Gate Opens',
                    'Average Gates per Sample',
                    'Average Processing Efficiency (%)',
                    'Static Samples (1 gate)',
                    'Low Dynamic Samples (2-3 gates)',
                    'Medium Dynamic Samples (4-6 gates)',
                    'High Dynamic Samples (>6 gates)',
                    'Most Dynamic Sample',
                    'Least Dynamic Sample'
                ],
                'Value': [
                    total_samples,
                    total_attention_gates,
                    round(avg_attention_gates, 2),
                    round(avg_processing_efficiency, 2),
                    static_samples,
                    low_dynamic,
                    medium_dynamic,
                    high_dynamic,
                    f"{max_gates_sample['File_ID']} ({max_gates_sample['Attention_Gate_Opens']} gates)",
                    f"{min_gates_sample['File_ID']} ({min_gates_sample['Attention_Gate_Opens']} gates)"
                ]
            }

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)

            # Distribution analysis
            distribution_data = []
            for gates in range(1, df['Attention_Gate_Opens'].max() + 1):
                count = len(df[df['Attention_Gate_Opens'] == gates])
                percentage = count / total_samples * 100
                distribution_data.append({
                    'Attention_Gates': gates,
                    'Sample_Count': count,
                    'Percentage': round(percentage, 2)
                })

            dist_df = pd.DataFrame(distribution_data)
            dist_df.to_excel(writer, sheet_name='Distribution_Analysis', index=False)

            # Format the main sheet
            workbook = writer.book
            worksheet = writer.sheets['Attention_Gate_Analysis']

            # Adjust column widths
            worksheet.column_dimensions['A'].width = 12  # File_ID
            worksheet.column_dimensions['B'].width = 12  # File_Index
            worksheet.column_dimensions['C'].width = 18  # Total_Length_Seconds
            worksheet.column_dimensions['D'].width = 18  # Change_Points_Count
            worksheet.column_dimensions['E'].width = 20  # Attention_Gate_Opens
            worksheet.column_dimensions['F'].width = 18  # Time_Sent_to_Qwen
            worksheet.column_dimensions['G'].width = 22  # Processing_Efficiency_Percent
            worksheet.column_dimensions['H'].width = 25  # Change_Points
            worksheet.column_dimensions['I'].width = 20  # Avg_Cosine_Similarity

            # Format header
            from openpyxl.styles import Font, PatternFill

            header_font = Font(bold=True)
            header_fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")

            for cell in worksheet[1]:
                cell.font = header_font
                cell.fill = header_fill

        print("✅ Excel file created successfully!")

        # Show some examples
        print(f"\n=== SAMPLE EXAMPLES ===")
        print("Files with highest attention gate activity:")
        top_5 = df.nlargest(5, 'Attention_Gate_Opens')[
            ['File_ID', 'Attention_Gate_Opens', 'Change_Points_Count', 'Processing_Efficiency_Percent']]
        for _, row in top_5.iterrows():
            print(
                f"  {row['File_ID']}: {row['Attention_Gate_Opens']} gates ({row['Change_Points_Count']} changes, {row['Processing_Efficiency_Percent']}% processed)")

        print("\nFiles with static behavior (1 gate only):")
        static_files = df[df['Attention_Gate_Opens'] == 1][['File_ID', 'Processing_Efficiency_Percent']].head(10)
        for _, row in static_files.iterrows():
            print(f"  {row['File_ID']}: Static ({row['Processing_Efficiency_Percent']}% processed)")

    except Exception as e:
        print(f"Error saving Excel file: {e}")


def main():
    """Main execution function"""
    print("Attention Gate Analysis - Pattern Change Processor")
    print("=" * 55)
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print("=" * 55)

    # Check if input file exists
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file '{INPUT_PATH}' does not exist")
        return

    calculate_attention_gates()

    print("=" * 55)
    print("Analysis completed successfully!")


if __name__ == "__main__":
    main()