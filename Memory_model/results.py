import os
import re
import pandas as pd
from glob import glob


def extract_environment_category(file_path):
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read().lower()  # Convert to lowercase for case-insensitive matching

    # Extract the file index from the filename
    file_index = re.search(r'W_component_R(\d+)\.txt', file_path)
    if file_index:
        index = file_index.group(1)
    else:
        index = os.path.basename(file_path)  # Fallback to filename if pattern doesn't match

    # Keywords for lively environment - expanded
    lively_keywords = ["lively/active", "lively", "active", "car", "busy", "traffic",
                       "noisy", "loud", "bustling", "crowded", "vehicle", "engine", "train", "airport"]

    # Keywords for tranquil environment
    tranquil_keywords = ["calming/tranquil", "tranquil", "calming", "quiet", "peaceful",
                         "serene", "gentle", "relaxing"]

    # Check for keywords
    for keyword in tranquil_keywords:
        if keyword in content:
            environment = "tranquil"
            break
    else:  # No tranquil keywords found
        for keyword in lively_keywords:
            if keyword in content:
                environment = "lively"
                break
        else:  # No lively keywords found
            environment = "neither"

    return index, environment


def main():
    # Path to the directory containing the files
    directory = "/home/zhyuan/Desktop/Qwen-Audio/results"

    # Find all text files matching the pattern
    file_pattern = os.path.join(directory, "W_component_R*.txt")
    files = glob(file_pattern)

    # Extract data from each file
    data = []
    for file_path in files:
        index, environment = extract_environment_category(file_path)
        data.append({"Index": index, "Environment": environment})

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Sort by index
    df["Index"] = df["Index"].astype(int)
    df = df.sort_values("Index")

    # Save to Excel
    output_path = os.path.join('/home/zhyuan/Desktop/Qwen-Audio/Memory_model/results', "environment_categories.xlsx")
    df.to_excel(output_path, index=False)

    print(f"Data extracted and saved to {output_path}")
    print(f"Processed {len(files)} files")


if __name__ == "__main__":
    main()