import os

folder_path = "/home/zhyuan/Desktop/PCD/data/LU_AVS/audio_dataset"
# /home/zhyuan/Desktop/PCD/data/LU_AVS/audio_dataset/-1WK72M4xeg.wav
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        clean_name = filename.lstrip("-")  # removes all leading dashes
        if clean_name != filename:
            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, clean_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")


print("Renaming complete.")
