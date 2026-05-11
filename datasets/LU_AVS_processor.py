import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_audio_from_video(video_path, audio_output_path):
    """
    Extract audio from video using ffmpeg
    """
    try:
        # Use ffmpeg to extract audio and convert to wav format
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # Exclude video
            '-acodec', 'pcm_s16le',  # 16-bit PCM encoding
            '-ar', '16000',  # Sample rate 16kHz
            '-ac', '1',  # Mono audio
            '-y',  # Overwrite output file
            audio_output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to extract audio: {video_path}")
            logger.error(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error occurred while extracting audio: {e}")
        return False


def load_json_annotations(json_paths):
    """
    Load all JSON annotation files
    """
    all_annotations = {}

    for json_path in json_paths:
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_annotations.update(data)
                logger.info(f"Loaded annotation file: {json_path}, containing {len(data)} videos")
            except Exception as e:
                logger.error(f"Failed to load JSON file {json_path}: {e}")
        else:
            logger.warning(f"Annotation file not found: {json_path}")

    return all_annotations


def convert_frame_to_time(frame_idx, fps):
    """
    Convert frame index to timestamp (seconds)
    """
    return frame_idx / fps


def process_videos_and_extract_audio():
    """
    Main processing function: Extract audio and organize labels
    """
    # Define paths
    video_dir = "/home/zhyuan/Desktop/PCD/data/LU_AVS/project/_liuchen/AVSBenchData/LAVS_Seg_Videos"
    json_paths = [
        "/home/zhyuan/Desktop/PCD/data/LU_AVS/seg_test.json",
        "/home/zhyuan/Desktop/PCD/data/LU_AVS/seg_train.json",
        "/home/zhyuan/Desktop/PCD/data/LU_AVS/seg_val.json"
    ]
    output_audio_dir = "/home/zhyuan/Desktop/PCD/data/LU_AVS/audio_dataset"
    output_label_path = "/home/zhyuan/Desktop/PCD/data/LU_AVS/audio_label.json"

    # Create output directory
    Path(output_audio_dir).mkdir(parents=True, exist_ok=True)

    # Load all annotations
    logger.info("Loading annotation files...")
    annotations = load_json_annotations(json_paths)

    if not annotations:
        logger.error("No annotation data loaded!")
        return

    logger.info(f"Loaded annotations for {len(annotations)} videos")

    # Check if video directory exists
    if not os.path.exists(video_dir):
        logger.error(f"Video directory does not exist: {video_dir}")
        return

    # Get all video files
    video_files = []
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI', '*.MOV', '*.MKV']

    logger.info(f"Searching for video files in: {video_dir}")
    for ext in video_extensions:
        found_files = list(Path(video_dir).glob(ext))
        video_files.extend(found_files)
        if found_files:
            logger.info(f"Found {len(found_files)} {ext} files")

    # If no files found, try recursive search
    if not video_files:
        logger.warning("No video files found in root directory, attempting recursive search...")
        for ext in video_extensions:
            found_files = list(Path(video_dir).rglob(ext))
            video_files.extend(found_files)
            if found_files:
                logger.info(f"Recursively found {len(found_files)} {ext} files")

    logger.info(f"Total {len(video_files)} video files found")

    # If still no video files found, list directory contents
    if not video_files:
        logger.error("No video files found!")
        logger.info("Directory contents:")
        try:
            for item in os.listdir(video_dir):
                item_path = os.path.join(video_dir, item)
                if os.path.isfile(item_path):
                    logger.info(f"  File: {item}")
                elif os.path.isdir(item_path):
                    logger.info(f"  Directory: {item}/")
        except Exception as e:
            logger.error(f"Failed to list directory contents: {e}")
        return

    # Final audio label data
    audio_labels = {}
    processed_count = 0

    # Process each video
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_name = video_file.stem  # Filename without extension

        # Check if there is a corresponding annotation
        if video_name not in annotations:
            logger.warning(f"Video {video_name} has no annotation, skipping")
            continue

        video_annotation = annotations[video_name]

        # Extract audio
        audio_output_path = os.path.join(output_audio_dir, f"{video_name}.wav")

        logger.info(f"Processing video: {video_name}")

        if extract_audio_from_video(str(video_file), audio_output_path):
            # Get video information
            fps = video_annotation.get('fps', 30.0)  # Default 30fps
            duration = video_annotation.get('duration', 0)

            # Process audio segment annotations
            audio_segments = []

            for annotation in video_annotation.get('annotations', []):
                segment = annotation.get('segment', [])
                if len(segment) >= 2:
                    start_frame = segment[0]
                    end_frame = segment[1]

                    # Convert to timestamps
                    start_time = convert_frame_to_time(start_frame, fps)
                    end_time = convert_frame_to_time(end_frame, fps)

                    audio_segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'label': annotation.get('label', ''),
                        'label_id': annotation.get('label_id', -1)
                    })

            # Save to audio label data
            audio_labels[video_name] = {
                'audio_file': f"{video_name}.wav",
                'video_duration': duration,
                'fps': fps,
                'audio_segments': audio_segments,
                'total_segments': len(audio_segments)
            }

            processed_count += 1
            logger.info(f"Successfully processed: {video_name}, containing {len(audio_segments)} audio segments")
        else:
            logger.error(f"Failed to extract audio: {video_name}")

    # Save audio label file
    logger.info(f"Saving audio labels to: {output_label_path}")
    try:
        with open(output_label_path, 'w', encoding='utf-8') as f:
            json.dump(audio_labels, f, ensure_ascii=False, indent=2)

        logger.info(f"Processing complete! Successfully processed {processed_count} videos")
        logger.info(f"Audio files saved in: {output_audio_dir}")
        logger.info(f"Label file saved at: {output_label_path}")

        # Statistics
        total_segments = sum(len(data['audio_segments']) for data in audio_labels.values())
        logger.info(f"Total {total_segments} audio segments extracted")

    except Exception as e:
        logger.error(f"Failed to save label file: {e}")


def print_dataset_statistics(label_path):
    """
    Print dataset statistics
    """
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_videos = len(data)
        total_segments = sum(len(video_data['audio_segments']) for video_data in data.values())

        # Label distribution statistics
        label_counts = {}
        for video_data in data.values():
            for segment in video_data['audio_segments']:
                label = segment['label']
                label_counts[label] = label_counts.get(label, 0) + 1

        print(f"\n=== Dataset Statistics ===")
        print(f"Total number of videos: {total_videos}")
        print(f"Total number of audio segments: {total_segments}")
        print(f"Average segments per video: {total_segments / total_videos:.2f}" if total_videos > 0 else "Average segments per video: 0")
        print(f"\n=== Label Distribution (Top 10) ===")
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for label, count in sorted_labels:
            print(f"{label}: {count}")

    except Exception as e:
        logger.error(f"Failed to read dataset statistics: {e}")



if __name__ == "__main__":
    logger.info("Starting to process LU-AVS dataset...")

    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        logger.info("FFmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg is not available, please install FFmpeg first")
        exit(1)

    # Run main processing
    process_videos_and_extract_audio()

    # Print statistics
    label_path = "/home/zhyuan/Desktop/PCD/data/LU_AVS/audio_label.json"
    if os.path.exists(label_path):
        print_dataset_statistics(label_path)
