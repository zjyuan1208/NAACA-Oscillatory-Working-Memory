import json
import os
from typing import Dict, List, Tuple, Optional
import numpy as np


class AudioDataset:
    """Audio dataset loader for LU_AVS dataset"""

    def __init__(self, audio_dir: str, label_file: str):
        """
        Initialize the dataset

        Args:
            audio_dir: Path to the directory containing audio files
            label_file: Path to the JSON file containing labels
        """
        self.audio_dir = audio_dir
        self.label_file = label_file
        self.data = self._load_labels()
        self.sample_ids = list(self.data.keys())

    def _load_labels(self) -> Dict:
        """Load labels from JSON file"""
        with open(self.label_file, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self) -> int:
        """Return number of samples in the dataset"""
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Tuple[str, Dict]:
        """
        Get a sample by index

        Args:
            index: Sample index

        Returns:
            Tuple of (sample_id, sample_data)
        """
        sample_id = self.sample_ids[index]
        sample_data = self.data[sample_id]
        return sample_id, sample_data

    def get_sample_by_id(self, sample_id: str) -> Optional[Dict]:
        """
        Get a sample by ID

        Args:
            sample_id: Sample identifier

        Returns:
            Sample data or None if not found
        """
        return self.data.get(sample_id, None)

    def get_audio_path(self, sample_id: str) -> str:
        """
        Get the full path to an audio file

        Args:
            sample_id: Sample identifier

        Returns:
            Full path to the audio file
        """
        sample_data = self.get_sample_by_id(sample_id)
        if sample_data is None:
            raise ValueError(f"Sample {sample_id} not found")

        audio_filename = sample_data['audio_file']
        return os.path.join(self.audio_dir, audio_filename)

    def get_segments_info(self, sample_id: str) -> List[Dict]:
        """
        Get segment information for a sample

        Args:
            sample_id: Sample identifier

        Returns:
            List of segment dictionaries
        """
        sample_data = self.get_sample_by_id(sample_id)
        if sample_data is None:
            raise ValueError(f"Sample {sample_id} not found")

        return sample_data.get('audio_segments', [])

    def get_change_points(self, sample_id: str) -> List[float]:
        """
        Get change points (start and end times of segments) for a sample

        Args:
            sample_id: Sample identifier

        Returns:
            List of change point timestamps
        """
        segments = self.get_segments_info(sample_id)
        change_points = []

        for segment in segments:
            change_points.append(segment['start_time'])
            change_points.append(segment['end_time'])

        # Remove duplicates and sort
        change_points = sorted(list(set(change_points)))
        return change_points

    def get_sample_info(self, sample_id: str) -> Dict:
        """
        Get comprehensive information about a sample

        Args:
            sample_id: Sample identifier

        Returns:
            Dictionary with sample information
        """
        sample_data = self.get_sample_by_id(sample_id)
        if sample_data is None:
            raise ValueError(f"Sample {sample_id} not found")

        audio_path = self.get_audio_path(sample_id)
        segments = self.get_segments_info(sample_id)
        change_points = self.get_change_points(sample_id)

        return {
            'sample_id': sample_id,
            'audio_path': audio_path,
            'video_duration': sample_data.get('video_duration', 0),
            'fps': sample_data.get('fps', 0),
            'total_segments': sample_data.get('total_segments', 0),
            'segments': segments,
            'change_points': change_points,
            'num_change_points': len(change_points)
        }

    def validate_dataset(self) -> Dict[str, List[str]]:
        """
        Validate the dataset by checking if audio files exist

        Returns:
            Dictionary with 'valid' and 'missing' lists of sample IDs
        """
        valid_samples = []
        missing_samples = []

        for sample_id in self.sample_ids:
            try:
                audio_path = self.get_audio_path(sample_id)
                if os.path.exists(audio_path):
                    valid_samples.append(sample_id)
                else:
                    missing_samples.append(sample_id)
            except Exception as e:
                missing_samples.append(sample_id)
                print(f"Error validating {sample_id}: {e}")

        return {
            'valid': valid_samples,
            'missing': missing_samples
        }

    def get_dataset_statistics(self) -> Dict:
        """
        Get statistics about the dataset

        Returns:
            Dictionary with dataset statistics
        """
        total_samples = len(self.sample_ids)
        total_segments = sum(sample_data.get('total_segments', 0) for sample_data in self.data.values())

        durations = [sample_data.get('video_duration', 0) for sample_data in self.data.values()]
        avg_duration = np.mean(durations) if durations else 0

        # Count labels
        label_counts = {}
        for sample_data in self.data.values():
            for segment in sample_data.get('audio_segments', []):
                label = segment.get('label', 'unknown')
                label_counts[label] = label_counts.get(label, 0) + 1

        return {
            'total_samples': total_samples,
            'total_segments': total_segments,
            'average_duration': avg_duration,
            'label_distribution': label_counts,
            'unique_labels': list(label_counts.keys())
        }


def create_dataloader(audio_dir: str, label_file: str, batch_size: int = 1) -> AudioDataset:
    """
    Create a dataloader for the audio dataset

    Args:
        audio_dir: Path to audio directory
        label_file: Path to label JSON file
        batch_size: Batch size (currently not used, but included for future extension)

    Returns:
        AudioDataset instance
    """
    return AudioDataset(audio_dir, label_file)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    audio_dir = "/home/zhyuan/Desktop/PCD/data/LU_AVS/audio_dataset"
    label_file = "/home/zhyuan/Desktop/PCD/data/LU_AVS/audio_label.json"

    # Create dataset
    dataset = create_dataloader(audio_dir, label_file)

    print(f"Dataset loaded with {len(dataset)} samples")

    # Validate dataset
    validation_results = dataset.validate_dataset()
    print(f"Valid samples: {len(validation_results['valid'])}")
    print(f"Missing samples: {len(validation_results['missing'])}")

    if validation_results['missing']:
        print("Missing files:")
        for missing in validation_results['missing'][:5]:  # Show first 5
            print(f"  - {missing}")

    # Get dataset statistics
    stats = dataset.get_dataset_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Total segments: {stats['total_segments']}")
    print(f"  Average duration: {stats['average_duration']:.2f}s")
    print(f"  Unique labels: {len(stats['unique_labels'])}")

    # Show first few samples
    print(f"\nFirst 3 samples:")
    for i in range(min(3, len(dataset))):
        sample_id, sample_data = dataset[i]
        info = dataset.get_sample_info(sample_id)
        print(f"  {sample_id}: {info['num_change_points']} change points, "
              f"{info['total_segments']} segments, {info['video_duration']:.2f}s")