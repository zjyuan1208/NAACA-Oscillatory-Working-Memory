import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from typing import List, Tuple, Dict, Optional, Set
import warnings
from collections import deque
import math

warnings.filterwarnings('ignore')


class AdaptiveOutlierDetector:
    """
    noteoutliernote
    note：note，note
    """

    def __init__(self,
                 input_dim: int = 527,
                 k_max: int = 5,
                 distance_metric: str = 'mahalanobis',
                 initial_threshold_factor: float = 3.0, # Implementation note.
                 threshold_adaptation_rate: float = 0.05, # Reduce.
                 consecutive_threshold: int = 3, # Implementation note.
                 cluster_update_interval: int = 20, # Implementation note.
                 min_samples_for_mahalanobis: int = 10, # distance.
                 stability_window: int = 5): # stability window.
        """
        note:
        - k_max: note
        - distance_metric: note ('cosine', 'euclidean', 'mahalanobis')
        - initial_threshold_factor: note (note = note + factor * note)
        - threshold_adaptation_rate: note
        - consecutive_threshold: note
        - cluster_update_interval: note
        - min_samples_for_mahalanobis: note
        - stability_window: note
        """
        self.input_dim = input_dim
        self.k_max = k_max
        self.distance_metric = distance_metric
        self.initial_threshold_factor = initial_threshold_factor
        self.threshold_adaptation_rate = threshold_adaptation_rate
        self.consecutive_threshold = consecutive_threshold
        self.cluster_update_interval = cluster_update_interval
        self.min_samples_for_mahalanobis = min_samples_for_mahalanobis
        self.stability_window = stability_window

        # State.
        self.reset()

        print(f"Initialize.")
        print(f"cluster.")
        print(f"distance.")
        print(f"Threshold.")
        print(f"Threshold.")
        print(f"Implementation note.")
        print(f"cluster.")
        print(f"distance.")
        print(f"stability window.")

    def reset(self):
        """ State. """
        self.frame_count = 0
        self.cluster_centers = []
        self.distance_history = []
        self.threshold = None
        self.consecutive_outliers = 0
        self.normal_samples = [] # covariance matrix.
        self.cov_matrix = None
        self.inv_cov_matrix = None
        self.last_cluster_creation = -self.stability_window # cluster.

        # Statistics.
        self.total_clusters_created = 0
        self.threshold_updates = 0
        self.false_positive_reduction = 0

    def _compute_mahalanobis_distance(self, feature: np.ndarray) -> float:
        """ cluster. """
        if not self.cluster_centers:
            return 0.0

        feature = feature.reshape(1, -1)
        min_distance = float('inf')

        # distance.
        if (len(self.normal_samples) < self.min_samples_for_mahalanobis or
                self.inv_cov_matrix is None):
            for center in self.cluster_centers:
                center = center.reshape(1, -1)
                distance = np.linalg.norm(feature - center)
                min_distance = min(min_distance, distance)
            return min_distance

        # distance.
        for center in self.cluster_centers:
            try:
                diff = feature - center.reshape(1, -1)
                distance = np.sqrt(np.clip(diff @ self.inv_cov_matrix @ diff.T, 0, None))[0, 0]
                min_distance = min(min_distance, distance)
            except:
                # distance.
                distance = np.linalg.norm(feature - center.reshape(1, -1))
                min_distance = min(min_distance, distance)

        return min_distance

    def _compute_distance(self, feature: np.ndarray) -> float:
        """ cluster. """
        if not self.cluster_centers:
            return 0.0

        feature = feature.reshape(1, -1)

        if self.distance_metric == 'mahalanobis':
            return self._compute_mahalanobis_distance(feature)

        elif self.distance_metric == 'cosine':
            # distance.
            from sklearn.metrics.pairwise import cosine_distances
            centers = np.array(self.cluster_centers)
            distances = cosine_distances(feature, centers)[0]
            return np.min(distances)

        elif self.distance_metric == 'euclidean':
            # distance.
            min_distance = float('inf')
            for center in self.cluster_centers:
                distance = np.linalg.norm(feature - center.reshape(1, -1))
                min_distance = min(min_distance, distance)
            return min_distance

        return 0.0

    def _update_covariance_matrix(self):
        """ covariance matrix. """
        if len(self.normal_samples) < self.min_samples_for_mahalanobis:
            return

        try:
            samples_array = np.array(self.normal_samples)
            self.cov_matrix = np.cov(samples_array.T)

            # avoid.
            eigenvals = np.linalg.eigvals(self.cov_matrix)
            min_eigenval = np.min(eigenvals)
            regularization = max(1e-4, -min_eigenval + 1e-6) * np.eye(self.cov_matrix.shape[0])
            self.cov_matrix += regularization

            self.inv_cov_matrix = np.linalg.pinv(self.cov_matrix)

        except Exception as e:
            print(f"covariance matrix.")
            self.inv_cov_matrix = None

    def _should_create_new_cluster(self, feature: np.ndarray, distance: float) -> bool:
        """ cluster. """
        # cluster.
        if (len(self.cluster_centers) < self.k_max and
                self.frame_count - self.last_cluster_creation > self.stability_window):

            # cluster.
            if len(self.distance_history) > 20:
                # Threshold.
                recent_distances = self.distance_history[-30:]
                distance_mean = np.mean(recent_distances)
                distance_std = np.std(recent_distances)
                # Implementation note.
                create_threshold = distance_mean + 2.5 * distance_std

                # Check.
                threshold_check = self.threshold * 1.5 if self.threshold else create_threshold

                return distance > max(create_threshold, threshold_check)
            else:
                # cluster.
                base_threshold = np.mean(self.distance_history) if self.distance_history else 1.0
                return distance > base_threshold * 3.0 # Implementation note.
        return False

    def _update_threshold(self, distance: float, is_outlier: bool = False):
        """ Threshold. """
        self.distance_history.append(distance)

        # Implementation note.
        max_history = 100 # Implementation note.
        if len(self.distance_history) > max_history:
            self.distance_history.pop(0)

        if len(self.distance_history) < 5:
            # Threshold.
            mean_dist = np.mean(self.distance_history)
            std_dist = np.std(self.distance_history) if len(self.distance_history) > 1 else mean_dist * 0.1
            self.threshold = mean_dist + self.initial_threshold_factor * (std_dist + 1e-6)
        else:
            # Threshold.
            recent_window = min(50, len(self.distance_history)) # distance.
            recent_distances = self.distance_history[-recent_window:]

            # Threshold.
            if self.threshold is not None:
                normal_distances = [d for d in recent_distances if d <= self.threshold * 1.2]
                if len(normal_distances) >= 5:
                    # Threshold.
                    normal_mean = np.mean(normal_distances)
                    normal_std = np.std(normal_distances)
                    new_threshold = normal_mean + self.initial_threshold_factor * normal_std
                else:
                    # distance.
                    new_threshold = np.mean(recent_distances) + self.initial_threshold_factor * np.std(recent_distances)
            else:
                new_threshold = np.mean(recent_distances) + self.initial_threshold_factor * np.std(recent_distances)

            if self.threshold is None:
                self.threshold = new_threshold
            else:
                # Threshold.
                if not is_outlier:
                    self.threshold = (1 - self.threshold_adaptation_rate) * self.threshold + \
                                     self.threshold_adaptation_rate * new_threshold
                    self.threshold_updates += 1

    def _add_cluster_center(self, feature: np.ndarray):
        """Add a new cluster center."""
        self.cluster_centers.append(feature.copy())
        self.total_clusters_created += 1
        self.last_cluster_creation = self.frame_count
        print(f"Created cluster center #{self.total_clusters_created} (frame {self.frame_count}), total centers: {len(self.cluster_centers)}")

    def _update_clusters_periodically(self):
        """Periodically recluster normal samples to refine centers."""
        if (len(self.normal_samples) >= 30 and  # Require enough samples for stable reclustering.
                len(self.normal_samples) % self.cluster_update_interval == 0 and
                len(self.cluster_centers) >= 2):

            try:
                # normal sample.
                recent_samples = min(50, len(self.normal_samples))
                samples_array = np.array(self.normal_samples[-recent_samples:])
                k_actual = min(len(self.cluster_centers), len(samples_array) // 5) # cluster.

                if k_actual >= 2:
                    # Implementation note.
                    kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init=10)
                    kmeans.fit(samples_array)
                    self.cluster_centers = list(kmeans.cluster_centers_)
                    print(f"cluster.")

            except Exception as e:
                print(f"cluster.")

    def process_frame(self, feature: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        note
        note: (note, note, note)
        """
        if isinstance(feature, torch.Tensor):
            feature = feature.cpu().numpy()

        feature = feature.flatten()
        self.frame_count += 1

        # cluster.
        if self.frame_count == 1:
            self._add_cluster_center(feature)
            self.normal_samples.append(feature)
            self._update_threshold(0.0) # distance.

            info = {
                'frame_count': self.frame_count,
                'distance': 0.0,
                'threshold': self.threshold,
                'cluster_count': len(self.cluster_centers),
                'consecutive_outliers': 0,
                'is_first_frame': True
            }
            return False, 0.0, info

        # cluster.
        distance = self._compute_distance(feature)
        change_detected = False

        # outlier detection.
        if self.threshold is not None and distance > self.threshold:
            self.consecutive_outliers += 1

            # consecutive outliers.
            if self.consecutive_outliers >= self.consecutive_threshold:
                change_detected = True
                print(
                    f"Threshold.")

                # cluster.
                if self._should_create_new_cluster(feature, distance):
                    self._add_cluster_center(feature)
                    self.normal_samples.append(feature)
                    self.consecutive_outliers = 0 # consecutive outliers.

            # Threshold.
            self._update_threshold(distance, is_outlier=True)

        else:
            # normal sample.
            if self.consecutive_outliers > 0:
                print(f"Threshold.")

            self.consecutive_outliers = 0

            # distance.
            if len(self.cluster_centers) == 0 or distance <= (self.threshold or float('inf')):
                self.normal_samples.append(feature)

                # normal sample.
                max_normal_samples = 100
                if len(self.normal_samples) > max_normal_samples:
                    self.normal_samples.pop(0)

                # distance.
                if self.distance_metric == 'mahalanobis':
                    self._update_covariance_matrix()

            # Threshold.
            self._update_threshold(distance, is_outlier=False)

            # cluster.
            self._update_clusters_periodically()

        info = {
            'frame_count': self.frame_count,
            'distance': distance,
            'threshold': self.threshold,
            'cluster_count': len(self.cluster_centers),
            'consecutive_outliers': self.consecutive_outliers,
            'normal_samples_count': len(self.normal_samples),
            'distance_history_size': len(self.distance_history),
            'total_clusters_created': self.total_clusters_created,
            'threshold_updates': self.threshold_updates
        }

        return change_detected, distance, info

    def batch_detect(self, data_sequence: torch.Tensor) -> Dict:
        """
        note
        """
        T = data_sequence.shape[0]

        detections = []
        confidence_scores = []
        timestamps = []
        detected_changes = []
        change_timestamps = []

        print(f"from the first frame.")
        print(f"Implementation note.")
        print(f"distance.")
        print(f"cluster.")

        for i in range(T):
            feature = data_sequence[i]
            is_change, distance, info = self.process_frame(feature)

            detections.append(is_change)
            confidence_scores.append(distance)
            timestamps.append(i)

            if is_change:
                change_timestamps.append(i)

                change_info = {
                    'position': i,
                    'timestamp': i,
                    'confidence': distance,
                    'distance': distance,
                    'threshold': info.get('threshold', 0),
                    'consecutive_outliers': info.get('consecutive_outliers', 0),
                    'cluster_count': info.get('cluster_count', 0),
                    'time_step': i
                }
                detected_changes.append(change_info)

        # Statistics.
        final_stats = {
            'total_frames_processed': T,
            'final_cluster_count': len(self.cluster_centers),
            'total_clusters_created': self.total_clusters_created,
            'final_threshold': self.threshold,
            'normal_samples_collected': len(self.normal_samples),
            'distance_metric_used': self.distance_metric,
            'threshold_updates': self.threshold_updates
        }

        print(f"Detection.")
        print(f"Implementation note.")
        print(f"cluster.")
        print(f"cluster.")
        print(f"Threshold.")
        print(f"normal sample.")
        print(f"Threshold.")

        return {
            'detections': detections,
            'detected_changes': detected_changes,
            'change_timestamps': change_timestamps,
            'confidence_scores': confidence_scores,
            'timestamps': timestamps,
            'total_detections': len(detected_changes),
            'total_windows': T,
            'phase_statistics': final_stats,
            'window_params': {
                'processing_mode': 'immediate_adaptive_outlier_detection',
                'audio_length': T,
                'distance_metric': self.distance_metric,
                'max_clusters': self.k_max,
                'initial_threshold_factor': self.initial_threshold_factor,
                'consecutive_threshold': self.consecutive_threshold
            }
        }


class OptimalKMeansDetector:
    """ Parameters. """

    def __init__(self, input_dim: int = 527, device: str = 'cpu'):
        self.input_dim = input_dim
        self.device = device
        self.optimal_k = None
        self.optimal_threshold_factor = None

    def find_optimal_k(self, sample_data: np.ndarray, k_range: range = range(3, 8)) -> int:
        """ silhouette score. """
        if len(sample_data) < max(k_range):
            return min(len(sample_data) // 2, 5)

        silhouette_scores = []
        k_values = []

        for k in k_range:
            if k >= len(sample_data):
                break

            try:
                if k == 1:
                    continue

                # Implementation note.
                gmm = GaussianMixture(n_components=k, random_state=42)
                labels = gmm.fit_predict(sample_data)

                if len(np.unique(labels)) > 1:
                    score = silhouette_score(sample_data, labels)
                    silhouette_scores.append(score)
                    k_values.append(k)
            except Exception as e:
                continue

        if silhouette_scores:
            optimal_k = k_values[np.argmax(silhouette_scores)]
            print(f"silhouette score.")
            return optimal_k
        else:
            return 4

    def tune_parameters(self, validation_data: torch.Tensor):
        """ Parameters. """
        if isinstance(validation_data, torch.Tensor):
            validation_data = validation_data.cpu().numpy()

        # Tune.
        sample_size = min(500, len(validation_data) // 2)
        sample_indices = np.random.choice(len(validation_data), sample_size, replace=False)
        sample_data = validation_data[sample_indices]

        self.optimal_k = self.find_optimal_k(sample_data)

        # Threshold.
        distances = []
        temp_detector = AdaptiveOutlierDetector(k_max=self.optimal_k)

        # Threshold.
        for i in range(min(50, len(sample_data))):
            _, distance, _ = temp_detector.process_frame(sample_data[i])
            if distance > 0:
                distances.append(distance)

        if distances:
            # Threshold.
            distance_std = np.std(distances)
            distance_mean = np.mean(distances)
            if distance_std > 0:
                self.optimal_threshold_factor = max(1.5, min(3.0, 2.0 * distance_std / distance_mean))
            else:
                self.optimal_threshold_factor = 2.0
        else:
            self.optimal_threshold_factor = 2.0

        print(f"Threshold.")

    def create_detector(self) -> AdaptiveOutlierDetector:
        """ Tune. """
        return AdaptiveOutlierDetector(
            input_dim=self.input_dim,
            k_max=self.optimal_k or 5,
            initial_threshold_factor=self.optimal_threshold_factor or 2.0,
            distance_metric='mahalanobis' # default.
        )


def smooth_detections(detections: List[bool], min_persistence: int = 2) -> List[bool]:
    """ Detection. """
    if len(detections) < min_persistence:
        return detections

    smoothed = [False] * len(detections)
    i = 0
    while i < len(detections):
        if detections[i]:
            consecutive_count = 0
            start_idx = i

            while i < len(detections) and detections[i]:
                consecutive_count += 1
                i += 1

            if consecutive_count >= min_persistence:
                for j in range(start_idx, start_idx + consecutive_count):
                    smoothed[j] = True
        else:
            i += 1

    return smoothed


# Compatibility.
ICLR2025ConsistentKMeans = AdaptiveOutlierDetector
OnlineKMeansChangeDetector = AdaptiveOutlierDetector