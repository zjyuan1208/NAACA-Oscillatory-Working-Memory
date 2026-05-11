import torch
import numpy as np
from collections import deque
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional
import scipy.signal as signal


class CircularBuffer:
    """Circular buffer for efficient memory management"""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def append(self, item):
        self.buffer.append(item)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

    @property
    def size(self):
        return len(self.buffer)

    def to_array(self):
        """Convert buffer to numpy array"""
        return np.array(list(self.buffer))


class OnlineEnergyCalculator:
    """Online calculator for BioOSS energy dynamics"""

    def __init__(self, memory_length: int = 5):
        self.energy_history = CircularBuffer(max_size=memory_length)
        self.gradient_history = CircularBuffer(max_size=memory_length - 1)

    def update(self, energy: float) -> float:
        """
        Update energy calculator and return energy acceleration

        Args:
            energy: Current system energy

        Returns:
            energy_acceleration: Second derivative of energy
        """
        self.energy_history.append(energy)

        if len(self.energy_history) >= 2:
            # Calculate energy gradient
            energy_gradient = abs(self.energy_history[-1] - self.energy_history[-2])
            self.gradient_history.append(energy_gradient)

            if len(self.gradient_history) >= 2:
                # Calculate energy acceleration (second derivative)
                energy_acceleration = abs(self.gradient_history[-1] - self.gradient_history[-2])
                return energy_acceleration

        return 0.0

    def reset(self):
        """Reset calculator state"""
        self.energy_history.clear()
        self.gradient_history.clear()


class OnlinePCACalculator:
    """Online calculator for PCA direction changes"""

    def __init__(self, window_size: int = 5, n_components: int = 3):
        self.window_size = window_size
        self.n_components = n_components
        self.state_buffer = CircularBuffer(max_size=window_size * 2)

    def update(self, state_vector: torch.Tensor) -> float:
        """
        Update PCA calculator and return subspace angle change

        Args:
            state_vector: Flattened system state vector

        Returns:
            subspace_angle: Angle between previous and current principal subspaces
        """
        # Convert to numpy and flatten
        if torch.is_tensor(state_vector):
            state_vector = state_vector.detach().cpu().numpy()

        state_vector = state_vector[: state_vector.shape[0] // 3].flatten()
        self.state_buffer.append(state_vector)

        if len(self.state_buffer) >= self.window_size * 2:
            # Split into previous and current windows
            all_states = self.state_buffer.to_array()
            prev_window = all_states[:self.window_size]
            curr_window = all_states[self.window_size:]

            # Check for sufficient variation
            if self._has_sufficient_variation(prev_window) and self._has_sufficient_variation(curr_window):
                try:
                    # Compute PCA for both windows
                    n_comp = min(self.n_components, self.window_size, prev_window.shape[1])
                    prev_pca = PCA(n_components=n_comp).fit(prev_window)
                    curr_pca = PCA(n_components=n_comp).fit(curr_window)

                    # Calculate principal angles between subspaces
                    subspace_angle = self._compute_principal_angles(
                        prev_pca.components_,
                        curr_pca.components_
                    )

                    return subspace_angle
                except:
                    return 0.0

        return 0.0

    def _has_sufficient_variation(self, data_matrix: np.ndarray) -> bool:
        """Check if data has sufficient variation for PCA"""
        return np.std(data_matrix) > 1e-6

    def _compute_principal_angles(self, A: np.ndarray, B: np.ndarray) -> float:
        """Compute principal angles between two subspaces"""
        try:
            # Compute SVD of A @ B.T
            U, s, Vt = np.linalg.svd(A @ B.T)
            # Principal angle is arccos of smallest singular value
            principal_angle = np.arccos(np.clip(s.min(), 0, 1))
            return principal_angle
        except:
            return 0.0

    def reset(self):
        """Reset calculator state"""
        self.state_buffer.clear()


class OnlineCosineCalculator:
    """Online calculator for cosine distance changes"""

    def __init__(self):
        self.prev_state = None

    def update(self, state_vector: torch.Tensor) -> float:
        """
        Update cosine calculator and return cosine distance

        Args:
            state_vector: Current system state vector

        Returns:
            cosine_distance: Cosine distance from previous state
        """
        # Convert to numpy
        if torch.is_tensor(state_vector):
            current_vector = state_vector[: state_vector.shape[0] // 3].detach().cpu().numpy().flatten()
        else:
            current_vector = state_vector[: state_vector.shape[0] // 3].flatten()

        if self.prev_state is not None:
            # Normalize vectors
            curr_norm = current_vector / (np.linalg.norm(current_vector) + 1e-8)
            prev_norm = self.prev_state / (np.linalg.norm(self.prev_state) + 1e-8)

            # Calculate cosine distance
            cosine_dist = 1 - np.dot(curr_norm, prev_norm)

            # Update previous state
            self.prev_state = current_vector.copy()

            return max(0.0, cosine_dist)  # Ensure non-negative
        else:
            # First call, initialize
            self.prev_state = current_vector.copy()
            return 0.0

    def reset(self):
        """Reset calculator state"""
        self.prev_state = None


class OnlineAdaptiveThreshold:
    """Online adaptive threshold calculator"""

    def __init__(self, window_size: int = 20, alpha: float = 0.2): # More permissive.
        self.window_size = window_size
        self.alpha = alpha  # Trend adjustment factor
        self.value_buffer = CircularBuffer(max_size=window_size)

    def update(self, new_value: float) -> float:
        """
        Update adaptive threshold

        Args:
            new_value: New metric value

        Returns:
            threshold: Adaptive threshold value
        """
        self.value_buffer.append(new_value)

        if len(self.value_buffer) < 5: # Reduce.
            # Insufficient data, use simple threshold
            values = self.value_buffer.to_array()
            mean_val = np.mean(values) if len(values) > 0 else 0
            std_val = np.std(values) if len(values) > 1 else 0.1
            return mean_val + 1.5 * std_val # Lower multiplier.

        # Calculate baselines statistics
        history = self.value_buffer.to_array()
        baseline_mean = np.mean(history)
        baseline_std = np.std(history)

        # Trend adjustment
        trend_factor = self._compute_trend_factor(history)

        # More permissive.
        adaptive_threshold = baseline_mean + 2 * baseline_std * (1 + self.alpha * trend_factor)

        return adaptive_threshold

    def _compute_trend_factor(self, sequence: np.ndarray) -> float:
        """Compute trend factor for threshold adjustment"""
        if len(sequence) < 3:
            return 0.0

        try:
            # Simple linear trend
            x = np.arange(len(sequence))
            slope, _ = np.polyfit(x, sequence, 1)
            trend_strength = abs(slope) / (np.std(sequence) + 1e-8)
            return trend_strength
        except:
            return 0.0


class OnlineMultiMetricChangeDetector:
    """Multi-metric fusion change detector"""

    def __init__(self,
                 energy_weight: float = 0.4,
                 pca_weight: float = 0.35,
                 cosine_weight: float = 0.25,
                 consensus_threshold: float = 0.3, # Lower default threshold.
                 min_persistence_duration: int = 3):

        # Metric weights
        self.metric_weights = {
            'energy': energy_weight,
            'pca': pca_weight,
            'cosine': cosine_weight
        }

        self.consensus_threshold = consensus_threshold
        self.min_persistence_duration = min_persistence_duration

        # Online calculators
        self.energy_calculator = OnlineEnergyCalculator()
        self.pca_calculator = OnlinePCACalculator()
        self.cosine_calculator = OnlineCosineCalculator()

        # Adaptive thresholds with more relaxed parameters
        self.adaptive_thresholds = {
            'energy': OnlineAdaptiveThreshold(window_size=20, alpha=0.2), # Smaller window.
            'pca': OnlineAdaptiveThreshold(window_size=20, alpha=0.2),
            'cosine': OnlineAdaptiveThreshold(window_size=20, alpha=0.2)
        }

        # Persistence filtering
        self.detection_buffer = CircularBuffer(max_size=10)

        # State tracking
        self.last_detection_time = -1
        self.cooldown_period = 3 # Reduce.

        # print(f"Change detector initialized:")
        # print(f"  Consensus threshold: {consensus_threshold}")
        # print(f"  Min persistence: {min_persistence_duration}")
        # print(f"  Cooldown period: {self.cooldown_period}")

    def update(self, energy: float, state_vector: torch.Tensor, timestamp: int) -> Tuple[bool, float, Dict]:
        """
        Update detector with new measurements

        Args:
            energy: System energy
            state_vector: System state vector
            timestamp: Current timestamp

        Returns:
            is_change_detected: Whether significant change is detected
            confidence: Detection confidence score
            metrics: Dictionary of individual metric values
        """
        # Calculate individual metrics
        energy_metric = self.energy_calculator.update(energy)
        pca_metric = self.pca_calculator.update(state_vector)
        cosine_metric = self.cosine_calculator.update(state_vector)

        current_metrics = {
            'energy': energy_metric,
            'pca': pca_metric,
            'cosine': cosine_metric
        }

        # Update adaptive thresholds
        thresholds = {}
        for metric_name, value in current_metrics.items():
            thresholds[metric_name] = self.adaptive_thresholds[metric_name].update(value)

        # Single metric detection
        detections = {}
        for metric_name, value in current_metrics.items():
            detections[metric_name] = value > thresholds[metric_name]

        # Weighted fusion
        consensus_score = sum(
            self.metric_weights[metric_name] * detections[metric_name]
            for metric_name in current_metrics.keys()
        )

        # Initial detection decision
        is_candidate = consensus_score > self.consensus_threshold
        self.detection_buffer.append(is_candidate)

        # More permissive.
        is_change_detected = False
        persistence_ratio = 0.0

        if len(self.detection_buffer) >= self.min_persistence_duration:
            recent_detections = list(self.detection_buffer)[-self.min_persistence_duration:]
            persistence_ratio = sum(recent_detections) / len(recent_detections)

            # Check cooldown period
            time_since_last = timestamp - self.last_detection_time
            is_not_in_cooldown = time_since_last > self.cooldown_period

            # More permissive.
            if persistence_ratio >= 0.5 and is_not_in_cooldown:
                is_change_detected = True
                self.last_detection_time = timestamp
                self._reset_detection_state()

        # Prepare detailed results
        detailed_metrics = {
            'raw_metrics': current_metrics,
            'thresholds': thresholds,
            'detections': detections,
            'consensus_score': consensus_score,
            'persistence_ratio': persistence_ratio
        }

        return is_change_detected, consensus_score, detailed_metrics

    def _reset_detection_state(self):
        """Reset detection state to avoid repeated detections"""
        self.detection_buffer.clear()

    def reset(self):
        """Reset entire detector state"""
        self.energy_calculator.reset()
        self.pca_calculator.reset()
        self.cosine_calculator.reset()
        self.detection_buffer.clear()
        self.last_detection_time = -1


def generate_interpretation(detection_result: Dict, timestamp: float) -> Dict:
    """
    Generate physical interpretation of detected change

    Args:
        detection_result: Detection result from multi-metric detector
        timestamp: Timestamp of detection

    Returns:
        interpretation: Dictionary containing interpretation
    """
    interpretation = {
        'timestamp': timestamp,
        'detection_confidence': detection_result['consensus_score'],
        'contributing_factors': [],
        'physical_interpretation': {},
        'change_characteristics': {}
    }

    raw_metrics = detection_result['raw_metrics']
    thresholds = detection_result['thresholds']

    # Analyze contributing factors
    for metric_name, value in raw_metrics.items():
        if value > thresholds[metric_name]:
            contribution_strength = value / thresholds[metric_name]
            interpretation['contributing_factors'].append({
                'metric': metric_name,
                'strength': contribution_strength,
                'raw_value': value,
                'threshold': thresholds[metric_name]
            })

    # Physical interpretation based on dominant metrics
    if raw_metrics['energy'] > thresholds['energy']:
        interpretation['physical_interpretation']['energy_dynamics'] = {
            'type': 'Energy jump detected',
            'magnitude': raw_metrics['energy'],
            'description': 'Rapid change in system energy indicating acoustic event transition'
        }

    if raw_metrics['pca'] > thresholds['pca']:
        interpretation['physical_interpretation']['structural_change'] = {
            'type': 'Subspace rotation detected',
            'angle': raw_metrics['pca'],
            'description': 'Change in feature correlation structure indicating new audio content'
        }

    if raw_metrics['cosine'] > thresholds['cosine']:
        interpretation['physical_interpretation']['semantic_shift'] = {
            'type': 'Feature vector rotation detected',
            'distance': raw_metrics['cosine'],
            'description': 'Change in feature vector direction indicating category shift'
        }

    return interpretation


def smooth_detections(detections: List[bool], window_size: int = 3) -> List[bool]:
    """
    Apply temporal smoothing to detection results - preserve original detection points

    Args:
        detections: List of detection results
        window_size: Smoothing window size

    Returns:
        smoothed: Smoothed detection results
    """
    if len(detections) < window_size:
        return detections

    # New strategy.
    # preserve detection precision.
    smoothed = detections.copy() # Return the raw detections directly.

    # Optional denoising.
    # may be noise.
    for i in range(1, len(detections) - 1):
        if detections[i] and not detections[i - 1] and not detections[i + 1]:
            # Check.
            start_idx = max(0, i - window_size)
            end_idx = min(len(detections), i + window_size + 1)
            window_detections = detections[start_idx:end_idx]

            # may be noise.
            # meaningful.
            pass # Detection.

    return smoothed