import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
from collections import deque
import librosa
from sklearn.decomposition import PCA
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Import PANN backbone from the separate file
from PANN_backbone import PANNBackbone


# ---------------- Model Initialization Utilities ----------------

def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


# ---------------- BioOSS FDTD Model with Batch-Aware State Management ----------------

class BioOSSFDTD2D(nn.Module):
    """
    BioOSS-based 2D FDTD network for audio feature processing
    Implements frequency-partitioned wave propagation based on PANN features
    Supports true batch processing with independent states for each batch item.
    """

    def __init__(self, grid_size: int = 64, dt: float = 0.01,
                 pann_dim: int = 527, freq_range: Tuple[float, float] = (50, 800),
                 boundary_condition: str = 'periodic', max_batch_size: int = 128):
        super(BioOSSFDTD2D, self).__init__()

        self.grid_size = grid_size
        self.dt = dt
        self.pann_dim = pann_dim
        self.freq_range = freq_range
        self.boundary_condition = boundary_condition
        self.max_batch_size = max_batch_size

        # Initialize physical parameters (shared across batches)
        self._initialize_physical_parameters()

        # Batch states - will be dynamically allocated
        self._batch_states = {}
        self._current_batch_size = 0

        # Memory for online processing
        self.state_history = deque(maxlen=5)

        print(f"BioOSS FDTD initialized:")
        print(f"  Grid size: {self.grid_size}x{self.grid_size}")
        print(f"  Frequency range: {self.freq_range[0]}-{self.freq_range[1]} Hz")
        print(f"  Time step: {self.dt}")
        print(f"  Using true batch processing with independent states")

    def _initialize_physical_parameters(self):
        """Initialize wave speeds and damping coefficients based on PANN index mapping"""

        # Target frequencies for each PANN dimension
        target_frequencies = np.linspace(self.freq_range[0], self.freq_range[1], self.pann_dim)

        # Store target frequencies for sine wave generation
        self.register_buffer('target_frequencies', torch.tensor(target_frequencies, dtype=torch.float32))

        # Initialize parameter tensors (shared across batches)
        c_grid = torch.ones(1, 1, self.grid_size, self.grid_size)
        kp_grid = torch.ones(1, 1, self.grid_size, self.grid_size) * 10.0
        ko_grid = torch.ones(1, 1, self.grid_size, self.grid_size) * 10.0

        # Map PANN indices to grid positions
        self.pann_to_grid_mapping = self._create_pann_to_grid_mapping()

        # Calculate wave speeds for each PANN dimension
        # target_frequencies_np = target_frequencies * self.dt
        target_frequencies_np = target_frequencies
        for pann_idx in range(self.pann_dim):
            target_freq = target_frequencies_np[pann_idx]
            grid_positions = self.pann_to_grid_mapping[pann_idx]

            # Calculate wave speed using BioOSS formula (Eq.21)
            c_value = self._calculate_wave_speed(target_freq)

            # Assign to grid positions
            for grid_x, grid_y in grid_positions:
                c_grid[0, 0, grid_x, grid_y] = c_value

        self.register_buffer('c', c_grid)
        self.register_buffer('kp', kp_grid)
        self.register_buffer('ko', ko_grid)

        print(f"  Wave speed range: {c_grid.min().item():.2f}-{c_grid.max().item():.2f}")

    def _create_pann_to_grid_mapping(self) -> dict:
        """Create mapping from PANN indices to grid positions"""

        mapping = {}
        grid_positions = []

        # Generate all grid positions
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                grid_positions.append((i, j))

        # Distribute PANN dimensions across grid
        positions_per_pann = len(grid_positions) // self.pann_dim

        for pann_idx in range(self.pann_dim):
            start_idx = pann_idx * positions_per_pann
            end_idx = min((pann_idx + 1) * positions_per_pann, len(grid_positions))

            mapping[pann_idx] = grid_positions[start_idx:end_idx]

            # Handle remaining positions
            if pann_idx == self.pann_dim - 1 and end_idx < len(grid_positions):
                mapping[pann_idx].extend(grid_positions[end_idx:])

        return mapping

    def _calculate_wave_speed(self, target_freq: float) -> float:
        """Calculate wave speed using BioOSS Eq.21"""

        # BioOSS parameters
        k_value = 10.0  # Damping coefficient

        # Spatial frequency (simplified)
        dx = 1.0
        spatial_freq_norm = np.sqrt(2) / dx  # Assuming isotropic case

        # BioOSS Eq.21: c = tan(πf·dt) * sqrt((1+dt*kp)*(1+dt*ko)) / (dt * sqrt(2) * |ξ|)
        try:
            tan_arg = np.pi * target_freq * self.dt
            sqrt_term = np.sqrt((1 + self.dt * k_value) * (1 + self.dt * k_value))

            c = np.tan(tan_arg) * sqrt_term / (self.dt * spatial_freq_norm)

            # Ensure stability
            c_max = dx / (self.dt * np.sqrt(2)) * sqrt_term
            c = min(c, c_max * 0.9)

            return max(c, 0.1)  # Minimum wave speed for numerical stability

        except:
            return 1.0  # Default wave speed

    def _allocate_batch_states(self, batch_size: int, device: torch.device):
        """Allocate or reallocate batch states"""
        if self._current_batch_size != batch_size:
            self._batch_states = {
                'p': torch.zeros(batch_size, 1, self.grid_size, self.grid_size, device=device),
                'vx': torch.zeros(batch_size, 1, self.grid_size, self.grid_size, device=device),
                'vy': torch.zeros(batch_size, 1, self.grid_size, self.grid_size, device=device),
                'time_steps': torch.zeros(batch_size, dtype=torch.long, device=device)
            }
            self._current_batch_size = batch_size

    def reset_fields(self, batch_indices: Optional[List[int]] = None):
        """Reset field variables for specific batch indices or all"""
        if not self._batch_states:
            return

        self.state_history.clear()

        if batch_indices is None:
            # Reset all batches
            for key in ['p', 'vx', 'vy']:
                self._batch_states[key].zero_()
            self._batch_states['time_steps'].zero_()
        else:
            # Reset specific batch indices
            for idx in batch_indices:
                if idx < self._current_batch_size:
                    for key in ['p', 'vx', 'vy']:
                        self._batch_states[key][idx].zero_()
                    self._batch_states['time_steps'][idx] = 0

    def forward(self, pann_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convert PANN features to spatial excitation and evolve FDTD
        Supports true batch processing with independent states.

        Args:
            pann_features: (batch_size, 527) PANN clipwise output

        Returns:
            p_field: (batch_size, grid_size, grid_size) evolved pressure fields, or a single field when batch_size=1
        """
        batch_size = pann_features.shape[0]
        device = pann_features.device

        # Allocate batch states if needed
        self._allocate_batch_states(batch_size, device)

        # Store current state for history (only for first batch item for compatibility)
        if batch_size > 0:
            current_state = {
                'p': self._batch_states['p'][0:1].clone() if batch_size == 1 else self._batch_states['p'][0:1].clone(),
                'vx': self._batch_states['vx'][0:1].clone() if batch_size == 1 else self._batch_states['vx'][
                                                                                    0:1].clone(),
                'vy': self._batch_states['vy'][0:1].clone() if batch_size == 1 else self._batch_states['vy'][
                                                                                    0:1].clone()
            }
            self.state_history.append(current_state)

        # Convert PANN features to spatial excitation for all batches
        spatial_excitation = self._pann_to_spatial_excitation_batch(pann_features)

        # Evolve FDTD system for all batches simultaneously
        self._fdtd_step_batch(spatial_excitation, batch_size)

        # Increment time steps for all batches
        self._batch_states['time_steps'] += 1

        # Return pressure fields - maintain compatibility with single batch interface
        if batch_size == 1:
            return self._batch_states['p'].squeeze()  # Remove all extra dimensions for single batch
        else:
            return self._batch_states['p'].squeeze(1)  # Remove channel dimension only

    def _pann_to_spatial_excitation_batch(self, pann_features: torch.Tensor) -> torch.Tensor:
        """Convert PANN features to spatial excitation pattern using frequency-specific sine waves for batches"""
        batch_size = pann_features.shape[0]
        device = pann_features.device

        excitation = torch.zeros(batch_size, 1, self.grid_size, self.grid_size, device=device)

        # Get current time for each batch (vectorized)
        current_times = self._batch_states['time_steps'].float() * self.dt + 1e-5  # (batch_size,)

        # Vectorized sine wave generation for all frequencies and batches
        # target_frequencies: (527,), current_times: (batch_size,)
        # Broadcasting: (batch_size, 1) * (1, 527) -> (batch_size, 527)
        freq_time_product = current_times.unsqueeze(1) * self.target_frequencies.unsqueeze(0)
        sine_amplitudes = torch.sin(2 * np.pi * freq_time_product)  # (batch_size, 527)

        # Scale by PANN probabilities
        scaled_excitations = pann_features * sine_amplitudes  # (batch_size, 527)

        # Apply to grid positions (this part still needs to be done per PANN dimension)
        for pann_idx in range(self.pann_dim):
            grid_positions = self.pann_to_grid_mapping[pann_idx]
            excitation_values = scaled_excitations[:, pann_idx]  # (batch_size,)

            for grid_x, grid_y in grid_positions:
                excitation[:, 0, grid_x, grid_y] += excitation_values

        return excitation

    def _fdtd_step_batch(self, source: Optional[torch.Tensor], batch_size: int):
        """Single FDTD time step for all batches simultaneously"""

        # Get current batch states
        p = self._batch_states['p']  # (batch_size, 1, H, W)
        vx = self._batch_states['vx']  # (batch_size, 1, H, W)
        vy = self._batch_states['vy']  # (batch_size, 1, H, W)

        # Expand shared parameters to match batch size
        c = self.c.expand(batch_size, -1, -1, -1)  # (batch_size, 1, H, W)
        kp = self.kp.expand(batch_size, -1, -1, -1)  # (batch_size, 1, H, W)
        ko = self.ko.expand(batch_size, -1, -1, -1)  # (batch_size, 1, H, W)

        # Calculate velocity divergence
        if self.boundary_condition == 'periodic':
            vx_shifted = torch.roll(vx, -1, dims=2)
            dvx_dx = vx_shifted - vx

            vy_shifted = torch.roll(vy, -1, dims=3)
            dvy_dy = vy_shifted - vy
        else:
            dvx_dx = F.pad(vx[:, :, 1:, :] - vx[:, :, :-1, :], (0, 0, 0, 1))
            dvy_dy = F.pad(vy[:, :, :, 1:] - vy[:, :, :, :-1], (0, 1, 0, 0))

        # Update pressure field for all batches
        new_p = ((1 - self.dt * kp) * p - c ** 2 * self.dt * (dvx_dx + dvy_dy))

        # Add source if provided
        if source is not None:
            new_p = new_p + source

        # Calculate pressure gradients
        if self.boundary_condition == 'periodic':
            p_shifted_x = torch.roll(new_p, 1, dims=2)
            dp_dx = new_p - p_shifted_x

            p_shifted_y = torch.roll(new_p, 1, dims=3)
            dp_dy = new_p - p_shifted_y
        else:
            dp_dx = F.pad(new_p[:, :, :-1, :] - new_p[:, :, 1:, :], (0, 0, 1, 0))
            dp_dy = F.pad(new_p[:, :, :, :-1] - new_p[:, :, :, 1:], (0, 1, 0, 0))

        # Update velocity fields for all batches
        new_vx = (1 - self.dt * ko) * vx - self.dt * dp_dx
        new_vy = (1 - self.dt * ko) * vy - self.dt * dp_dy

        # Store updated states
        self._batch_states['p'] = new_p
        self._batch_states['vx'] = new_vx
        self._batch_states['vy'] = new_vy

    def get_system_energy(self) -> float:
        """Calculate total system energy for first batch item (compatibility)"""
        if not self._batch_states:
            return 0.0

        p = self._batch_states['p'][0:1]  # First batch item
        vx = self._batch_states['vx'][0:1]
        vy = self._batch_states['vy'][0:1]

        kinetic_energy = 0.5 * (torch.sum(vx ** 2) + torch.sum(vy ** 2))
        potential_energy = 0.5 * torch.sum(p ** 2)

        return (kinetic_energy + potential_energy).item()

    def get_system_energy_batch(self) -> torch.Tensor:
        """Calculate total system energy for each batch item"""
        if not self._batch_states:
            return torch.tensor([])

        p = self._batch_states['p']
        vx = self._batch_states['vx']
        vy = self._batch_states['vy']

        # Calculate energy for each batch item
        kinetic_energy = 0.5 * (torch.sum(vx ** 2, dim=(1, 2, 3)) + torch.sum(vy ** 2, dim=(1, 2, 3)))
        potential_energy = 0.5 * torch.sum(p ** 2, dim=(1, 2, 3))

        return kinetic_energy + potential_energy  # (batch_size,)

    def get_system_state_vector(self) -> torch.Tensor:
        """Get flattened system state for analysis for first batch item (compatibility)"""
        if not self._batch_states:
            return torch.tensor([])

        p = self._batch_states['p'][0]  # First batch item, remove batch dimension
        vx = self._batch_states['vx'][0]
        vy = self._batch_states['vy'][0]

        return torch.cat([
            p.flatten(),
            vx.flatten(),
            vy.flatten()
        ])

    def get_system_state_vector_batch(self) -> torch.Tensor:
        """Get flattened system state for analysis for each batch item"""
        if not self._batch_states:
            return torch.tensor([])

        p = self._batch_states['p']
        vx = self._batch_states['vx']
        vy = self._batch_states['vy']

        # Flatten each batch item independently
        batch_size = p.shape[0]
        state_vectors = []

        for i in range(batch_size):
            state_vector = torch.cat([
                p[i].flatten(),
                vx[i].flatten(),
                vy[i].flatten()
            ])
            state_vectors.append(state_vector)

        return torch.stack(state_vectors)  # (batch_size, flattened_state_size)


# ---------------- Detection Utility Classes ----------------

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

    def __init__(self, window_size: int = 20, alpha: float = 0.2):
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

        if len(self.value_buffer) < 5:
            # Insufficient data, use simple threshold
            values = self.value_buffer.to_array()
            mean_val = np.mean(values) if len(values) > 0 else 0
            std_val = np.std(values) if len(values) > 1 else 0.1
            return mean_val + 1.5 * std_val

        # Calculate baselines statistics
        history = self.value_buffer.to_array()
        baseline_mean = np.mean(history)
        baseline_std = np.std(history)

        # Trend adjustment
        trend_factor = self._compute_trend_factor(history)

        # Adaptive threshold with trend adjustment
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
                 consensus_threshold: float = 0.3,
                 min_persistence_duration: int = 3,
                 min_confidence: float = 0.3):

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

        # Adaptive thresholds
        self.adaptive_thresholds = {
            'energy': OnlineAdaptiveThreshold(window_size=30, alpha=0.2),
            'pca': OnlineAdaptiveThreshold(window_size=30, alpha=0.2),
            'cosine': OnlineAdaptiveThreshold(window_size=30, alpha=0.2)
        }

        # Persistence filtering
        self.detection_buffer = CircularBuffer(max_size=10)
        self.min_confidence = min_confidence

        # State tracking
        self.last_detection_time = -1
        self.cooldown_period = 3

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

        # Persistence validation
        is_change_detected = False
        persistence_ratio = 0.0

        if len(self.detection_buffer) >= self.min_persistence_duration:
            recent_detections = list(self.detection_buffer)[-self.min_persistence_duration:]
            persistence_ratio = sum(recent_detections) / len(recent_detections)

            # Check cooldown period
            time_since_last = timestamp - self.last_detection_time
            is_not_in_cooldown = time_since_last > self.cooldown_period

            # if persistence_ratio >= 0.5 and is_not_in_cooldown:
            #     is_change_detected = True
            #     self.last_detection_time = timestamp
            #     self._reset_detection_state()

            if (persistence_ratio >= 0.5 and
                    is_not_in_cooldown and
                    consensus_score > self.min_confidence):  # Use the configured confidence threshold
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

    # Preserve original detection points without diffusion
    smoothed = detections.copy()
    return smoothed


# ---------------- Batch-Aware Processor ----------------

class BatchOnlineBioOSSProcessor:
    """Batch-aware online processor for BioOSS FDTD system with change detection"""

    def __init__(self, biooss_model: BioOSSFDTD2D, memory_length: int = 10):
        self.biooss_model = biooss_model
        self.memory_length = memory_length

        # Batch-aware state history (dict of batch_id -> deque)
        self.batch_state_histories: Dict[str, deque] = {}
        self.batch_energy_histories: Dict[str, deque] = {}

    def process_timestep_batch(self, pann_features: torch.Tensor, batch_ids: Optional[List[str]] = None) -> Dict:
        """
        Process batch of timesteps and return system state and metrics

        Args:
            pann_features: (batch_size, 527) PANN features for current timestep
            batch_ids: Optional list of batch identifiers for state tracking

        Returns:
            result: Dictionary containing batched system state and metrics
        """
        batch_size = pann_features.shape[0]

        # Generate default batch IDs if not provided
        if batch_ids is None:
            batch_ids = [f"batch_{i}" for i in range(batch_size)]

        # Process through BioOSS
        pressure_fields = self.biooss_model(pann_features)  # (batch_size, H, W) or (H, W) for single

        # Calculate metrics for each batch item
        if batch_size == 1:
            # Single batch compatibility
            energies = torch.tensor([self.biooss_model.get_system_energy()])
            state_vectors = self.biooss_model.get_system_state_vector().unsqueeze(0)
        else:
            energies = self.biooss_model.get_system_energy_batch()  # (batch_size,)
            state_vectors = self.biooss_model.get_system_state_vector_batch()  # (batch_size, state_dim)

        # Store in history for each batch
        results = []
        for i, batch_id in enumerate(batch_ids):
            # Initialize history for new batch IDs
            if batch_id not in self.batch_state_histories:
                self.batch_state_histories[batch_id] = deque(maxlen=self.memory_length)
                self.batch_energy_histories[batch_id] = deque(maxlen=self.memory_length)

            # Store current state
            self.batch_state_histories[batch_id].append(state_vectors[i].clone())
            self.batch_energy_histories[batch_id].append(energies[i].item())

            # Prepare result for this batch item
            if batch_size == 1:
                result = {
                    'pressure_field': pressure_fields,  # Already squeezed for single batch
                    'energy': energies[i].item(),
                    'state_vector': state_vectors[i],
                    'timestamp': len(self.batch_energy_histories[batch_id]) - 1,
                    'batch_id': batch_id
                }
            else:
                result = {
                    'pressure_field': pressure_fields[i],
                    'energy': energies[i].item(),
                    'state_vector': state_vectors[i],
                    'timestamp': len(self.batch_energy_histories[batch_id]) - 1,
                    'batch_id': batch_id
                }
            results.append(result)

        return {
            'batch_results': results,
            'batch_size': batch_size
        }

    def process_single_timestep(self, pann_features: torch.Tensor) -> Dict:
        """
        Process single timestep (for compatibility with existing code)

        Args:
            pann_features: (1, 527) PANN features for current timestep

        Returns:
            result: Dictionary containing system state and metrics
        """
        batch_result = self.process_timestep_batch(pann_features, ["single"])
        return batch_result['batch_results'][0]

    def reset_batch(self, batch_ids: Optional[List[str]] = None):
        """Reset processor state for specific batches or all"""
        if batch_ids is None:
            # Reset all batches
            self.batch_state_histories.clear()
            self.batch_energy_histories.clear()
            self.biooss_model.reset_fields()
        else:
            # Reset specific batches
            batch_indices = []
            for batch_id in batch_ids:
                if batch_id in self.batch_state_histories:
                    del self.batch_state_histories[batch_id]
                if batch_id in self.batch_energy_histories:
                    del self.batch_energy_histories[batch_id]

                # Try to extract batch index for field reset
                try:
                    batch_idx = int(batch_id.split('_')[-1])
                    batch_indices.append(batch_idx)
                except:
                    pass

            if batch_indices:
                self.biooss_model.reset_fields(batch_indices)

    def reset(self):
        """Reset all processor state (for compatibility)"""
        self.reset_batch()


# ---------------- Audio Encoder ----------------

class AudioEncoder(nn.Module):
    def __init__(self, args):
        super(AudioEncoder, self).__init__()
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

        # Use the updated PANNBackbone
        self.model = PANNBackbone(
            sample_rate=args.sample_rate,
            window_size=args.window_size,
            hop_size=args.hop_size,
            mel_bins=args.mel_bins,
            fmin=args.fmin,
            fmax=args.fmax,
            classes_num=args.output_size
        )

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        print(f"AudioEncoder initialized with device: {self.device}")
        print(f"Model output dimension: {args.output_size}")

    def forward(self, audio_path):
        """Extract PANN clipwise features from audio file"""
        try:
            # Load audio
            waveform, _ = librosa.load(audio_path, sr=self.model.sample_rate, mono=True)
            waveform = torch.tensor(waveform).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Get PANN output
                output = self.model(waveform)
                # Return clipwise_output which is already 527-dimensional and sigmoid-activated
                clipwise_features = output['clipwise_output'].cpu()

                return clipwise_features

        except Exception as e:
            print(f"Error in AudioEncoder.forward: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# ---------------- BioOSS Pattern Change Detector with Compatible Interface ----------------

class BioOSSPatternChangeDetector:
    def __init__(self, args,
                 energy_weight=0.4, pca_weight=0.35, cosine_weight=0.25,
                 consensus_threshold=0.3, min_persistence=2, min_confidence_threshold=0.3):
        self.audio_encoder = AudioEncoder(args)

        # Initialize BioOSS FDTD model with correct PANN dimension
        self.biooss_model = BioOSSFDTD2D(
            grid_size=64, dt=0.01,
            pann_dim=args.output_size,  # This should be 527
            freq_range=(args.freq_min, args.freq_max),
            boundary_condition='periodic'
        )
        self.biooss_model.to(self.audio_encoder.device)

        # Initialize multi-metric change detector with configurable parameters
        self.change_detector = OnlineMultiMetricChangeDetector(
            energy_weight=energy_weight,
            pca_weight=pca_weight,
            cosine_weight=cosine_weight,
            consensus_threshold=consensus_threshold,
            min_persistence_duration=min_persistence,
            min_confidence=min_confidence_threshold
        )

        # Initialize the batch-aware processor
        self.processor = BatchOnlineBioOSSProcessor(self.biooss_model)

        self.prev_energy = None
        self.running = True
        self.timestep = 0

        print(f"BioOSS Pattern Detector initialized:")
        print(f"  PANN dimension: {args.output_size}")
        print(f"  Grid size: 64x64")
        print(f"  Device: {self.audio_encoder.device}")
        print(f"  Multi-metric weights: Energy={energy_weight}, PCA={pca_weight}, Cosine={cosine_weight}")
        print(f"  Consensus threshold: {consensus_threshold}")
        print(f"  Using true batch processing with independent states")

    def detect_pattern_change(self, audio_path):
        """Detects pattern change using BioOSS multi-metric dynamics"""
        try:
            # Extract PANN clipwise features (527-dimensional)
            pann_features = self.audio_encoder(audio_path)

            if pann_features is None:
                print(f"Warning: No PANN features extracted for {audio_path}. Skipping.")
                return 0, None

            # Ensure correct shape for BioOSS
            if pann_features.dim() == 1:
                pann_features = pann_features.unsqueeze(0)  # Add batch dimension

            # Verify dimensions match expectation
            expected_dim = self.biooss_model.pann_dim
            actual_dim = pann_features.shape[1]

            if actual_dim != expected_dim:
                print(f"Warning: PANN feature dimension mismatch. Expected {expected_dim}, got {actual_dim}")
                return 0, None

            # Move features to correct device
            pann_features = pann_features.to(self.audio_encoder.device)

            # Process through BioOSS using the processor
            biooss_result = self.processor.process_single_timestep(pann_features)

            # Extract energy and state vector from the result
            energy = biooss_result['energy']
            state_vector = biooss_result['state_vector']

            # Use multi-metric change detector to determine if pattern changed
            is_change, confidence, detailed_metrics = self.change_detector.update(
                energy=energy,
                state_vector=state_vector,
                timestamp=self.timestep
            )

            self.timestep += 1

            # Calculate similarity score for compatibility with original interface
            if self.prev_energy is not None:
                energy_change_ratio = abs(energy - self.prev_energy) / (self.prev_energy + 1e-8)
                # Convert to similarity score (lower means more different)
                similarity = 1.0 / (1.0 + energy_change_ratio)
            else:
                similarity = 1.0

            self.prev_energy = energy

            # # More detailed logging
            # print(f"PANN Features Shape: {pann_features.shape}, BioOSS Energy: {energy:.4f}")
            # print(f"Change Detected: {is_change}, Confidence: {confidence:.4f}, Similarity: {similarity:.4f}")
            # if is_change:
            #     print(f"  Detailed metrics: {detailed_metrics}")

            return similarity, {
                'energy': energy,
                'change_detected': is_change,
                'confidence': confidence,
                'detailed_metrics': detailed_metrics,
                'state_vector': state_vector
            }

        except Exception as e:
            print(f"Error in BioOSS pattern detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0, None

    def reset_fields(self):
        """Reset BioOSS fields and detector state"""
        self.biooss_model.reset_fields()
        self.change_detector.reset()
        self.processor.reset()
        self.prev_energy = None
        self.timestep = 0

    def stop(self):
        self.running = False
