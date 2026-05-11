import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
from collections import deque


class BioOSSFDTD2D(nn.Module):
    """
    BioOSS-based 2D FDTD network for audio feature processing
    Implements frequency-partitioned wave propagation based on PANN features
    Now supports true batch processing with independent states per batch
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

    def _initialize_physical_parameters(self):
        """Initialize wave speeds and damping coefficients based on PANN index mapping"""

        # Target frequencies for each PANN dimension
        target_frequencies = np.linspace(self.freq_range[0], self.freq_range[1], self.pann_dim)
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

            # Calculate wave speed using BioOSS formula
            c_value = self._calculate_wave_speed(target_freq)

            # Assign to grid positions
            for grid_x, grid_y in grid_positions:
                c_grid[0, 0, grid_x, grid_y] = c_value

        self.register_buffer('c', c_grid)
        self.register_buffer('kp', kp_grid)
        self.register_buffer('ko', ko_grid)

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
        k_value = 10.0  # Damping coefficient
        dx = 1.0
        spatial_freq_norm = np.sqrt(2) / dx

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
        Now supports true batch processing with independent states

        Args:
            pann_features: (batch_size, 527) PANN clipwise output

        Returns:
            p_field: (batch_size, grid_size, grid_size) evolved pressure fields
        """
        batch_size = pann_features.shape[0]
        device = pann_features.device

        # Allocate batch states if needed
        self._allocate_batch_states(batch_size, device)

        # Convert PANN features to spatial excitation for all batches
        spatial_excitation = self._pann_to_spatial_excitation_batch(pann_features)

        # Evolve FDTD system for all batches simultaneously
        self._fdtd_step_batch(spatial_excitation, batch_size)

        # Increment time steps for all batches
        self._batch_states['time_steps'] += 1

        # Return pressure fields for all batches
        return self._batch_states['p'].squeeze(1)  # Remove channel dimension

    def _pann_to_spatial_excitation_batch(self, pann_features: torch.Tensor) -> torch.Tensor:
        """Convert PANN features to spatial excitation pattern using frequency-specific sine waves for batches"""
        batch_size = pann_features.shape[0]
        device = pann_features.device

        excitation = torch.zeros(batch_size, 1, self.grid_size, self.grid_size, device=device)

        # Get current time for each batch (vectorized)
        current_times = self._batch_states['time_steps'].float() * self.dt + 1e-5  # (batch_size,)
        # current_times = self._batch_states['time_steps'].float()  # (batch_size,)

        # Vectorized sine wave generation for all frequencies and batches
        # target_frequencies: (527,), current_times: (batch_size,)
        # Broadcasting: (batch_size, 1) * (1, 527) -> (batch_size, 527)
        freq_time_product = current_times.unsqueeze(1) * self.target_frequencies.unsqueeze(0)
        # freq_time_product = current_times.unsqueeze(1) * 5
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
        pressure_fields = self.biooss_model(pann_features)  # (batch_size, H, W)

        # Calculate metrics for each batch item
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


# Compatibility alias for existing code
OnlineBioOSSProcessor = BatchOnlineBioOSSProcessor


# class BioOSSFDTD2D(nn.Module):
#     """
#     BioOSS-based 2D FDTD network for audio feature processing
#     Implements frequency-partitioned wave propagation based on PANN features
#     """
#
#     def __init__(self, grid_size: int = 64, dt: float = 0.01,
#                  pann_dim: int = 527, freq_range: Tuple[float, float] = (50, 800),
#                  boundary_condition: str = 'periodic'):
#         super(BioOSSFDTD2D, self).__init__()
#
#         self.grid_size = grid_size
#         self.dt = dt
#         self.pann_dim = pann_dim
#         self.freq_range = freq_range
#         self.boundary_condition = boundary_condition
#
#         # Initialize pressure and velocity fields
#         self.register_buffer('p', torch.zeros(1, 1, grid_size, grid_size))
#         self.register_buffer('vx', torch.zeros(1, 1, grid_size, grid_size))
#         self.register_buffer('vy', torch.zeros(1, 1, grid_size, grid_size))
#
#         # Initialize physical parameters based on frequency mapping
#         self._initialize_physical_parameters()
#
#         # Memory for online processing
#         self.state_history = deque(maxlen=5)
#
#         # Time counter for sine wave generation
#         self.time_step = 0
#
#     def _initialize_physical_parameters(self):
#         """Initialize wave speeds and damping coefficients based on PANN index mapping"""
#
#         # Target frequencies for each PANN dimension
#         target_frequencies = np.linspace(self.freq_range[0], self.freq_range[1], self.pann_dim)
#
#         # Store target frequencies for sine wave generation
#         self.register_buffer('target_frequencies', torch.tensor(target_frequencies, dtype=torch.float32))
#
#         # Initialize parameter tensors
#         c_grid = torch.ones(1, 1, self.grid_size, self.grid_size)
#         kp_grid = torch.ones(1, 1, self.grid_size, self.grid_size) * 10.0
#         ko_grid = torch.ones(1, 1, self.grid_size, self.grid_size) * 10.0
#
#         # Map PANN indices to grid positions
#         self.pann_to_grid_mapping = self._create_pann_to_grid_mapping()
#
#         # Calculate wave speeds for each PANN dimension
#         for pann_idx in range(self.pann_dim):
#             target_freq = target_frequencies[pann_idx]
#             grid_positions = self.pann_to_grid_mapping[pann_idx]
#
#             # Calculate wave speed using BioOSS formula (Eq.21)
#             c_value = self._calculate_wave_speed(target_freq)
#
#             # Assign to grid positions
#             for grid_x, grid_y in grid_positions:
#                 c_grid[0, 0, grid_x, grid_y] = c_value
#
#         self.register_buffer('c', c_grid)
#         self.register_buffer('kp', kp_grid)
#         self.register_buffer('ko', ko_grid)
#
#         print(f"BioOSS FDTD initialized:")
#         print(f"  Grid size: {self.grid_size}x{self.grid_size}")
#         print(f"  Frequency range: {self.freq_range[0]}-{self.freq_range[1]} Hz")
#         print(f"  Time step: {self.dt}")
#         print(f"  Wave speed range: {c_grid.min().item():.2f}-{c_grid.max().item():.2f}")
#
#     def _create_pann_to_grid_mapping(self) -> dict:
#         """Create mapping from PANN indices to grid positions"""
#
#         mapping = {}
#         grid_positions = []
#
#         # Generate all grid positions
#         for i in range(self.grid_size):
#             for j in range(self.grid_size):
#                 grid_positions.append((i, j))
#
#         # Distribute PANN dimensions across grid
#         positions_per_pann = len(grid_positions) // self.pann_dim
#
#         for pann_idx in range(self.pann_dim):
#             start_idx = pann_idx * positions_per_pann
#             end_idx = min((pann_idx + 1) * positions_per_pann, len(grid_positions))
#
#             mapping[pann_idx] = grid_positions[start_idx:end_idx]
#
#             # Handle remaining positions
#             if pann_idx == self.pann_dim - 1 and end_idx < len(grid_positions):
#                 mapping[pann_idx].extend(grid_positions[end_idx:])
#
#         return mapping
#
#     def _calculate_wave_speed(self, target_freq: float) -> float:
#         """Calculate wave speed using BioOSS Eq.21"""
#
#         # BioOSS parameters
#         k_value = 10.0  # Damping coefficient
#
#         # Spatial frequency (simplified)
#         dx = 1.0
#         spatial_freq_norm = np.sqrt(2) / dx  # Assuming isotropic case
#
#         # BioOSS Eq.21: c = tan(πf·dt) * sqrt((1+dt*kp)*(1+dt*ko)) / (dt * sqrt(2) * |ξ|)
#         try:
#             tan_arg = np.pi * target_freq * self.dt
#             sqrt_term = np.sqrt((1 + self.dt * k_value) * (1 + self.dt * k_value))
#
#             c = np.tan(tan_arg) * sqrt_term / (self.dt * spatial_freq_norm)
#
#             # Ensure stability
#             c_max = dx / (self.dt * np.sqrt(2)) * sqrt_term
#             c = min(c, c_max * 0.9)
#
#             return max(c, 0.1)  # Minimum wave speed for numerical stability
#
#         except:
#             return 1.0  # Default wave speed
#
#     def reset_fields(self):
#         """Reset all field variables"""
#         self.p.zero_()
#         self.vx.zero_()
#         self.vy.zero_()
#         self.state_history.clear()
#         # Reset time step counter for sine wave generation
#         self.time_step = 0
#
#     def forward(self, pann_features: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass: convert PANN features to spatial excitation and evolve FDTD
#
#         Args:
#             pann_features: (batch_size, 527) PANN clipwise output
#
#         Returns:
#             p_field: (batch_size, grid_size, grid_size) evolved pressure field
#         """
#         batch_size = pann_features.shape[0]
#
#         # Convert PANN features to spatial excitation
#         spatial_excitation = self._pann_to_spatial_excitation(pann_features)
#
#         # Store current state
#         current_state = {
#             'p': self.p.clone(),
#             'vx': self.vx.clone(),
#             'vy': self.vy.clone()
#         }
#         self.state_history.append(current_state)
#
#         # Evolve FDTD system
#         self._fdtd_step(spatial_excitation)
#
#         # Increment time step for sine wave generation
#         self.time_step += 1
#
#         return self.p.squeeze()
#
#     def _pann_to_spatial_excitation(self, pann_features: torch.Tensor) -> torch.Tensor:
#         """Convert PANN features to spatial excitation pattern using frequency-specific sine waves"""
#
#         batch_size = pann_features.shape[0]
#         excitation = torch.zeros(batch_size, 1, self.grid_size, self.grid_size,
#                                  device=pann_features.device)
#
#         # Current time for sine wave generation
#         current_time = self.time_step * self.dt
#
#         for batch_idx in range(batch_size):
#             for pann_idx in range(self.pann_dim):
#                 # Get PANN probability as scaling factor
#                 probability = pann_features[batch_idx, pann_idx]
#
#                 # Get target frequency for this PANN dimension
#                 target_freq = self.target_frequencies[pann_idx]
#
#                 # Generate sine wave at target frequency
#                 sine_amplitude = torch.sin(2 * np.pi * target_freq * current_time)
#
#                 # Scale sine wave by PANN probability
#                 scaled_excitation = probability * sine_amplitude
#
#                 # Get corresponding grid positions
#                 grid_positions = self.pann_to_grid_mapping[pann_idx]
#
#                 # Apply scaled sine excitation to grid positions
#                 for grid_x, grid_y in grid_positions:
#                     excitation[batch_idx, 0, grid_x, grid_y] += scaled_excitation
#
#         return excitation
#
#     def _fdtd_step(self, source: Optional[torch.Tensor] = None):
#         """Single FDTD time step"""
#
#         # Calculate velocity divergence
#         if self.boundary_condition == 'periodic':
#             vx_shifted = torch.roll(self.vx, -1, dims=2)
#             dvx_dx = vx_shifted - self.vx
#
#             vy_shifted = torch.roll(self.vy, -1, dims=3)
#             dvy_dy = vy_shifted - self.vy
#         else:
#             dvx_dx = F.pad(self.vx[:, :, 1:, :] - self.vx[:, :, :-1, :], (0, 0, 0, 1))
#             dvy_dy = F.pad(self.vy[:, :, :, 1:] - self.vy[:, :, :, :-1], (0, 1, 0, 0))
#
#         # Update pressure field
#         self.p = ((1 - self.dt * self.kp) * self.p -
#                   self.c ** 2 * self.dt * (dvx_dx + dvy_dy))
#
#         # Add source if provided
#         if source is not None:
#             self.p = self.p + source
#
#         # Calculate pressure gradients
#         if self.boundary_condition == 'periodic':
#             p_shifted_x = torch.roll(self.p, 1, dims=2)
#             dp_dx = self.p - p_shifted_x
#
#             p_shifted_y = torch.roll(self.p, 1, dims=3)
#             dp_dy = self.p - p_shifted_y
#         else:
#             dp_dx = F.pad(self.p[:, :, :-1, :] - self.p[:, :, 1:, :], (0, 0, 1, 0))
#             dp_dy = F.pad(self.p[:, :, :, :-1] - self.p[:, :, :, 1:], (0, 1, 0, 0))
#
#         # Update velocity fields
#         self.vx = (1 - self.dt * self.ko) * self.vx - self.dt * dp_dx
#         self.vy = (1 - self.dt * self.ko) * self.vy - self.dt * dp_dy
#
#     def get_system_energy(self) -> float:
#         """Calculate total system energy"""
#         kinetic_energy = 0.5 * (torch.sum(self.vx ** 2) + torch.sum(self.vy ** 2))
#         potential_energy = 0.5 * torch.sum(self.p ** 2)
#
#         return (kinetic_energy + potential_energy).item()
#
#     def get_system_state_vector(self) -> torch.Tensor:
#         """Get flattened system state for analysis"""
#         return torch.cat([
#             self.p.flatten(),
#             self.vx.flatten(),
#             self.vy.flatten()
#         ])
#
#
# class OnlineBioOSSProcessor:
#     """Online processor for BioOSS FDTD system with change detection"""
#
#     def __init__(self, biooss_model: BioOSSFDTD2D, memory_length: int = 10):
#         self.biooss_model = biooss_model
#         self.memory_length = memory_length
#
#         # State history for change detection
#         self.state_history = deque(maxlen=memory_length)
#         self.energy_history = deque(maxlen=memory_length)
#
#         # Initialize system
#         self.biooss_model.reset_fields()
#
#     def process_timestep(self, pann_features: torch.Tensor) -> dict:
#         """
#         Process single timestep and return system state and metrics
#
#         Args:
#             pann_features: (1, 527) PANN features for current timestep
#
#         Returns:
#             result: Dictionary containing system state and metrics
#         """
#         # Process through BioOSS
#         pressure_field = self.biooss_model(pann_features)
#
#         # Calculate metrics
#         energy = self.biooss_model.get_system_energy()
#         state_vector = self.biooss_model.get_system_state_vector()
#
#         # Store in history
#         self.state_history.append(state_vector.clone())
#         self.energy_history.append(energy)
#
#         # Prepare results
#         result = {
#             'pressure_field': pressure_field,
#             'energy': energy,
#             'state_vector': state_vector,
#             'timestamp': len(self.energy_history) - 1
#         }
#
#         return result
#
#     def reset(self):
#         """Reset processor state"""
#         self.biooss_model.reset_fields()
#         self.state_history.clear()
#         self.energy_history.clear()
#         # Reset time step counter
#         self.biooss_model.time_step = 0