import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class Encoder(nn.Module):
    """Encoder following the original MCD-DD implementation"""

    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class MCD_DD:
    """
    Maximum Concept Discrepancy-based Drift Detector
    Following the original implementation exactly
    """

    def __init__(self,
                 input_dim: int = 527,  # PANN clipwise output dimension
                 hidden_size: int = 100,  # Hidden layer size
                 output_size: int = 50,  # Output embedding size
                 sub_window_num: int = 10,  # Number of sub-windows in each sliding window
                 n: int = 1,  # Sample size per sample set (for audio data)
                 k: int = 10,  # Number of sample sets
                 eps_small: float = 0.01,  # Small noise for weak negative samples
                 eps_big: float = 0.1,  # Big noise for strong negative samples
                 temperature: float = 0.5,  # Temperature for contrastive loss
                 lamb: float = 1.0,  # Lambda for gradient penalty
                 percentile: float = 0.95,  # Percentile for threshold calculation
                 epochs: int = 1,  # Training epochs per window
                 learning_rate: float = 0.005,
                 device: str = 'cpu'):
        """
        Initialize MCD-DD detector following original implementation
        """
        self.sub_window_num = sub_window_num
        self.n = n
        self.k = k
        self.eps_small = eps_small
        self.eps_big = eps_big
        self.temperature = temperature
        self.lamb = lamb
        self.percentile = percentile
        self.epochs = epochs
        self.device = torch.device(device)

        # Initialize model and optimizer (following original structure)
        self.model = Encoder(input_dim, hidden_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Detection state
        self.threshold = 0.0
        self.first_window = True
        self.detection_history = []

    def generate_samples(self, sub_win_data: torch.Tensor) -> torch.Tensor:
        """
        Generate samples following original implementation
        Args:
            sub_win_data: Sub-window data tensor
        Returns:
            Samples tensor of shape [k, n, feature_dim]
        """
        sub_win_size = sub_win_data.size(0)

        if sub_win_size == 0:
            # Handle empty sub-window
            return torch.zeros(self.k, self.n, sub_win_data.size(-1)).to(self.device)

        # Generate indices for sampling (following original)
        indices = torch.randint(0, sub_win_size, (self.n * self.k,))
        samples = sub_win_data[indices].clone().detach().requires_grad_(True)
        samples = samples.view(self.k, self.n, -1)

        return samples.to(self.device)

    def compute_positive_loss(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute positive sample loss following original implementation
        Args:
            embeddings: Tensor of shape [k, output_size]
        Returns:
            mean_loss, all_losses
        """
        # Compute pairwise distances
        diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)  # [k, k, output_size]
        norm_diff = torch.norm(diff, p=2, dim=2)  # [k, k]

        # Upper triangular mask (excluding diagonal)
        mask = torch.triu(torch.ones_like(norm_diff), diagonal=1).bool()
        masked_norm_diff = norm_diff.masked_select(mask)

        mean_loss = masked_norm_diff.mean()
        return mean_loss, masked_norm_diff

    def compute_negative_loss(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute negative sample loss following original implementation
        """
        diff = embeddings1.unsqueeze(1) - embeddings2.unsqueeze(0)
        norm_diff = torch.norm(diff, p=2, dim=2)

        # Exclude diagonal elements if same size
        if norm_diff.size(0) == norm_diff.size(1):
            mask = torch.eye(norm_diff.size(0), dtype=torch.bool, device=norm_diff.device)
            masked_norm_diff = norm_diff.masked_select(~mask)
        else:
            masked_norm_diff = norm_diff.flatten()

        mean_loss = masked_norm_diff.mean()
        return mean_loss

    def contrastive_loss_function(self, pos_losses: List[torch.Tensor],
                                  neg_losses: List[torch.Tensor]) -> torch.Tensor:
        """
        InfoNCE-like contrastive loss following original implementation
        """
        if not pos_losses or not neg_losses:
            return torch.tensor(0.0, device=self.device)

        exp_pos_losses = torch.exp(torch.stack(pos_losses) / self.temperature)
        exp_neg_losses = torch.exp(torch.stack(neg_losses) / self.temperature)

        numerator = torch.sum(exp_pos_losses)
        denominator = numerator + torch.sum(exp_neg_losses)

        loss = torch.log(numerator / (denominator + 1e-8))
        return loss

    def gradient_penalty(self, samples: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Gradient penalty for Lipschitz constraint following original implementation
        """
        try:
            gradients = torch.autograd.grad(
                outputs=output,
                inputs=samples,
                grad_outputs=torch.ones_like(output),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
            penalty = ((gradients_norm - 1) ** 2).mean()
            return penalty
        except:
            return torch.tensor(0.0, device=self.device)

    def train(self, window_data: torch.Tensor) -> float:
        """
        Train on window data following original implementation exactly
        Args:
            window_data: Window data tensor
        Returns:
            threshold_dis: Calculated threshold
        """
        self.model.train()

        # Split window into sub-windows (following original logic)
        sub_window_size = int(window_data.size(0) / self.sub_window_num)
        windows = []
        for i in range(self.sub_window_num):
            start_idx = i * sub_window_size
            end_idx = min((i + 1) * sub_window_size, window_data.size(0))
            if end_idx > start_idx:
                windows.append(window_data[start_idx:end_idx])

        if len(windows) < self.sub_window_num:
            # Pad with the last available window if needed
            while len(windows) < self.sub_window_num:
                if windows:
                    windows.append(windows[-1])
                else:
                    break

        # Generate samples for each sub-window
        sub_win_samples = [self.generate_samples(sub_win) for sub_win in windows]

        # Training loop (following original implementation)
        for epoch in range(self.epochs):
            pos_losses = []
            weak_neg_losses = []

            for samples in sub_win_samples:
                # Positive sample pairs
                samples = samples.to(self.device)

                # h_p1 and h_p2 (following original variable naming)
                embeddings = self.model(samples)  # [k, n, output_size]
                embeddings_mean = embeddings.mean(dim=1)  # [k, output_size]

                # Loss for positive sample pairs
                pos_loss, _ = self.compute_positive_loss(embeddings_mean)
                pos_losses.append(pos_loss)

                # Weak negative sample pairs (following original logic exactly)
                unchanged_samples = samples[:self.k // 2]
                altered_samples = samples[self.k // 2:].clone() + torch.normal(
                    mean=0, std=self.eps_small,
                    size=samples[self.k // 2:].shape
                ).to(self.device)

                # h_wn1 and h_wn2
                embeddings_unchanged = self.model(unchanged_samples).mean(dim=1)
                embeddings_altered = self.model(altered_samples).mean(dim=1)

                # Loss for weak negative sample pairs
                weak_neg_loss = self.compute_negative_loss(embeddings_unchanged, embeddings_altered)
                weak_neg_losses.append(weak_neg_loss)

            # Strong negative sample pairs (following original logic)
            # sub-window 1 and sub-window N_sub
            if len(sub_win_samples) >= 2:
                first_sub_win_samples = sub_win_samples[0]
                last_sub_win_samples = sub_win_samples[-1]
                last_sub_win_samples_altered = last_sub_win_samples.clone() + torch.normal(
                    mean=0, std=self.eps_big,
                    size=last_sub_win_samples.shape
                ).to(self.device)

                # h_sn1 and h_sn2
                embeddings_first = self.model(first_sub_win_samples).mean(dim=1)
                embeddings_last_altered = self.model(last_sub_win_samples_altered).mean(dim=1)

                # Loss for strong sample pairs
                strong_neg_loss = self.compute_negative_loss(embeddings_first, embeddings_last_altered)

                # Gradient penalty (following original implementation)
                gp = self.gradient_penalty(samples, embeddings)

                # Total Loss with GP (following original formula)
                extended_neg_losses = weak_neg_losses + [strong_neg_loss]
                total_loss = self.contrastive_loss_function(pos_losses, extended_neg_losses) + self.lamb * gp

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        # Calculate threshold (following original implementation exactly)
        threshold_dis = self.calculate_threshold(windows)
        return threshold_dis

    def calculate_threshold(self, windows: List[torch.Tensor]) -> float:
        """
        Calculate threshold following original implementation exactly
        """
        self.model.eval()
        all_pos_losses = []

        with torch.no_grad():
            for sub_win in windows:
                samples = self.generate_samples(sub_win)
                embeddings = self.model(samples).mean(dim=1)
                _, all_pos_loss = self.compute_positive_loss(embeddings)
                all_pos_losses.append(all_pos_loss)

        if all_pos_losses:
            all_losses_tensor = torch.cat(all_pos_losses)
            threshold_dis = torch.quantile(all_losses_tensor, self.percentile)
            return threshold_dis.item()
        else:
            return self.threshold

    def test(self, window_data: torch.Tensor) -> List[float]:
        """
        Test on window data following original implementation exactly
        Args:
            window_data: Window data tensor
        Returns:
            distances: List of distances between last sub-window and previous ones
        """
        self.model.eval()

        # New sub-window data points at next sliding window (following original comment)
        sub_window_size = int(window_data.size(0) / self.sub_window_num)
        windows = []
        for i in range(self.sub_window_num):
            start_idx = i * sub_window_size
            end_idx = min((i + 1) * sub_window_size, window_data.size(0))
            if end_idx > start_idx:
                windows.append(window_data[start_idx:end_idx])

        embeddings = []
        distances = []

        # h_j and h_j' (following original variable naming)
        with torch.no_grad():
            for sub_win in windows:
                sub_win_samples = self.generate_samples(sub_win)
                embedding = self.model(sub_win_samples).mean(dim=1)
                embeddings.append(embedding)

        # Distance between the last sub-window and all previous sub-windows in the same sliding window
        # (following original comment exactly)
        if len(embeddings) >= 2:
            last_interval_embedding = embeddings[-1]
            distances = []
            for embedding in embeddings[:-1]:
                distance = self.compute_negative_loss(embedding, last_interval_embedding)
                distances.append(distance.item())

        return distances

    def batch_detect(self, data_sequence: torch.Tensor) -> Dict:
        """
        Batch process following the original main() logic
        """
        # Calculate window parameters (following original logic)
        T = data_sequence.shape[0]
        window_size = max(50, T // 10)  # Reasonable window size for audio
        slide = max(1, int(window_size / self.sub_window_num))

        detections = []
        confidence_scores = []
        timestamps = []
        thresholds = []

        # Process sliding windows (following original main loop)
        for i in range(0, T - window_size + 1, slide):
            window_data = data_sequence[i:i + window_size]

            if not self.first_window:
                # Test phase (following original logic)
                distances = self.test(window_data)
                if distances:
                    max_distance = max(distances)
                    is_drift = max_distance > self.threshold

                    detections.append(is_drift)
                    confidence = max(0.0, (max_distance - self.threshold) / (self.threshold + 1e-8))
                    confidence_scores.append(confidence)
                    timestamps.append(i)
                    thresholds.append(self.threshold)

                    if is_drift:
                        # Following original drift detection message format
                        drift_start = max(0, i + window_size - slide)
                        drift_end = min(T, i + window_size)
                        print(f"Drift detected between {drift_start} and {drift_end}")

            # Training phase (following original logic)
            self.threshold = self.train(window_data)
            self.first_window = False

        return {
            'detections': detections,
            'confidence_scores': confidence_scores,
            'timestamps': timestamps,
            'thresholds': thresholds
        }

    def reset(self):
        """Reset detector state"""
        self.first_window = True
        self.detection_history = []
        self.threshold = 0.0

        # Reinitialize model
        input_dim = self.model.layers[0].in_features
        hidden_size = self.model.layers[0].out_features
        output_size = self.model.layers[2].out_features

        self.model = Encoder(input_dim, hidden_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)


def smooth_detections(detections: List[bool], min_persistence: int = 2) -> List[bool]:
    """
    Apply temporal smoothing to reduce false positives
    """
    if len(detections) < min_persistence:
        return detections

    smoothed = [False] * len(detections)

    # Find consecutive detection sequences
    i = 0
    while i < len(detections):
        if detections[i]:
            # Count consecutive detections
            consecutive_count = 0
            start_idx = i

            while i < len(detections) and detections[i]:
                consecutive_count += 1
                i += 1

            # Keep detections if they meet persistence requirement
            if consecutive_count >= min_persistence:
                for j in range(start_idx, start_idx + consecutive_count):
                    smoothed[j] = True
        else:
            i += 1

    return smoothed