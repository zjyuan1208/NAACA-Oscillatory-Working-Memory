import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import librosa
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import gc
import traceback

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Import the PANNBackbone from the same location as the reference script
from PANN_backbone import PANNBackbone


class PANNClipwiseDetector:
    """Pattern change detector using PANN clipwise output (527 dimensions)"""

    def __init__(self, args, similarity_metric='cosine', similarity_threshold=0.945):
        self.args = args
        self.similarity_metric = similarity_metric  # 'cosine' or 'kl_divergence'
        self.similarity_threshold = similarity_threshold
        self.prev_clipwise = None
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

        # Initialize PANN model using PANNBackbone
        self._init_pann_model()

        print(f"PANN Clipwise Detector initialized:")
        print(f"  - Similarity metric: {similarity_metric}")
        print(f"  - Threshold: {similarity_threshold}")
        print(f"  - Device: {self.device}")
        print(f"  - PANN output dimension: {args.output_size}")

    def _init_pann_model(self):
        """Initialize PANN model using PANNBackbone (same as reference script)"""
        try:
            # Create model using PANNBackbone (same as reference script)
            self.model = PANNBackbone(
                sample_rate=self.args.sample_rate,
                window_size=self.args.window_size,
                hop_size=self.args.hop_size,
                mel_bins=self.args.mel_bins,
                fmin=self.args.fmin,
                fmax=self.args.fmax,
                classes_num=self.args.output_size
            )

            # Load checkpoint (same logic as reference script)
            if os.path.exists(self.args.checkpoint_path):
                checkpoint = torch.load(self.args.checkpoint_path, map_location=self.device)

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
                print(f"PANN model loaded from: {self.args.checkpoint_path}")
            else:
                print(f"Warning: Checkpoint not found at {self.args.checkpoint_path}")

            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            print(f"Error initializing PANN model: {str(e)}")
            traceback.print_exc()
            raise

    def extract_clipwise_output(self, audio_path):
        """Extract clipwise output from PANN model"""
        try:
            # Load audio (same as reference script)
            waveform, _ = librosa.load(audio_path, sr=self.model.sample_rate, mono=True)
            waveform = torch.tensor(waveform).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Get model output (same as reference script)
                output = self.model(waveform)
                # Return clipwise_output which is already 527-dimensional and sigmoid-activated
                clipwise_output = output['clipwise_output']  # Shape: [batch_size, 527]

                # Convert to numpy and squeeze batch dimension
                clipwise_np = clipwise_output.cpu().numpy().squeeze(0)  # Shape: [527]

                return clipwise_np

        except Exception as e:
            print(f"Error extracting clipwise output from {audio_path}: {str(e)}")
            traceback.print_exc()
            return None

    def compute_cosine_similarity(self, clipwise1, clipwise2):
        """Compute cosine similarity between two clipwise outputs"""
        try:
            # Cosine similarity = 1 - cosine distance
            similarity = 1 - cosine(clipwise1, clipwise2)
            return similarity
        except Exception as e:
            print(f"Error computing cosine similarity: {str(e)}")
            return 0.0

    def compute_js_divergence(self, clipwise1, clipwise2, epsilon=1e-8):
        """Compute Jensen-Shannon divergence between two clipwise outputs"""
        try:
            # Convert to probability distributions
            p = F.softmax(torch.tensor(clipwise1), dim=0).numpy()
            q = F.softmax(torch.tensor(clipwise2), dim=0).numpy()

            # Add epsilon and normalize
            p = p + epsilon
            q = q + epsilon
            p = p / np.sum(p)
            q = q / np.sum(q)

            # Compute average distribution M = (P + Q) / 2
            m = (p + q) / 2.0

            # JS divergence = (KL(P||M) + KL(Q||M)) / 2
            js_div = (entropy(p, m) + entropy(q, m)) / 2.0

            # Convert to similarity score (1 = identical, 0 = completely different)
            similarity = 1 - js_div

            return similarity

        except Exception as e:
            print(f"Error computing JS divergence: {e}")
            return float('inf')

    def compute_symmetric_kl_divergence(self, clipwise1, clipwise2, epsilon=1e-8):
        """Compute symmetric KL divergence between two clipwise outputs"""
        try:
            # The clipwise outputs are already sigmoid-activated probabilities
            # But we can still apply softmax for better probability distribution
            p = F.softmax(torch.tensor(clipwise1), dim=0).numpy()
            q = F.softmax(torch.tensor(clipwise2), dim=0).numpy()

            # Add small epsilon to avoid log(0)
            p = p + epsilon
            q = q + epsilon

            # Normalize to ensure they sum to 1
            p = p / np.sum(p)
            q = q / np.sum(q)

            # Compute symmetric KL divergence: (KL(P||Q) + KL(Q||P)) / 2
            kl_pq = entropy(p, q)
            kl_qp = entropy(q, p)
            symmetric_kl = (kl_pq + kl_qp) / 2.0

            # Convert to similarity (lower KL = higher similarity)
            # Use exponential decay to convert divergence to similarity [0,1]
            similarity = np.exp(-symmetric_kl)

            return similarity


        except Exception as e:
            print(f"Error computing symmetric KL divergence: {str(e)}")
            return 0.0

    def detect_pattern_change(self, audio_path):
        """Detect pattern change using PANN clipwise output"""
        try:
            # Extract clipwise output
            current_clipwise = self.extract_clipwise_output(audio_path)

            if current_clipwise is None:
                return 0.0, current_clipwise

            # If this is the first call, just store the clipwise output
            if self.prev_clipwise is None:
                self.prev_clipwise = current_clipwise
                return 1.0, current_clipwise  # High similarity for first call

            # Compute similarity based on chosen metric
            if self.similarity_metric == 'cosine':
                similarity = self.compute_cosine_similarity(self.prev_clipwise, current_clipwise)
            elif self.similarity_metric == 'kl_divergence':
                similarity = self.compute_symmetric_kl_divergence(self.prev_clipwise, current_clipwise)
            elif self.similarity_metric == 'js_divergence':
                similarity = self.compute_js_divergence(self.prev_clipwise, current_clipwise)
            else:
                raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

            # Update previous clipwise output
            self.prev_clipwise = current_clipwise

            return similarity, current_clipwise

        except Exception as e:
            print(f"Error in detect_pattern_change: {str(e)}")
            traceback.print_exc()
            return 0.0, None

    def stop(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'model'):
                del self.model
            self.prev_clipwise = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error in stop(): {str(e)}")

    def reset_fields(self):
        """Reset detector state"""
        self.prev_clipwise = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
