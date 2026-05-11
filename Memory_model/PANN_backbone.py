import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from typing import Optional, Tuple, List
import soundfile as sf


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise ValueError("Invalid pool_type!")
        return x


class PANNBackbone(nn.Module):
    """
    PANN CNN14 model for audio feature extraction
    Returns 527-dimensional clipwise output for audio classification
    """

    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320,
                 mel_bins=64, fmin=50, fmax=14000, classes_num=527):
        super(PANNBackbone, self).__init__()

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = min(fmax, sample_rate // 2 - 100)  # Ensure fmax is valid
        self.classes_num = classes_num

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window='hann', center=True, pad_mode='reflect', freeze_parameters=True)

        # Log-Mel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins,
            fmin=fmin, fmax=self.fmax, ref=1.0, amin=1e-10, top_db=None,
            freeze_parameters=True)

        # Spectrogram Augmentation (not used during inference)
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        # Convolutional blocks
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input_waveform):
        """
        Forward pass of PANN CNN14

        Args:
            input_waveform: (batch_size, samples) raw audio waveform

        Returns:
            output: Dictionary containing embedding, logits, and clipwise_output
        """
        # Extract spectrogram and log-mel features
        x = self.spectrogram_extractor(input_waveform)
        x = self.logmel_extractor(x)

        # Transpose for batch normalization: (batch, mel_bins, time) -> (batch, time, mel_bins) -> (batch, mel_bins, time)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # Add channel dimension for conv2d: (batch, mel_bins, time) -> (batch, 1, mel_bins, time)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # CNN layers with dropout
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        # Global pooling: combine max and mean pooling
        x = torch.mean(x, dim=3)  # Average over frequency dimension

        # Temporal pooling: max + mean
        x_max = torch.max(x, dim=2)[0]  # Max over time dimension
        x_mean = torch.mean(x, dim=2)  # Mean over time dimension
        x = x_max + x_mean

        # Fully connected layers
        embedding = F.relu_(self.fc1(x))
        logits = self.fc_audioset(embedding)
        clipwise_output = torch.sigmoid(logits)

        output = {
            'embedding': embedding,
            'logits': logits,
            'clipwise_output': clipwise_output
        }

        return output

    def extract_features_from_segments(self, audio_segments):
        """
        Extract PANN features from audio segments

        Args:
            audio_segments: List of audio segments or tensor (batch_size, samples)

        Returns:
            features: (num_segments, 527) PANN clipwise features
        """
        if isinstance(audio_segments, list):
            features = []
            for segment in audio_segments:
                if len(segment.shape) == 1:
                    segment = segment.unsqueeze(0)  # Add batch dimension
                output = self.forward(segment)
                features.append(output['clipwise_output'])
            return torch.cat(features, dim=0)
        else:
            output = self.forward(audio_segments)
            return output['clipwise_output']


class AudioEncoder(nn.Module):
    """Audio encoder wrapper for easy use"""

    def __init__(self, checkpoint_path: str, sample_rate: int = 32000,
                 window_size: int = 1024, hop_size: int = 320,
                 mel_bins: int = 64, fmin: int = 50, fmax: int = 14000,
                 classes_num: int = 527, device: str = 'cpu'):
        super(AudioEncoder, self).__init__()

        self.device = torch.device(device)
        self.sample_rate = sample_rate

        # Initialize PANN model
        self.model = PANNBackbone(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num
        )

        # Load pretrained weights
        self._load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self, checkpoint_path: str):
        """Load pretrained checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

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
        print(f"Loaded checkpoint from: {checkpoint_path}")

    def forward(self, audio_path_or_waveform):
        """
        Extract features from audio file or waveform

        Args:
            audio_path_or_waveform: Either path to audio file or waveform tensor

        Returns:
            output: Dictionary containing embedding, logits, and clipwise_output
        """
        if isinstance(audio_path_or_waveform, str):
            # Load from file
            waveform, _ = librosa.load(audio_path_or_waveform, sr=self.sample_rate, mono=True)
            waveform = torch.tensor(waveform).float().unsqueeze(0).to(self.device)
        else:
            # Use provided waveform
            waveform = audio_path_or_waveform
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            waveform = waveform.to(self.device)

        with torch.no_grad():
            output = self.model(waveform)
            # Move to CPU for further processing
            return {
                'embedding': output['embedding'].cpu(),
                'logits': output['logits'].cpu(),
                'clipwise_output': output['clipwise_output'].cpu()
            }


def load_pretrained_pann(checkpoint_path: str, device: str = 'cpu', **kwargs) -> PANNBackbone:
    """
    Load pretrained PANN model

    Args:
        checkpoint_path: Path to the pretrained checkpoint
        device: Device to load the model on
        **kwargs: Additional arguments for PANNBackbone

    Returns:
        model: Loaded PANN model
    """
    # Default parameters
    default_params = {
        'sample_rate': 32000,
        'window_size': 1024,
        'hop_size': 320,
        'mel_bins': 64,
        'fmin': 50,
        'fmax': 14000,
        'classes_num': 527
    }

    # Update with provided kwargs
    default_params.update(kwargs)

    model = PANNBackbone(**default_params)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

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

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model.to(device)

    print(f"Loaded PANN model from: {checkpoint_path}")

    return model


def create_sliding_windows(audio: np.ndarray, sr: int, window_length: float = 4.0,
                           stride: float = 1.0) -> List[np.ndarray]:
    """
    Create sliding windows from audio

    Args:
        audio: Audio waveform
        sr: Sample rate
        window_length: Window length in seconds
        stride: Stride length in seconds

    Returns:
        windows: List of audio windows
    """
    window_samples = int(window_length * sr)
    stride_samples = int(stride * sr)

    windows = []
    start = 0

    while start + window_samples <= len(audio):
        window = audio[start:start + window_samples]
        windows.append(window)
        start += stride_samples

    # Handle last window if not enough samples
    if start < len(audio):
        last_window = audio[start:]
        # Pad with zeros if necessary
        if len(last_window) < window_samples:
            padding = window_samples - len(last_window)
            last_window = np.pad(last_window, (0, padding))
        windows.append(last_window)

    return windows


def load_audio_file(audio_path: str, sample_rate: int = 32000) -> Tuple[np.ndarray, int]:
    """
    Load audio file with proper sample rate

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate

    Returns:
        audio: Audio waveform
        sr: Sample rate
    """
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    return audio, sr