import torch
import torch.nn as nn
import numpy as np
from queue import Queue
import threading
import time
from scipy.spatial.distance import cosine
import torch.nn.functional as F
import librosa
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

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

# ---------------- Convolutional Backbone (Cnn14) ----------------

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

class Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super(Cnn14, self).__init__()

        self.sample_rate = sample_rate  # Ensure sample rate is stored
        self.fmax = min(fmax, sample_rate // 2 - 100)  # Ensure valid fmax

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window='hann', center=True, pad_mode='reflect', freeze_parameters=True)

        # Log-Mel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins,
            fmin=fmin, fmax=self.fmax, ref=1.0, amin=1e-10, top_db=None,
            freeze_parameters=True)

        # Spectrogram Augmentation
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

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

    def forward(self, x):
        """Extracts embeddings from audio waveforms"""
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2))
        x = self.conv_block2(x, pool_size=(2, 2))
        x = self.conv_block3(x, pool_size=(2, 2))
        x = self.conv_block4(x, pool_size=(2, 2))
        x = self.conv_block5(x, pool_size=(2, 2))
        x = self.conv_block6(x, pool_size=(1, 1))

        x = torch.mean(x, dim=3)

        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)
        x = F.relu_(self.fc1(x))
        embedding = x

        return {'embedding': embedding}

# ---------------- Audio Encoder ----------------

class AudioEncoder(nn.Module):
    def __init__(self, args):
        super(AudioEncoder, self).__init__()
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

        self.model = Cnn14(
            sample_rate=args.sample_rate, window_size=args.window_size, hop_size=args.hop_size,
            mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax, classes_num=args.output_size)

        checkpoint = torch.load(args.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()

    def forward(self, audio_path):
        waveform, _ = librosa.load(audio_path, sr=self.model.sample_rate, mono=True)
        waveform = torch.tensor(waveform).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            # print(self.model(waveform)['embedding'].data.cpu().shape)
            return self.model(waveform)['embedding'].data.cpu()

# ---------------- Pattern Change Detector ----------------

class PatternChangeDetector:
    def __init__(self, args, similarity_threshold=0.99):
        self.similarity_threshold = similarity_threshold
        self.audio_encoder = AudioEncoder(args)  # Load pre-trained encoder
        self.audio_queue = Queue()
        self.recall_queue = Queue()
        self.prev_embedding = None
        self.running = True

    def process_audio_stream(self):
        while self.running:
            if not self.audio_queue.empty():
                audio_path = self.audio_queue.get()
                self.detect_pattern_change(audio_path)

    def detect_pattern_change(self, audio_path):
        """Extracts embedding, flattens it to 1D, and detects pattern change."""
        embedding = self.audio_encoder(audio_path)

        if embedding is None:
            print(f"Warning: No embedding extracted for {audio_path}. Skipping.")
            return

        embedding = embedding.squeeze().numpy()  # Flatten embedding to 1D

        similarity = 0
        if self.prev_embedding is not None:
            similarity = 1 - cosine(self.prev_embedding, embedding)
            print(f"Cosine Similarity: {similarity:.2f}")

            if similarity < self.similarity_threshold:
                print(f"Pattern change detected! Cosine similarity: {similarity:.2f}")
                self.recall_queue.put(audio_path)
        # else:
        #     similarity = 0

        # Set prev_embedding after comparison (not just when similarity is calculated)
        self.prev_embedding = embedding  # Store embedding for next comparison
        # print("there is prev_embedding", self.prev_embedding.shape)
        return similarity, embedding

    def stop(self):
        self.running = False

# ---------------- Run Detection ----------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=320)
    parser.add_argument("--mel_bins", type=int, default=64)
    parser.add_argument("--fmin", type=int, default=50)
    parser.add_argument("--fmax", type=int, default=14000)
    parser.add_argument("--output_size", type=int, default=527)
    parser.add_argument("--checkpoint_path", type=str, default='/home/zhyuan/Desktop/audioset_tagging_cnn/Cnn14_mAP=0.431.pth')

    args = parser.parse_args()

    detector = PatternChangeDetector(args)
    detection_thread = threading.Thread(target=detector.process_audio_stream)
    detection_thread.start()

    test_audio = '/home/zhyuan/Desktop/seq-memory/data/Speech/filtered_subset/3170-137482-0011.flac'
    detector.audio_queue.put(test_audio)
    time.sleep(5)

    detector.stop()
    detection_thread.join()
