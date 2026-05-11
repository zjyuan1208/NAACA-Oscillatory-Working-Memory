import torch
import torch.nn as nn
import numpy as np
import os
import time
import threading
from scipy.io import savemat
from queue import Queue
from torch.optim import Adam
import random
import torchaudio
import torchaudio.transforms as T
import time


# ---------------- Model Definition ----------------

class Tanh(nn.Module):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0

class Linear(nn.Module):
    def forward(self, inp):
        return inp

    def deriv(self, inp):
        return torch.ones((1,)).to(inp.device)

class MultilayertPC(nn.Module):
    """Multi-layer tPC for Echoic Memory"""
    def __init__(self, hidden_size, output_size, nonlin='tanh'):
        super(MultilayertPC, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wr.requires_grad_(True)
        self.Wout = nn.Linear(hidden_size, output_size, bias=False)
        self.Wout.requires_grad_(True)
        self.dropout = nn.Dropout(p=0.5)

        if nonlin == 'linear':
            self.nonlin = Linear()
        elif nonlin == 'tanh':
            self.nonlin = Tanh()
        else:
            raise ValueError("Invalid nonlinearity")

    def forward(self, prev_z):
        pred_z = self.Wr(self.nonlin(prev_z))
        pred_x = self.Wout(self.nonlin(pred_z))
        return pred_z, pred_x

    def init_hidden(self, bsz):
        return nn.init.kaiming_uniform_(torch.empty(bsz, self.hidden_size))

    def inference(self, inf_iters, inf_lr, x, prev_z):
        """Inference with multi-layer tPC"""
        with torch.no_grad():
            self.z, _ = self.forward(prev_z)
            for i in range(inf_iters):
                err_z, err_x = self.update_errs(x, prev_z)
                delta_z = err_z - self.nonlin.deriv(self.z) * torch.matmul(err_x, self.Wout.weight.detach().clone())
                self.z -= inf_lr * delta_z

    def update_errs(self, x, prev_z):
        pred_z, _ = self.forward(prev_z)
        pred_x = self.Wout(self.nonlin(self.z))
        err_z = self.z - pred_z
        err_x = x - pred_x
        return err_z, err_x


# ---------------- Memory Writing & Recall ----------------
def read_audio(audio_path, resample=True):
    r"""Loads audio file or array and returns a torch tensor"""
    # Randomly sample a segment of audio_duration from the clip or pad to match duration
    audio_time_series, sample_rate = torchaudio.load(audio_path)
    sampling_rate = 44100
    resample_rate = sampling_rate
    if resample and resample_rate != sample_rate:
        resampler = T.Resample(sample_rate, resample_rate)
        audio_time_series = resampler(audio_time_series)
    return audio_time_series, resample_rate

def load_audio_into_tensor(audio_path, audio_duration, resample=False):
    r"""Loads audio file and returns raw audio."""
    # Randomly sample a segment of audio_duration from the clip or pad to match duration
    audio_time_series, sample_rate = read_audio(audio_path, resample=resample)
    audio_time_series = audio_time_series.reshape(-1)

    # audio_time_series is shorter than predefined audio duration,
    # so audio_time_series is extended
    if audio_duration*sample_rate >= audio_time_series.shape[0]:
        repeat_factor = int(np.ceil((audio_duration*sample_rate) /
                                    audio_time_series.shape[0]))
        # Repeat audio_time_series by repeat_factor to match audio_duration
        audio_time_series = audio_time_series.repeat(repeat_factor)
        # remove excess part of audio_time_series
        audio_time_series = audio_time_series[0:audio_duration*sample_rate]
    else:
        # audio_time_series is longer than predefined audio duration,
        # so audio_time_series is trimmed
        start_index = random.randrange(
            audio_time_series.shape[0] - audio_duration*sample_rate)
        audio_time_series = audio_time_series[start_index:start_index +
                                              int(audio_duration*sample_rate)]
    return torch.FloatTensor(audio_time_series)

class EchoicMemory:
    """Handles streaming audio storage, memory updates, and recall"""
    def __init__(self, device='cuda:0', seq_len=20, input_size=8820, latent_size=3200, nonlin='tanh'):
        self.device = device
        self.seq_len = seq_len
        self.input_size = input_size
        self.latent_size = latent_size
        self.nonlin = nonlin
        self.model = MultilayertPC(latent_size, input_size, nonlin=nonlin).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        self.model_path = os.path.join('./results/echoic_memory/models')

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.memory_queue = Queue()  # For storing incoming audio chunks
        self.recall_queue = Queue()  # For triggering recall
        self.running = True  # Flag to keep processing threads active

    def process_memory_stream(self):
        """Continuously processes incoming audio for memory storage"""
        while self.running:
            if not self.memory_queue.empty():
                audio_path = self.memory_queue.get()
                self.train_memory(audio_path)

    def process_recall_stream(self):
        """Continuously listens for recall triggers"""
        while self.running:
            if not self.recall_queue.empty():
                recall_path = self.recall_queue.get()
                self.recall_memory(recall_path)

    # def train_memory(self, audio_path):
    #     """Writes incoming segment to echoic memory"""
    #     seq = self.load_audio_waveform(audio_path, sample_num=self.seq_len).to(self.device)
    #     print("seq shape is ", seq.shape)
    #
    #     prev_z = self.model.init_hidden(1).to(self.device)
    #
    #     for k in range(seq.shape[0]):
    #         x = seq[k].clone().detach()
    #         self.optimizer.zero_grad()
    #         self.model.inference(inf_iters=100, inf_lr=1e-2, x=x, prev_z=prev_z)
    #         energy = self.model.update_errs(x, prev_z)[0].sum()
    #         energy.backward()
    #         self.optimizer.step()
    #         prev_z = self.model.z.clone().detach()
    #         print(f"{k}-th segments")
    #
    #     self.save_model()
    #     print(f"Memory updated for {audio_path}.")
    def train_memory(self, audio_path, learn_iters=1, inf_iters=1, inf_lr=1e-2):
    # def train_memory(self, audio_path, learn_iters=10, inf_iters=100, inf_lr=1e-2):
        """Trains the memory with incoming audio segments."""
        seq = self.load_audio_waveform(audio_path, sample_rate=44100).to(self.device)
        # print("seq shape is ", seq.shape)

        seq_len = seq.shape[0]
        losses = []
        start_time = time.time()

        for learn_iter in range(learn_iters):
            epoch_loss = 0
            prev_z = self.model.init_hidden(1).to(self.device)

            for k in range(seq_len):
                x = seq[k].clone().detach()
                self.optimizer.zero_grad()

                # Perform inference and update model
                self.model.inference(inf_iters=inf_iters, inf_lr=inf_lr, x=x, prev_z=prev_z)
                energy = self.model.update_errs(x, prev_z)[0].sum()
                energy.backward()

                self.optimizer.step()

                # Detach and update previous z value
                prev_z = self.model.z.detach()

                # Add up the loss value at each time step
                epoch_loss += energy.item() / seq_len


            losses.append(epoch_loss)
            # if (learn_iter + 1) % 10 == 0:
            #     print(f'Epoch {learn_iter + 1}, loss {epoch_loss}')

        self.save_model()
        # print(f"Memory updated for {audio_path}, training complete. Time: {time.time() - start_time}")
        return losses

    # def train_memory(self, audio_paths, learn_iters=100, inf_iters=100, inf_lr=1e-2):
    #     """Writes incoming segments to echoic memory and trains on a batch of audio chunks."""
    #
    #     # Initialize list to hold sequences of audio
    #     seq_list = []
    #
    #     # Load and process each audio chunk
    #     for audio_path in audio_paths:
    #         seq = self.load_audio_waveform(audio_path, sample_num=self.seq_len).to(self.device)
    #         if seq is None or seq.numel() == 0:
    #             print(f"🔴 Warning: No data loaded from {audio_path}. Skipping.")
    #             continue
    #         seq_list.append(seq)
    #
    #     # Stack sequences into a batch (should have the shape [batch_size, seq_len, features])
    #     if len(seq_list) < 4:
    #         print("🔴 Not enough valid segments to form a full batch. Skipping processing.")
    #         return
    #
    #     # Concatenate all sequences along the batch dimension
    #     seq = torch.cat(seq_list, dim=0)
    #     print(f"🟢 Combined batch shape: {seq.shape}")
    #
    #     seq_len = seq.shape[0]
    #     losses = []
    #     start_time = time.time()
    #
    #     prev_z = self.model.init_hidden(1).to(self.device)  # Initialize hidden state
    #
    #     for learn_iter in range(learn_iters):  # Multiple epochs
    #         epoch_loss = 0
    #
    #         # Process each time step in the sequence
    #         for k in range(seq_len):
    #             x = seq[k].clone().detach()  # Extract one segment at a time
    #
    #             self.optimizer.zero_grad()  # Clear previous gradients
    #
    #             # Perform inference on the segment
    #             self.model.inference(inf_iters, inf_lr, x, prev_z)
    #
    #             # Calculate the energy loss
    #             energy = self.model.update_grads(x, prev_z)  # This is where the energy is calculated
    #             energy.backward()  # Perform backpropagation
    #
    #             self.optimizer.step()  # Update the model's parameters
    #
    #             # Detach the current z to avoid unnecessary backpropagation
    #             prev_z = self.model.z.detach()
    #
    #             # Add up the loss at each time step (to calculate average epoch loss)
    #             epoch_loss += energy.item() / seq_len
    #
    #         # Store the epoch loss and print every 10 epochs
    #         losses.append(epoch_loss)
    #         if (learn_iter + 1) % 10 == 0:
    #             print(f"Epoch {learn_iter + 1}, Loss: {epoch_loss:.4f}")
    #
    #     print(f"Training complete, time: {time.time() - start_time:.2f} seconds")
    #     return losses

    def recall_memory(self, recall_path):
        """Retrieves and reconstructs audio from memory"""
        self.model.load_state_dict(torch.load(os.path.join(self.model_path, f'mPC_len{self.seq_len}_{self.nonlin}.pt'),
                                              map_location=torch.device(self.device)))
        self.model.eval()

        seq = self.load_audio_waveform(recall_path, sample_rate=44100).to(self.device)
        recall_output = torch.zeros_like(seq).to(self.device)

        prev_z = self.model.init_hidden(1).to(self.device)

        for k in range(seq.shape[0]):
            x = seq[k].clone().detach()
            self.model.inference(inf_iters=500, inf_lr=1e-2, x=x, prev_z=prev_z)
            prev_z, pred_x = self.model(prev_z)
            recall_output[k] = pred_x

        array = recall_output.detach().cpu().numpy()
        savemat('./results/recalled_audio.mat', {'tensor': array})
        # print(f"Memory recalled for {recall_path}.")
        return array

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.model_path, f'mPC_len{self.seq_len}_{self.nonlin}.pt'))

    def load_audio_waveform(self, datapath, sample_rate):
        """Loads waveform as in the original code"""
        audio_files = [f'{datapath}']
        duration = 4  # duration in seconds
        chunk_duration = 0.2  # 200 msec
        audio_tensors = []

        for audio_file in audio_files:
            audio_tensor = load_audio_into_tensor(audio_file, duration, resample=True)
            audio_tensor = audio_tensor.reshape(1, -1).to(self.device)

            # sample_rate = 44100  # Replace with actual sample rate
            samples_per_chunk = int(chunk_duration * sample_rate)

            chunks = [audio_tensor[:, i:i + samples_per_chunk] for i in range(0, audio_tensor.size(1), samples_per_chunk)]
            audio_tensors.append(chunks)

        audio_tensors = torch.stack(audio_tensors[0][:-1]).squeeze()
        return audio_tensors

    def stop(self):
        """Stops memory processing"""
        self.running = False

# ---------------- Start Streaming Memory Processing ----------------

if __name__ == "__main__":
    echoic_memory = EchoicMemory()

    # Start streaming threads
    memory_thread = threading.Thread(target=echoic_memory.process_memory_stream)
    recall_thread = threading.Thread(target=echoic_memory.process_recall_stream)

    memory_thread.start()
    recall_thread.start()

    # Simulate incoming audio processing
    test_audio_path = '/home/zhyuan/Desktop/seq-memory/data/Speech/filtered_subset/3170-137482-0011.flac'
    echoic_memory.memory_queue.put(test_audio_path)  # Store new audio in memory
    time.sleep(5)  # Simulate delay
    echoic_memory.recall_queue.put(test_audio_path)  # Trigger recall

    time.sleep(10)  # Allow processing time
    echoic_memory.stop()
    memory_thread.join()
    recall_thread.join()
