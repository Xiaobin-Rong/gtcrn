import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


class VCTKDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, max_length=160000):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.clean_files = sorted(os.listdir(clean_dir))
        self.noisy_files = sorted(os.listdir(noisy_dir))
        assert len(self.clean_files) == len(self.noisy_files), \
            "Mismatch between number of clean and noisy files."
        self.max_length = max_length
        self.n_fft = 512
        self.hop_length = 256
        self.win_length = 512
        self.window_fn = torch.hann_window(
            self.win_length).sqrt()  # create once

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])

        clean_signal, _ = torchaudio.load(clean_path)
        noisy_signal, _ = torchaudio.load(noisy_path)

        min_length = min(clean_signal.size(
            1), noisy_signal.size(1), self.max_length)
        clean_signal = clean_signal[:, :min_length]
        noisy_signal = noisy_signal[:, :min_length]

        if clean_signal.shape[1] < self.max_length:
            pad_amount = self.max_length - clean_signal.shape[1]
            clean_signal = torch.nn.functional.pad(
                clean_signal, (0, pad_amount))
            noisy_signal = torch.nn.functional.pad(
                noisy_signal, (0, pad_amount))

        # Ensure the window tensor is on the same device as the input tensors
        window_fn = self.window_fn.to(clean_signal.device)

        clean_spec = torch.stft(
            clean_signal, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window_fn, return_complex=True
        )
        noisy_spec = torch.stft(
            noisy_signal, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window_fn, return_complex=True
        )

        clean_spec = torch.view_as_real(clean_spec)
        noisy_spec = torch.view_as_real(noisy_spec)

        return noisy_spec, clean_spec


def create_dataloader(clean_dir, noisy_dir, batch_size, max_length=160000):
    dataset = VCTKDataset(clean_dir, noisy_dir, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
