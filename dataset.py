import os
import torchaudio
from torch.utils.data import Dataset


class VCTKDEMANDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.clean_files = sorted(os.listdir(
            os.path.join(root_dir, 'train', 'clean')))
        self.noisy_files = sorted(os.listdir(
            os.path.join(root_dir, 'train', 'noisy')))

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = os.path.join(
            self.root_dir, 'train', 'clean', self.clean_files[idx])
        noisy_path = os.path.join(
            self.root_dir, 'train', 'noisy', self.noisy_files[idx])

        clean_waveform, _ = torchaudio.load(clean_path)
        noisy_waveform, _ = torchaudio.load(noisy_path)

        if self.transform:
            clean_waveform = self.transform(clean_waveform)
            noisy_waveform = self.transform(noisy_waveform)

        return noisy_waveform, clean_waveform
