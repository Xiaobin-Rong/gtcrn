import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from gtcrn import GTCRN  # Assuming GTCRN is defined in gtcrn.py
from loss import HybridLoss  # Assuming HybridLoss is defined in loss.py
from dataset import VCTKDEMANDDataset  # Custom Dataset class
from tqdm import tqdm

# Configuration parameters
batch_size = 1
num_epochs = 200
learning_rate = 0.001
checkpoint_dir = 'checkpoints'  # Directory to save the checkpoint

# Check if a GPU is available and if so, use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Ensure the checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

# Custom collate function to pad sequences


def collate_fn(batch):
    """Pads batch of variable length Tensors."""
    # Find the longest sequence in the batch
    max_length = max([x[0].shape[1] for x in batch])

    # Pad sequences to the max length
    padded_noisy = torch.zeros(len(batch), 1, max_length)
    padded_clean = torch.zeros(len(batch), 1, max_length)

    for i, (noisy, clean) in enumerate(batch):
        length = noisy.shape[1]
        padded_noisy[i, 0, :length] = noisy
        padded_clean[i, 0, :length] = clean

    return padded_noisy, padded_clean


# Instantiate the dataset and DataLoader
train_dataset = VCTKDEMANDDataset(root_dir='VCTK-DEMAND')
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Instantiate the model and loss function
model = GTCRN().to(device).train()
loss_func = HybridLoss().to(device)

# Define an optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with early stopping based on loss threshold
loss_threshold = 0.5  # Desired loss threshold

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for noisy_waveform, clean_waveform in tqdm(train_loader):
        # Move data to the same device as the model
        noisy_waveform = noisy_waveform.to(device)
        clean_waveform = clean_waveform.to(device)

        optimizer.zero_grad()

        # Convert waveform to spectrogram
        noisy_spectrogram = torch.stft(noisy_waveform.squeeze(
            1), n_fft=512, hop_length=256, return_complex=False)
        clean_spectrogram = torch.stft(clean_waveform.squeeze(
            1), n_fft=512, hop_length=256, return_complex=False)

        # Forward pass
        outputs = model(noisy_spectrogram)
        loss = loss_func(outputs, clean_spectrogram)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")
     # Check if the average loss is below the threshold
    if avg_loss <= loss_threshold:
        print(f"Stopping training as loss is below {loss_threshold}")
        break  # Stop training if loss is below the threshold

# Save the trained model
torch.save({'model': model.state_dict()}, os.path.join(
    checkpoint_dir, f'model_trained_on_vctk_jrqian{epoch+1}.tar'))


print("Training complete and model saved")
