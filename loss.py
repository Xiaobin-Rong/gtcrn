import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_stft, true_stft):
        device = pred_stft.device

        pred_stft_real, pred_stft_imag = pred_stft[:,:,:,0], pred_stft[:,:,:,1]
        true_stft_real, true_stft_imag = true_stft[:,:,:,0], true_stft[:,:,:,1]
        pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag**(0.7))
        pred_imag_c = pred_stft_imag / (pred_mag**(0.7))
        true_real_c = true_stft_real / (true_mag**(0.7))
        true_imag_c = true_stft_imag / (true_mag**(0.7))
        real_loss = nn.MSELoss()(pred_real_c, true_real_c)
        imag_loss = nn.MSELoss()(pred_imag_c, true_imag_c)
        mag_loss = nn.MSELoss()(pred_mag**(0.3), true_mag**(0.3))
        
        y_pred = torch.istft(pred_stft_real+1j*pred_stft_imag, 512, 256, 512, window=torch.hann_window(512).pow(0.5).to(device))
        y_true = torch.istft(true_stft_real+1j*true_stft_imag, 512, 256, 512, window=torch.hann_window(512).pow(0.5).to(device))
        y_true = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true),dim=-1,keepdim=True) + 1e-8)

        sisnr =  - torch.log10(torch.norm(y_true, dim=-1, keepdim=True)**2 / (torch.norm(y_pred - y_true, dim=-1, keepdim=True)**2+1e-8) + 1e-8).mean()

        return 30*(real_loss + imag_loss) + 70*mag_loss + sisnr


if __name__ == "__main__":
    loss_func = HybridLoss()

    pred_stft = torch.randn(1, 257, 63, 2)
    true_stft = torch.randn(1, 257, 63, 2)
    loss = loss_func(pred_stft, true_stft)
    print(loss)
