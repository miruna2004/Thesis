import torch
import torch.nn as nn
import torch.fft

class FrequencyMSELoss(nn.Module):
    """Focal Frequency Loss - focuses on hard-to-reconstruct frequencies."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, target):
        # Compute FFT
        pred_freq = torch.fft.fftn(pred, dim=(-3, -2, -1))
        target_freq = torch.fft.fftn(target, dim=(-3, -2, -1))
        
        # Magnitude spectrum
        pred_mag = torch.abs(pred_freq)
        target_mag = torch.abs(target_freq)
        
        # Frequency error (normalized to prevent explosion)
        freq_error = torch.abs(pred_mag - target_mag)
        freq_error = freq_error / (target_mag.max() + 1e-8)  # normalize
        
        # FOCAL weighting: penalize hard frequencies more
        # But clamp to prevent NaN
        weight = torch.clamp(freq_error, 0, 1) ** self.alpha + 1e-8
        
        # Weighted loss
        loss = torch.mean(weight * (freq_error ** 2))
        
        return loss
    
class CombinedLoss(nn.Module):
    """L1 + FFL"""
    def __init__(self, l1_weight = 1.0, ffl_weight=0.1, ffl_alpha=1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.ffl_weight = ffl_weight
        self.l1_loss = nn.L1Loss()
        self.ffl_loss = FrequencyMSELoss(alpha=ffl_alpha)


    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ffl = self.ffl_loss(pred, target)

        total_loss = self.l1_weight * l1 + self.ffl_weight * ffl
        loss_dict = {'l1': l1.item(), 'ffl': ffl.item()}

        return total_loss, loss_dict

if __name__ == "__main__":
   
    pred = torch.randn(1, 1, 32, 32, 32)
    target = torch.randn(1, 1, 32, 32, 32)

    ffl = FrequencyMSELoss()
    loss_ffl = ffl(pred, target)
    print(f"Frequency MSE Loss: {loss_ffl.item()}")

    combined = CombinedLoss(l1_weight=1.0, ffl_weight=0.1)
    loss_total, loss_dict = combined(pred, target)

    print(f"Combined loss: {loss_total.item():.4f}")
    print(f"  L1: {loss_dict['l1']:.4f}, FFL: {loss_dict['ffl']:.4f}")