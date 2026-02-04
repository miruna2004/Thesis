import torch
import torch.nn as nn
import torch.fft

class FrequencyMSELoss(nn.Module):
    """forces the network ro recover high frequcny components like edges  textiure, MSE IS BLURRY"""
    def __init__(self, alpha = 1.0, patch_factor=1, ave_spectrum=False):
        super(FrequencyMSELoss, self).__init__()
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum

    def forward(self, pred, target):
        """
        predicted volume is (B C D H W )
        and target is the ground truth volume ( B C D H W   )


        """

        pred_freq = torch.fft.fftn(pred, dim=(-3, -2, -1))
        target_freq = torch.fft.fftn(target, dim=(-3, -2, -1))

        pred_freq = torch.fft.fftshift(pred_freq, dim=(-3, -2, -1))
        target_freq = torch.fft.fftshift(target_freq, dim=(-3, -2, -1))

        #magnitude spectrum 
        pred_mag = torch.abs(pred_freq)
        target_mag = torch.abs(target_freq)

        #freq distance (error in frequency domain)
        freq_distance = (pred_mag - target_mag) ** 2

        weight = freq_distance ** self.alpha

        loss = torch.mean(weight * freq_distance)

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