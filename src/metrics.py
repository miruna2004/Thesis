import torch
import torch.nn.functional as F
import math
import lpips

lpips_fn = lpips.LPIPS(net='squeeze', spatial=False)
#from evaluate import compute_lpips
def compute_lpips(pred, target):
    """
    Perceptual similarity (lower = better).
    Rewards sharp, realistic images over blurry ones.
    """
    # LPIPS expects (B, C, H, W) and values in [-1, 1]
    # We'll use a middle slice for 2D comparison
    mid = pred.shape[-1] // 2
    
    pred_slice = pred[:, :, :, mid].unsqueeze(0)  # (1, 1, H, W)
    target_slice = target[:, :, :, mid].unsqueeze(0)
    
    # Repeat to 3 channels (LPIPS expects RGB)
    pred_rgb = pred_slice.repeat(1, 3, 1, 1)
    target_rgb = target_slice.repeat(1, 3, 1, 1)
    
    # Scale to [-1, 1]
    pred_rgb = pred_rgb * 2 - 1
    target_rgb = target_rgb * 2 - 1
    
    with torch.no_grad():
        score = lpips_fn(pred_rgb, target_rgb)
    
    return score.item()

def psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio (higher = better)."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val) - 10 * math.log10(mse.item())


def ssim(pred, target, window_size=11):
    """Structural Similarity Index (higher = better, max=1)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Simple mean-based SSIM (not windowed, but good approximation)
    mu_pred = torch.mean(pred)
    mu_target = torch.mean(target)
    
    sigma_pred = torch.var(pred)
    sigma_target = torch.var(target)
    sigma_both = torch.mean((pred - mu_pred) * (target - mu_target))
    
    ssim_val = ((2 * mu_pred * mu_target + C1) * (2 * sigma_both + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
    
    return ssim_val.item()


def compute_all_metrics(pred, target):
    """Compute all metrics for a prediction."""
    return {
        'PSNR': psnr(pred, target),
        'SSIM': ssim(pred, target),
        'MAE': torch.mean(torch.abs(pred - target)).item(),
        'LPIPS': compute_lpips(pred, target),
    }


# Test
if __name__ == "__main__":
    # Perfect match
    a = torch.rand(1, 64, 64, 64)
    print("Perfect match:")
    print(compute_all_metrics(a, a))
    
    # Different images
    b = torch.rand(1, 64, 64, 64)
    print("\nRandom images:")
    print(compute_all_metrics(a, b))