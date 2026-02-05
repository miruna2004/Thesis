import torch
from pathlib import Path
from dataset import BraTSDataset
from train import SimpleConvNet, downsample
from metrics import compute_all_metrics
import lpips

lpips_fn = lpips.LPIPS(net='squeeze', spatial=False)

# Load models
model_l1 = SimpleConvNet(in_channels=2, out_channels=1)
model_l1.load_state_dict(torch.load("../outputs/model_l1_only.pth"))
model_l1.eval()

model_ffl = SimpleConvNet(in_channels=2, out_channels=1)
model_ffl.load_state_dict(torch.load("../outputs/model_l1_ffl.pth"))
model_ffl.eval()

print("Models loaded!\n")

# Load test data
data_dir = Path.home() / "thesis" / "data" / "brats"
dataset = BraTSDataset(data_dir=data_dir, dummy=False, max_samples=10)

# Evaluate on all samples
results_l1 = {'PSNR': [], 'SSIM': [], 'MAE': [], 'LPIPS': []}
results_ffl = {'PSNR': [], 'SSIM': [], 'MAE': [], 'LPIPS': []}

print("Evaluating on test samples...")
for i in range(len(dataset)):
    x, y = dataset[i]
    x_down = downsample(x, size=(64, 64, 64)).unsqueeze(0)
    y_down = downsample(y, size=(64, 64, 64))
    
    with torch.no_grad():
        pred_l1 = model_l1(x_down).squeeze(0)
        pred_ffl = model_ffl(x_down).squeeze(0)
    
    metrics_l1 = compute_all_metrics(pred_l1, y_down)
    metrics_ffl = compute_all_metrics(pred_ffl, y_down)
    
    for k in results_l1.keys():
        results_l1[k].append(metrics_l1[k])
        results_ffl[k].append(metrics_ffl[k])
    
    print(f"  Sample {i+1}/{len(dataset)} done")

# Compute averages
print("\n" + "=" * 50)
print("RESULTS: L1 Only vs L1 + FFL")
print("=" * 50)
print(f"\n{'Metric':<10} {'L1 Only':<15} {'L1 + FFL':<15} {'Winner'}")
print("-" * 50)

for metric in ['PSNR', 'SSIM', 'MAE', 'LPIPS']:
    avg_l1 = sum(results_l1[metric]) / len(results_l1[metric])
    avg_ffl = sum(results_ffl[metric]) / len(results_ffl[metric])
    
    # Higher is better for PSNR/SSIM, lower for MAE/LPIPS
    if metric in ['MAE', 'LPIPS']:
        winner = "FFL ✓" if avg_ffl < avg_l1 else "L1"
    else:
        winner = "FFL ✓" if avg_ffl > avg_l1 else "L1"
    
    print(f"{metric:<10} {avg_l1:<15.4f} {avg_ffl:<15.4f} {winner}")
print("\n" + "=" * 50)
print("Copy these numbers into your thesis!")
print("=" * 50)