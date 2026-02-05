import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import BraTSDataset
from train import SimpleConvNet, downsample
from unet3d import UNet3D

# Load models
simple = SimpleConvNet(in_channels=2, out_channels=1)
simple.load_state_dict(torch.load("../outputs/model_l1_ffl.pth"))
simple.eval()

unet = UNet3D(in_channels=2, out_channels=1)
unet.load_state_dict(torch.load("../outputs/model_unet3d_ffl.pth"))
unet.eval()

# Load sample
data_dir = Path.home() / "thesis" / "data" / "brats"
dataset = BraTSDataset(data_dir=data_dir, dummy=False, max_samples=5)
x, y = dataset[0]

x_down = downsample(x).unsqueeze(0)
y_down = downsample(y)

# Predict
with torch.no_grad():
    pred_simple = simple(x_down).squeeze(0)
    pred_unet = unet(x_down).squeeze(0)

# Visualize
mid = 32
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Inputs and Ground Truth
axes[0, 0].imshow(x_down[0, 0, :, :, mid].numpy(), cmap='gray')
axes[0, 0].set_title('T1 (Input)')
axes[0, 0].axis('off')

axes[0, 1].imshow(x_down[0, 1, :, :, mid].numpy(), cmap='gray')
axes[0, 1].set_title('FLAIR (Input)')
axes[0, 1].axis('off')

axes[0, 2].imshow(y_down[0, :, :, mid].numpy(), cmap='gray')
axes[0, 2].set_title('Ground Truth T1-Gd')
axes[0, 2].axis('off')

axes[0, 3].axis('off')  # Empty

# Row 2: Predictions
axes[1, 0].imshow(pred_simple[0, :, :, mid].numpy(), cmap='gray')
axes[1, 0].set_title('SimpleConvNet\nPSNR: 30.26, LPIPS: 0.21')
axes[1, 0].axis('off')

axes[1, 1].imshow(pred_unet[0, :, :, mid].numpy(), cmap='gray')
axes[1, 1].set_title('UNet3D\nPSNR: 28.43, LPIPS: 0.12 ✓')
axes[1, 1].axis('off')

# Difference maps
axes[1, 2].imshow(torch.abs(pred_simple[0, :, :, mid] - y_down[0, :, :, mid]).numpy(), cmap='hot')
axes[1, 2].set_title('SimpleConvNet Error')
axes[1, 2].axis('off')

axes[1, 3].imshow(torch.abs(pred_unet[0, :, :, mid] - y_down[0, :, :, mid]).numpy(), cmap='hot')
axes[1, 3].set_title('UNet3D Error')
axes[1, 3].axis('off')

plt.suptitle('Architecture Comparison: SimpleConvNet vs UNet3D with FFL', fontsize=14)
plt.tight_layout()
plt.savefig('../outputs/architecture_comparison.png', dpi=150)
plt.show()

print("✓ Saved to outputs/architecture_comparison.png")