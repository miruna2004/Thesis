import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import BraTSDataset

# Load real data
data_dir = Path.home() / "thesis" / "data" / "brats"
dataset = BraTSDataset(data_dir=data_dir, dummy=False, max_samples=1)

print("Loading brain MRI...")
x, y = dataset[0]
print(f"Input shape: {x.shape}, Target shape: {y.shape}")

# Pick middle slice (axial view)
mid_slice = x.shape[-1] // 2  # ~77

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# T1 (input channel 0)
axes[0].imshow(x[0, :, :, mid_slice].numpy(), cmap='gray')
axes[0].set_title('T1 (Input)')
axes[0].axis('off')

# FLAIR (input channel 1)
axes[1].imshow(x[1, :, :, mid_slice].numpy(), cmap='gray')
axes[1].set_title('FLAIR (Input)')
axes[1].axis('off')

# T1-Gd (target - what we want to predict!)
axes[2].imshow(y[0, :, :, mid_slice].numpy(), cmap='gray')
axes[2].set_title('T1-Gd (Target)')
axes[2].axis('off')

plt.suptitle(f'Real BraTS Brain MRI - Axial Slice {mid_slice}')
plt.tight_layout()
plt.savefig('../outputs/real_brain_slices.png', dpi=150)
plt.show()

print("Saved to outputs/real_brain_slices.png")