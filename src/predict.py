import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import BraTSDataset
from train import SimpleConvNet, downsample

# Load model
model = SimpleConvNet(in_channels=2, out_channels=1)
model.load_state_dict(torch.load("../outputs/model_v1.pth"))
model.eval()
print("Model loaded!")

# Load a sample
data_dir = Path.home() / "thesis" / "data" / "brats"
dataset = BraTSDataset(data_dir=data_dir, dummy=False, max_samples=5)

# Get sample (use index 0 or try others)
x, y = dataset[0]

# Downsample to match training
x_down = downsample(x, size=(64, 64, 64)).unsqueeze(0)  # (1, 2, 64, 64, 64)
y_down = downsample(y, size=(64, 64, 64))  # (1, 64, 64, 64)

# Predict
with torch.no_grad():
    pred = model(x_down).squeeze(0)  # (1, 64, 64, 64)

print(f"Input shape: {x_down.shape}")
print(f"Prediction shape: {pred.shape}")
print(f"Target shape: {y_down.shape}")

# Visualize middle slice
mid = 32

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# T1 input
axes[0].imshow(x_down[0, 0, :, :, mid].numpy(), cmap='gray')
axes[0].set_title('T1 (Input)')
axes[0].axis('off')

# FLAIR input
axes[1].imshow(x_down[0, 1, :, :, mid].numpy(), cmap='gray')
axes[1].set_title('FLAIR (Input)')
axes[1].axis('off')

# Prediction
axes[2].imshow(pred[0, :, :, mid].numpy(), cmap='gray')
axes[2].set_title('Predicted T1-Gd')
axes[2].axis('off')

# Ground truth
axes[3].imshow(y_down[0, :, :, mid].numpy(), cmap='gray')
axes[3].set_title('Real T1-Gd (Target)')
axes[3].axis('off')

plt.suptitle('Contrast Synthesis: Your Model vs Ground Truth')
plt.tight_layout()
plt.savefig('../outputs/prediction_v1.png', dpi=150)
plt.show()

print("Saved to outputs/prediction_v1.png")