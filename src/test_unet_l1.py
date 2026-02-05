import torch
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import BraTSDataset
from unet3d import UNet3D
from train import downsample

data_dir = Path.home() / "thesis" / "data" / "brats"
dataset = BraTSDataset(data_dir=data_dir, dummy=False, max_samples=5)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = UNet3D(in_channels=2, out_channels=1)
criterion = torch.nn.L1Loss()  # Pure L1, no FFL
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Testing UNet3D with L1 only...")
for epoch in range(10):
    epoch_loss = 0
    for inputs, targets in dataloader:
        inputs = torch.stack([downsample(inp) for inp in inputs])
        targets = torch.stack([downsample(tgt) for tgt in targets])
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"  Epoch {epoch+1}/10 | Loss: {epoch_loss/len(dataloader):.4f}")

print("\n✓ If no NaN above, U-Net is stable — problem is in FFL")