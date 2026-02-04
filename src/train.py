import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import BraTSDataset
from losses import CombinedLoss
from augmentations import PhysicsAugmentation

class SimpleConvNet(nn.Module):
    """Temporary simple network for testing."""
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, out_channels, 3, padding=1),
        )
    
    def forward(self, x):
        return self.net(x)


def downsample(volume, size=(64, 64, 64)):
    """Downsample volume to fit in memory."""
    return F.interpolate(volume.unsqueeze(0), size=size, mode='trilinear', align_corners=False).squeeze(0)


def train_one_epoch(model, dataloader, criterion, optimizer, augmentation=None):
    model.train()
    total_loss = 0
    total_l1 = 0
    total_ffl = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Downsample to fit in CPU memory
        inputs = torch.stack([downsample(inp) for inp in inputs])
        targets = torch.stack([downsample(tgt) for tgt in targets])
        
        if augmentation:
            inputs = augmentation(inputs)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss, loss_dict = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_l1 += loss_dict['l1']
        total_ffl += loss_dict['ffl']
        
        print(f"  Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
    
    n = len(dataloader)
    return {'loss': total_loss/n, 'l1': total_l1/n, 'ffl': total_ffl/n}


if __name__ == "__main__":
    # Settings - small for CPU!
    BATCH_SIZE = 1
    EPOCHS = 10
    LR = 1e-3
    NUM_SAMPLES = 5  # Use only 5 brains for now
    
    # Data
    data_dir = Path.home() / "thesis" / "data" / "brats"
    dataset = BraTSDataset(data_dir=data_dir, dummy=False, max_samples=NUM_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset: {len(dataset)} samples")
    
    # Model
    model = SimpleConvNet(in_channels=2, out_channels=1)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss & optimizer
    criterion = CombinedLoss(l1_weight=1.0, ffl_weight=0.01)  # was 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Train!
    print("\n--- Training on REAL brain MRI! ---")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        metrics = train_one_epoch(model, dataloader, criterion, optimizer)
        print(f"Epoch {epoch+1} done | Loss: {metrics['loss']:.4f} | L1: {metrics['l1']:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "../outputs/model_v1.pth")
    print("\n✓ Model saved to outputs/model_v1.pth")