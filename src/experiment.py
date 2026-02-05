import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from dataset import BraTSDataset
from losses import CombinedLoss
from train import SimpleConvNet, downsample

def train_model(model, dataloader, criterion, optimizer, epochs, name):
    """Train and return loss history."""
    history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_l1 = 0
        
        for inputs, targets in dataloader:
            inputs = torch.stack([downsample(inp) for inp in inputs])
            targets = torch.stack([downsample(tgt) for tgt in targets])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, loss_dict = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_l1 += loss_dict['l1']
        
        epoch_loss /= len(dataloader)
        epoch_l1 /= len(dataloader)
        history.append({'loss': epoch_loss, 'l1': epoch_l1})
        print(f"[{name}] Epoch {epoch+1}/{epochs} | L1: {epoch_l1:.4f}")
    
    return history


if __name__ == "__main__":
    # Settings
    EPOCHS = 15
    NUM_SAMPLES = 10
    LR = 1e-3
    
    # Data
    data_dir = Path.home() / "thesis" / "data" / "brats"
    dataset = BraTSDataset(data_dir=data_dir, dummy=False, max_samples=NUM_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Dataset: {len(dataset)} samples\n")
    
    # ============ MODEL 1: L1 ONLY (baseline) ============
    print("=" * 50)
    print("Training Model 1: L1 ONLY (baseline)")
    print("=" * 50)
    model_l1 = SimpleConvNet(in_channels=2, out_channels=1)
    criterion_l1 = CombinedLoss(l1_weight=1.0, ffl_weight=0.0)  # NO FFL!
    optimizer_l1 = torch.optim.Adam(model_l1.parameters(), lr=LR)
    history_l1 = train_model(model_l1, dataloader, criterion_l1, optimizer_l1, EPOCHS, "L1 Only")
    torch.save(model_l1.state_dict(), "../outputs/model_l1_only.pth")
    
    # ============ MODEL 2: L1 + FFL (yours) ============
    print("\n" + "=" * 50)
    print("Training Model 2: L1 + FFL (your method)")
    print("=" * 50)
    model_ffl = SimpleConvNet(in_channels=2, out_channels=1)
    criterion_ffl = CombinedLoss(l1_weight=1.0, ffl_weight=0.01, ffl_alpha=0.0)  # WITH FFL!
    optimizer_ffl = torch.optim.Adam(model_ffl.parameters(), lr=LR)
    history_ffl = train_model(model_ffl, dataloader, criterion_ffl, optimizer_ffl, EPOCHS, "L1 + FFL")
    torch.save(model_ffl.state_dict(), "../outputs/model_l1_ffl.pth")
    
    # ============ COMPARE PREDICTIONS ============
    print("\n" + "=" * 50)
    print("Comparing predictions...")
    print("=" * 50)
    
    # Load a test sample
    x, y = dataset[0]
    x_down = downsample(x, size=(64, 64, 64)).unsqueeze(0)
    y_down = downsample(y, size=(64, 64, 64))
    
    model_l1.eval()
    model_ffl.eval()
    
    with torch.no_grad():
        pred_l1 = model_l1(x_down).squeeze(0)
        pred_ffl = model_ffl(x_down).squeeze(0)
    
    # Visualize comparison
    mid = 32
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Top row: inputs and target
    axes[0, 0].imshow(x_down[0, 0, :, :, mid].numpy(), cmap='gray')
    axes[0, 0].set_title('T1 (Input)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(x_down[0, 1, :, :, mid].numpy(), cmap='gray')
    axes[0, 1].set_title('FLAIR (Input)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(y_down[0, :, :, mid].numpy(), cmap='gray')
    axes[0, 2].set_title('Ground Truth T1-Gd')
    axes[0, 2].axis('off')
    
    # Bottom row: predictions
    axes[1, 0].imshow(pred_l1[0, :, :, mid].numpy(), cmap='gray')
    axes[1, 0].set_title('L1 Only (Baseline)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_ffl[0, :, :, mid].numpy(), cmap='gray')
    axes[1, 1].set_title('L1 + FFL (Yours)')
    axes[1, 1].axis('off')
    
    # Difference map
    diff = torch.abs(pred_ffl - pred_l1)
    axes[1, 2].imshow(diff[0, :, :, mid].numpy(), cmap='hot')
    axes[1, 2].set_title('Difference (FFL vs L1)')
    axes[1, 2].axis('off')
    
    plt.suptitle('Thesis Experiment: Does Focal Frequency Loss Help?', fontsize=14)
    plt.tight_layout()
    plt.savefig('../outputs/experiment_comparison.png', dpi=150)
    plt.show()
    
    # Plot training curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([h['l1'] for h in history_l1], label='L1 Only', linewidth=2)
    ax.plot([h['l1'] for h in history_ffl], label='L1 + FFL', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L1 Loss')
    ax.set_title('Training Curves: L1 vs L1+FFL')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('../outputs/training_curves.png', dpi=150)
    plt.show()
    
    print("\n✓ Saved: experiment_comparison.png")
    print("✓ Saved: training_curves.png")
    print("\nThis is thesis-worthy data! 📊")