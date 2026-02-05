import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import BraTSDataset
from train import SimpleConvNet, downsample
from losses import CombinedLoss
from metrics import compute_all_metrics

def train_and_evaluate(ffl_weight, alpha, epochs=10):
    """Quick train and evaluate with given settings."""
    
    # Data
    data_dir = Path.home() / "thesis" / "data" / "brats"
    dataset = BraTSDataset(data_dir=data_dir, dummy=False, max_samples=5)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Model
    model = SimpleConvNet(in_channels=2, out_channels=1)
    criterion = CombinedLoss(l1_weight=1.0, ffl_weight=ffl_weight, ffl_alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs = torch.stack([downsample(inp) for inp in inputs])
            targets = torch.stack([downsample(tgt) for tgt in targets])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, _ = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    x, y = dataset[0]
    x_down = downsample(x).unsqueeze(0)
    y_down = downsample(y)
    
    with torch.no_grad():
        pred = model(x_down).squeeze(0)
    
    metrics = compute_all_metrics(pred, y_down)
    return metrics


if __name__ == "__main__":
    # In quick_test.py, replace configs with:
    # In quick_test.py
    configs = [
    {'ffl_weight': 0.0,  'alpha': 0.0, 'name': 'L1 Only (baseline)'},
    {'ffl_weight': 1.0,  'alpha': 0.0, 'name': 'FFL α=0 (no focal)'},
    {'ffl_weight': 1.0,  'alpha': 0.5, 'name': 'FFL α=0.5 (mild focal)'},
    {'ffl_weight': 1.0,  'alpha': 1.0, 'name': 'FFL α=1.0 (full focal)'},
]
    
    print("=" * 70)
    print("HYPERPARAMETER SEARCH: Finding best FFL settings")
    print("=" * 70)
    
    results = []
    for cfg in configs:
        print(f"\nTraining: {cfg['name']}...")
        metrics = train_and_evaluate(cfg['ffl_weight'], cfg['alpha'])
        metrics['name'] = cfg['name']
        results.append(metrics)
        print(f"  PSNR: {metrics['PSNR']:.2f}, SSIM: {metrics['SSIM']:.4f}, LPIPS: {metrics['LPIPS']:.4f}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Config':<35} {'PSNR':<10} {'SSIM':<10} {'LPIPS':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<35} {r['PSNR']:<10.2f} {r['SSIM']:<10.4f} {r['LPIPS']:<10.4f}")