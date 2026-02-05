import torch
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import BraTSDataset
from unet3d import UNet3D
from losses import CombinedLoss
from train import downsample
from metrics import compute_all_metrics

def train_and_evaluate(model_name, model, criterion, epochs=15):
    """Train model and return metrics."""
    
    data_dir = Path.home() / "thesis" / "data" / "brats"
    dataset = BraTSDataset(data_dir=data_dir, dummy=False, max_samples=10)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"\nTraining {model_name}...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for inputs, targets in dataloader:
            inputs = torch.stack([downsample(inp) for inp in inputs])
            targets = torch.stack([downsample(tgt) for tgt in targets])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, _ = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
    
    # Evaluate
    model.eval()
    all_metrics = {'PSNR': [], 'SSIM': [], 'MAE': [], 'LPIPS': []}
    
    for i in range(len(dataset)):
        x, y = dataset[i]
        x_down = downsample(x).unsqueeze(0)
        y_down = downsample(y)
        
        with torch.no_grad():
            pred = model(x_down).squeeze(0)
        
        metrics = compute_all_metrics(pred, y_down)
        for k in all_metrics:
            all_metrics[k].append(metrics[k])
    
    # Average metrics
    return {k: sum(v)/len(v) for k, v in all_metrics.items()}


if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT: SimpleConvNet vs UNet3D with FFL")
    print("=" * 60)
    
    # Import SimpleConvNet for comparison
    from train import SimpleConvNet
    
    # Best settings from yesterday
    criterion = CombinedLoss(l1_weight=1.0, ffl_weight=0.5, ffl_alpha=0.5)
    
    # Train SimpleConvNet
    model_simple = SimpleConvNet(in_channels=2, out_channels=1)
    results_simple = train_and_evaluate("SimpleConvNet", model_simple, criterion)
    
    # Train UNet3D
    model_unet = UNet3D(in_channels=2, out_channels=1)
    results_unet = train_and_evaluate("UNet3D", model_unet, criterion)
    
    # Compare
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Model':<20} {'PSNR':<10} {'SSIM':<10} {'LPIPS':<10}")
    print("-" * 60)
    print(f"{'SimpleConvNet':<20} {results_simple['PSNR']:<10.2f} {results_simple['SSIM']:<10.4f} {results_simple['LPIPS']:<10.4f}")
    print(f"{'UNet3D':<20} {results_unet['PSNR']:<10.2f} {results_unet['SSIM']:<10.4f} {results_unet['LPIPS']:<10.4f}")
    
    # Save U-Net model
    torch.save(model_unet.state_dict(), "../outputs/model_unet3d_ffl.pth")
    print("\n✓ UNet3D saved to outputs/model_unet3d_ffl.pth")