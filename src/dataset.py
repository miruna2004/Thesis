import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from pathlib import Path

class BraTSDataset(Dataset):
    """
    BraTS dataset for contrast synthesis.
    Input: T1 + FLAIR (2 channels)
    Target: T1-Gd (1 channel)
    """
    def __init__(self, data_dir, transform=None, dummy=False, max_samples=None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.transform = transform
        self.dummy = dummy
        
        if dummy:
            self.samples = list(range(10))
        else:
            # Find all subject folders
            self.samples = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
            if max_samples:
                self.samples = self.samples[:max_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.dummy:
            t1 = torch.randn(1, 32, 32, 32)
            flair = torch.randn(1, 32, 32, 32)
            t1ce = torch.randn(1, 32, 32, 32)
        else:
            subject_dir = self.samples[idx]
            subject_id = subject_dir.name
            
            # Load NIfTI files
            t1 = self._load_nifti(subject_dir / f"{subject_id}_t1.nii.gz")
            flair = self._load_nifti(subject_dir / f"{subject_id}_flair.nii.gz")
            t1ce = self._load_nifti(subject_dir / f"{subject_id}_t1ce.nii.gz")
            
            # Normalize to [0, 1]
            t1 = self._normalize(t1)
            flair = self._normalize(flair)
            t1ce = self._normalize(t1ce)
            
            # Convert to tensor and add channel dim
            t1 = torch.from_numpy(t1).float().unsqueeze(0)
            flair = torch.from_numpy(flair).float().unsqueeze(0)
            t1ce = torch.from_numpy(t1ce).float().unsqueeze(0)
        
        # Stack T1 + FLAIR as 2-channel input
        input_vol = torch.cat([t1, flair], dim=0)  # (2, D, H, W)
        target_vol = t1ce  # (1, D, H, W)
        
        if self.transform:
            input_vol = self.transform(input_vol)
        
        return input_vol, target_vol
    
    def _load_nifti(self, path):
        """Load a NIfTI file and return numpy array."""
        img = nib.load(path)
        return img.get_fdata()
    
    def _normalize(self, volume):
        """Normalize volume to [0, 1]."""
        volume = volume - volume.min()
        if volume.max() > 0:
            volume = volume / volume.max()
        return volume


# Test with real data
if __name__ == "__main__":
    data_dir = Path.home() / "thesis" / "data" / "brats"
    
    # Load just 2 samples for testing
    dataset = BraTSDataset(data_dir=data_dir, dummy=False, max_samples=2)
    print(f"Dataset size: {len(dataset)}")
    
    print("Loading first sample (this may take a moment)...")
    x, y = dataset[0]
    print(f"Input shape: {x.shape}")   # Should be (2, 240, 240, 155)
    print(f"Target shape: {y.shape}")  # Should be (1, 240, 240, 155)
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")