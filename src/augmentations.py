import torch
import torch.nn as nn
import numpy as np

class KSpaceMotionArtifact(nn.Module):

    """
    for motion artifact simulation (frequency domain)
    "MRI motion causes ghosting and blurring in the phase encode direction"

    """
    def __init__(self, corruption_prob=0.5, max_lines=10):
        super().__init__()
        self.corruption_prob = corruption_prob
        self.max_lines = max_lines

    def forward(self, x):
        if torch.rand(1).item() > self.corruption_prob:
            return x
        
        kspace = torch.fft.fftn(x, dim=(-3, -2, -1))
        kspace = torch.fft.fftshift(kspace, dim=(-3, -2, -1))

        #corruption of random lines
        n_lines = torch.randint(1, self.max_lines + 1, (1,)).item()
        line_indices = torch.randint(0, x.shape[-2], (n_lines,))

        kspace[ ..., line_indices, :] *= 0.1

        #back to image space

        kspace = torch.fft.ifftshift(kspace, dim=(-3, -2, -1))
        corrupted = torch.fft.ifftn(kspace, dim=(-3, -2, -1))

        return corrupted.real
    
class BiasFieldArtifact(nn.Module):
    """
    B0 field inhomogeneity simulation (spatial domain)
    smooth intensitivity variations

    """

    def __init__(self, corruption_prob=0.5, strength = 0.3):
        super().__init__()
        self.corruption_prob = corruption_prob
        self.strength = strength

    def forward(self, x):
        if torch.rand(1).item() > self.corruption_prob:
            return x
            
        #generate smooth bias field
        shape = x.shape[-3:]
        low_res = torch.randn(1,1,4,4,4)
        bias_field = torch.nn.functional.interpolate(low_res, size=shape, mode = 'trilinear', align_corners = False)

        bias_field = 1 + self.strength * (bias_field / bias_field.abs().max())
        return x*bias_field.to(x.device)


class ResolutionDegradation(nn.Module):
    """
    thick slice aquisition simulation
    """

    def __init__(self, corruption_prob=0.5, downsample_factor=2):
        super().__init__()
        self.corruption_prob = corruption_prob
        self.scale_factor = downsample_factor

    def forward(self, x):
        if torch.rand(1).item() > self.corruption_prob:
            return x
        
        original_shape = x.shape[-3:]

        downsampled = torch.nn.functional.interpolate(
            x, scale_factor=(1/self.scale_factor, 1, 1), mode='trilinear', align_corners=False
        )
        restored = torch.nn.functional.interpolate(
            downsampled, size=original_shape, mode='trilinear', align_corners=False
        )
        
        return restored
    
class PhysicsAugmentation(nn.Module):
    """"combines all physics informed augmentation"""
    def __init__(self, prob=0.5):
        super().__init__()
        self.augmentations = [KSpaceMotionArtifact(corruption_prob=prob), BiasFieldArtifact(corruption_prob=prob), ResolutionDegradation(corruption_prob=prob)   ]


    def forward(self, x):
        for aug in self.augmentations:
            x = aug(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(1, 1, 32, 32, 32)

    print(f"Original shape: {x.shape}")
    print(f"Original mean: {x.mean():.4f}, std: {x.std():.4f}")

    motion = KSpaceMotionArtifact(corruption_prob=1.0)
    x_motion = motion(x)
    print(f"After motion artifact - mean: {x_motion.mean():.4f}, std: {x_motion.std():.4f}")
    
    bias = BiasFieldArtifact(corruption_prob=1.0)
    x_bias = bias(x)
    print(f"After bias field - mean: {x_bias.mean():.4f}, std: {x_bias.std():.4f}")
    
    resolution = ResolutionDegradation(corruption_prob=1.0)
    x_res = resolution(x)
    print(f"After resolution degradation - mean: {x_res.mean():.4f}, std: {x_res.std():.4f}")
    

    combined_aug = PhysicsAugmentation(prob=1.0)
    x_aug = combined_aug(x)
    print(f"After all augmentations - mean: {x_aug.mean():.4f}, std: {x_aug.std():.4f}")
    print(f"Output shape: {x_aug.shape}")
