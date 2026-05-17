import torch
import torch.nn as nn
import torchvision.models as models

class CNNFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # EfficientNet-B4
        if pretrained:
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b4(weights=weights)
        else:
            self.model = models.efficientnet_b4(weights=None)
            
        # Remove classifier to get embeddings
        # efficientnet has .classifier which is a Sequential.
        # We can replace it with Identity to get the output from the pooling layer
        self.model.classifier = nn.Identity()
        
    def forward(self, x):
        # x is expected to be [B, 3, 224, 224]
        return self.model(x)

class ViTFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            self.model = models.vit_b_16(weights=weights)
        else:
            self.model = models.vit_b_16(weights=None)
            
        # VisionTransformer in torchvision applies heads to the CLS token
        # By setting heads to Identity, it will just return the CLS token
        self.model.heads = nn.Identity()
        
    def forward(self, x):
        return self.model(x)

class SpectralFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.model = models.resnet18(weights=weights)
        else:
            self.model = models.resnet18(weights=None)
            
        # Modify the first conv layer to accept 1 channel (magnitude spectrum) instead of 3
        # Keep the pretrained weights for the rest of the model where possible
        original_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # If pretrained, we can initialize the 1-channel conv by averaging the weights of the 3-channel conv
        if pretrained:
            with torch.no_grad():
                self.model.conv1.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
                
        # Replace the final fully connected layer with Identity to output features
        self.model.fc = nn.Identity()
        
    def forward(self, x):
        """
        x: [B, 3, 224, 224] RGB images.
        We will compute the 2D FFT magnitude spectrum to extract frequency features.
        """
        # Convert RGB to Grayscale
        # Y = 0.2989 R + 0.5870 G + 0.1140 B
        if x.shape[1] == 3:
            gray = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
        else:
            gray = x
            
        # Compute 2D FFT
        # fft2 computes the 2-dimensional discrete Fourier transform
        fft_complex = torch.fft.fft2(gray)
        
        # Shift the zero-frequency component to the center of the spectrum
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
        
        # Calculate magnitude spectrum and apply log scale to compress dynamic range
        magnitude_spectrum = torch.abs(fft_shifted)
        # Add epsilon to avoid log(0)
        log_magnitude = torch.log(magnitude_spectrum + 1e-8)
        
        # Normalize to [0, 1] approximately, or just standardize per batch
        # This helps the CNN process it effectively
        mean = log_magnitude.mean(dim=(2, 3), keepdim=True)
        std = log_magnitude.std(dim=(2, 3), keepdim=True) + 1e-8
        normalized_spectrum = (log_magnitude - mean) / std
        
        # Pass through the CNN
        features = self.model(normalized_spectrum)
        return features

def get_extractors(device):
    """Utility to initialize and return all 3 extractors moved to the device"""
    cnn = CNNFeatureExtractor().to(device).eval()
    vit = ViTFeatureExtractor().to(device).eval()
    spectral = SpectralFeatureExtractor().to(device).eval()
    return {"cnn": cnn, "vit": vit, "spectral": spectral}
