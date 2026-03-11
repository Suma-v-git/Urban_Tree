import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetTreeSegmentation(nn.Module):
    """U-Net model for tree segmentation"""
    
    def __init__(self, encoder_name='resnet34', pretrained=True, in_channels=3, classes=1):
        super(UNetTreeSegmentation, self).__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=classes,
            activation=None  # No activation, we'll use BCEWithLogitsLoss
        )
    
    def forward(self, x):
        return self.model(x)

class CustomUNet(nn.Module):
    """Custom U-Net implementation for more control"""
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(CustomUNet, self).__init__()
        
        # Encoder (downsampling)
        self.downs = nn.ModuleList()
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder (upsampling)
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip_connection = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip_connection.shape:
                x = nn.functional.pad(x, [0, skip_connection.shape[3] - x.shape[3],
                                       0, skip_connection.shape[2] - x.shape[2]])
            
            # Concatenate and process
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        
        # Final output
        return self.final_conv(x)

class DoubleConv(nn.Module):
    """Double convolution block"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

def get_model(model_type='pretrained', **kwargs):
    """Get model based on type"""
    if model_type == 'pretrained':
        return UNetTreeSegmentation(**kwargs)
    elif model_type == 'custom':
        return CustomUNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def test_model():
    """Test model with dummy input"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test pretrained model
    model = get_model('pretrained')
    model.to(device)
    
    # Test custom model
    custom_model = get_model('custom')
    custom_model.to(device)
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        output1 = model(dummy_input)
        output2 = custom_model(dummy_input)
    
    print(f"Pretrained model output shape: {output1.shape}")
    print(f"Custom model output shape: {output2.shape}")
    print("Model test completed successfully!")

if __name__ == "__main__":
    test_model()
