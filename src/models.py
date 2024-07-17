import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from einops.layers.torch import Rearrange
from braindecode.models import EEGNetv4  # Make sure you have braindecode installed

class EEGNetClassifier(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, input_window_samples: int) -> None:
        super().__init__()

        # Load pretrained EEGNet
        self.eegnet = EEGNetv4(in_chans=in_channels, n_classes=num_classes, input_window_samples=input_window_samples, final_conv_length='auto')
        
        # You can freeze the pretrained weights if required
        # for param in self.eegnet.parameters():
        #     param.requires_grad = False

        # Determine the number of output features
        self._initialize_head(num_classes, in_channels, input_window_samples)

    def _initialize_head(self, num_classes, in_channels, input_window_samples):
        # Create a dummy input tensor with the correct shape
        dummy_input = torch.randn(1, in_channels, input_window_samples)
        
        # Pass the dummy input through EEGNet to determine the output size
        with torch.no_grad():
            dummy_output = self.eegnet(dummy_input)
        
        output_features = dummy_output.shape[1]  # Number of output features
        
        # Define the linear layer with the correct number of input features
        self.head = nn.Linear(output_features, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass
        Args:
            X (Tensor): Input tensor of shape (b, c, t)
        Returns:
            Tensor: Output tensor of shape (b, num_classes)
        """
        X = self.eegnet(X)
        # print(f"Shape after EEGNet: {X.shape}")  # Comment out or remove this line
        return self.head(X)
    
class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass
        Args:
            X (Tensor): Input tensor of shape (b, c, t)
        Returns:
            Tensor: Output tensor of shape (b, num_classes)
        """
        X = self.blocks(X)
        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)
    

