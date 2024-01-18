import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, inp, hidden):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(inp, hidden, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden, inp, 3, padding="same")
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        F = self.conv_block(x)
        # IMPORTANT BIT: we sum the result of the
        # convolutions to the input image
        H = F + x
        # Now we apply ReLU and return
        return self.relu(H)
    
    
def get_conv(input_channels, output_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()
       
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, padding=7),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            get_conv(64, 128, 3, stride=2, padding=1),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            get_conv(128, 256, 3, stride=2, padding=1),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            get_conv(256, 512, 3, stride=2, padding=1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),            
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.model(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
