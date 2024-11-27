import torch

def get_cnn():
    modules = [
        torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),    # Convolution, 1 -> 32 channels, kernel size of 5, zero padding for same output size
        torch.nn.ReLU(),                                    # ReLU activation function
        torch.nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2),   # Convolution, 32 -> 64 channels, kernel size of 5, zero padding of 2, stride of 2
        torch.nn.ReLU(),                                    # ReLU activation function
        torch.nn.Conv2d(64, 64, kernel_size=5, padding=2),  # Convolution, 64 -> 64 channels, kernel size of 5, zero padding for same output size
        torch.nn.ReLU(),                                    # ReLU activation function
        torch.nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),  # Convolution, 64 -> 128 channels, kernel size of 5, zero padding of 2, stride of 2
        torch.nn.ReLU(),                                    # ReLU activation function
        torch.nn.AdaptiveAvgPool2d((1, 1)),                 # Average Pooling to make the output height and width 1
        torch.nn.Conv2d(128, 10, kernel_size=1),            # Convolution, 128 -> 10 channels, kernel size of 1
        torch.nn.Flatten(),                                  # Flatten to get N,10 for classification
    ]
    model = torch.nn.Sequential(*modules)
    print(f"Model:\n {model}")
    return model



class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        # Define C1, N1, R1
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu1 = torch.nn.ReLU(inplace=True)

        # Define C2, N2
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        # If dimensions change, define additional layers C3, N3
        self.has_skip_conv = stride != 1 or in_channels != out_channels
        if self.has_skip_conv:
            self.conv3 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn3 = torch.nn.BatchNorm2d(out_channels)

        self.relu2 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        # Residual path
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # If dimensions change, apply C3 and N3 to skip connection
        if self.has_skip_conv:
            residual = self.conv3(x)
            residual = self.bn3(residual)

        out += residual
        out = self.relu2(out)

        return out



class NonResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.C1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
        )
        self.N1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.R1 = torch.nn.ReLU()
        self.C2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.N2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.R2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.C1(x)
        x = self.N1(x)
        x = self.R1(x)
        x = self.C2(x)
        x = self.N2(x)
        x = self.R2(x)
        return x


def get_residual(depth, block_type="ResidualBlock", base_width=16):
    if block_type == "ResidualBlock":
        block_factory = ResidualBlock
    elif block_type == "NonResidualBlock":
        block_factory = NonResidualBlock
    else:
        raise ValueError()

    # Input layers
    modules = [
        torch.nn.Conv2d(1, base_width, 3, padding=1),
        torch.nn.BatchNorm2d(base_width),
        torch.nn.ReLU(),
    ]

    # Blocks and stages (based off the configuration used in the ResNet paper)
    blocks_per_stage = (depth - 2) // 6
    assert depth == blocks_per_stage * 6 + 2
    in_channels = base_width
    out_channels = base_width
    for stage_idx in range(3):
        for block_idx in range(blocks_per_stage):
            stride = 2 if block_idx == 0 and stage_idx > 0 else 1
            modules.append(
                block_factory(
                    in_channels,
                    out_channels,
                    stride,
                )
            )
            in_channels = out_channels
        out_channels = out_channels * 2

    # Output layers
    modules.extend(
        [
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(in_channels, 10),
        ]
    )

    model = torch.nn.Sequential(*modules)
    print(f"Model:\n {model}")
    return model