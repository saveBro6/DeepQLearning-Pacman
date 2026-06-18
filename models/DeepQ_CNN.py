import torch
import torch.nn as nn

class DQNCNN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQNCNN, self).__init__()
        
        # 3 Convolutional Blocks
        self.features = nn.Sequential(
            # Block 1: Input (4, 84, 84) -> Output (32, 20, 20)
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Block 2: Input (32, 20, 20) -> Output (64, 9, 9)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Block 3: Input (64, 9, 9) -> Output (64, 7, 7)
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        # Scale pixel values from [0, 255] to [0.0, 1.0]
        x = x.float() / 255.0
        x = self.features(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        return self.fc(x)
