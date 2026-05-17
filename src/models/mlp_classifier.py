import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # As per the technical requirements (tz.txt):
        # Input -> 512 + ReLU + Dropout(0.5) -> 128 + ReLU -> 1 + Sigmoid
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Squeeze output to shape (B,)
        return self.net(x).squeeze(-1)
