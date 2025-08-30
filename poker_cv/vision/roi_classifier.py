import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCardNet(nn.Module):
    """
    Tiny CNN with two heads:
      - rank_logits: 13 classes
      - suit_logits: 4 classes
    Input: grayscale 1x64x64
    """
    def __init__(self, rank_classes: int = 13, suit_classes: int = 4):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.fc = nn.Linear(128*4*4, 128)
        self.rank_head = nn.Linear(128, rank_classes)
        self.suit_head = nn.Linear(128, suit_classes)

    def forward(self, x):
        z = self.feature(x)
        z = z.view(z.size(0), -1)
        z = F.relu(self.fc(z))
        return self.rank_head(z), self.suit_head(z)

def cross_entropy_multihead(rank_logits, suit_logits, rank_targets, suit_targets):
    return F.cross_entropy(rank_logits, rank_targets) + F.cross_entropy(suit_logits, suit_targets)
