"""
network.py — ResNet policy + value network for AlphaGo Zero-style training.

Architecture:
  Input:  N x 9 x 9 feature planes (current player's stones, opponent's stones,
          color-to-play, and optional history planes)
  Trunk:  5 residual blocks, 64 filters each, 3×3 convolutions
  Policy head: conv → flatten → linear → softmax over 81 moves
  Value head:  conv → flatten → linear → tanh → scalar in [-1, 1]

Value conventions:
  +1  = current player wins
  -1  = current player loses

Training inputs come from self_play.py; inference is called by ai.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Number of input feature planes fed to the network.
# Minimal: 2 (current player's stones, opponent's stones)
# Extended: add N history pairs + 1 color plane (see self_play.py)
NUM_INPUT_PLANES = 17  # 8 history pairs + 1 colour plane (AlphaGo Zero style)
NUM_FILTERS      = 64
NUM_RESIDUAL_BLOCKS = 5
BOARD_SIZE       = 9
NUM_MOVES        = BOARD_SIZE * BOARD_SIZE  # 81


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class GoNetwork(nn.Module):
    """
    Dual-head ResNet: takes a batch of board state planes and outputs
    (policy_logits, value) where policy_logits are raw pre-softmax scores
    over 81 intersections.
    """

    def __init__(
        self,
        input_planes: int = NUM_INPUT_PLANES,
        filters:      int = NUM_FILTERS,
        num_blocks:   int = NUM_RESIDUAL_BLOCKS,
    ) -> None:
        super().__init__()

        # Input convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_planes, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
        )

        # Residual tower
        self.residual_tower = nn.Sequential(
            *[_ResidualBlock(filters) for _ in range(num_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(filters, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, NUM_MOVES)

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_planes, 9, 9) float tensor
        Returns:
            policy_logits: (batch, 81)
            value:         (batch, 1)  in [-1, 1]
        """
        x = self.input_conv(x)
        x = self.residual_tower(x)

        p = self.policy_conv(x).flatten(1)
        p = self.policy_fc(p)

        v = self.value_conv(x).flatten(1)
        v = self.value_fc(v)

        return p, v


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def load_model(path: str, device: str = "cpu") -> GoNetwork:
    """Load a GoNetwork checkpoint from disk."""
    model = GoNetwork()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def save_model(model: GoNetwork, path: str) -> None:
    torch.save(model.state_dict(), path)
