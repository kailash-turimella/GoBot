"""
ai.py — Neural network Go AI.

Loads a trained GoNetwork from `models/` and picks moves
by running a forward pass through the policy head. No MCTS.

Move selection:
  1. Encode the board into feature planes.
  2. Forward pass → 81 policy logits.
  3. Zero out illegal moves, softmax the rest.
  4. Return the highest-probability legal move, or None to pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from go_engine.board import Board, SIZE
from go_engine.rules import is_legal

MODEL_PATH = Path(__file__).resolve().parent / "models" / "v4.pth"


class ModelNotFoundError(FileNotFoundError):
    """Raised when the model file does not exist."""


def _load_model(path: Path):
    try:
        import torch
        from go_engine.network import GoNetwork
    except ImportError:
        raise ModelNotFoundError("PyTorch is not installed. Run: pip install torch")

    if not path.exists():
        raise ModelNotFoundError(
            f"No model found at '{path}'. Train and save the network there first."
        )

    model = GoNetwork()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def _encode(board: Board, color: int):
    """Encode the board into a (1, NUM_INPUT_PLANES, 9, 9) float32 tensor."""
    import torch
    from go_engine.network import NUM_INPUT_PLANES

    planes = torch.zeros(NUM_INPUT_PLANES, SIZE, SIZE, dtype=torch.float32)
    opponent = 3 - color
    for r in range(SIZE):
        for c in range(SIZE):
            v = board.grid[r][c]
            if v == color:
                planes[0, r, c] = 1.0
            elif v == opponent:
                planes[1, r, c] = 1.0
    planes[2] = 1.0 if color == 1 else 0.0
    return planes.unsqueeze(0)  # (1, NUM_INPUT_PLANES, 9, 9)


class AIPlayer:
    """
    Policy-network AI. Picks the highest-probability legal move from the
    network's policy head output.

    Raises ModelNotFoundError on construction if the model file is missing.
    """

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self._model = _load_model(model_path)

    def select_move(self, board: Board, color: int) -> Optional[tuple[int, int]]:
        """
        Return (row, col) of the best legal move according to the policy
        network, or None if no legal moves exist (AI should pass).
        """
        import torch
        import torch.nn.functional as F

        x = _encode(board, color)

        with torch.no_grad():
            logits, _ = self._model(x)          # (1, 81)

        logits = logits.squeeze(0)               # (81,)

        # Mask illegal moves with -inf before softmax
        for r in range(SIZE):
            for c in range(SIZE):
                if not is_legal(board, r, c, color):
                    logits[r * SIZE + c] = float("-inf")

        if logits.max().item() == float("-inf"):
            return None  # no legal moves → pass

        probs = F.softmax(logits, dim=0)
        idx = int(probs.argmax().item())
        return divmod(idx, SIZE)
