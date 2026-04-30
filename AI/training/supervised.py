"""
supervised.py — Pre-train GoNetwork on a dataset of SGF game files.

Step 1 of the training pipeline:
    python -m training.supervised --games AI/training/data/Games/ --epochs 20 --promote
    python -m training.self_play  --games 100 --sims 400
    python -m training.train      --epochs 10
    python -m training.eval       --candidate models/model_v10.pth

Put .sgf files anywhere under AI/training/data/Games/. Only 9×9 games are used.
Resigns and unknown results are excluded.

Usage:
    python -m training.supervised --games AI/training/data/Games/ --epochs 20 --promote
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from go_engine.board import Board
from go_engine.network import GoNetwork, NUM_INPUT_PLANES, save_model
from go_engine.rules import is_legal

SIZE = 9

_SIZE_RE   = re.compile(r'SZ\[(\d+)\]')
_RESULT_RE = re.compile(r'RE\[([^\]]*)\]')
_MOVE_RE   = re.compile(r';([BW])\[([a-i]{0,2})\]')


# ---------------------------------------------------------------------------
# SGF parsing
# ---------------------------------------------------------------------------

def _parse_sgf(text: str) -> tuple[list[tuple[int, int, int]], Optional[int]]:
    """
    Parse one SGF string.

    Returns:
        moves:  list of (color, row, col) — 1=black, 2=white; passes excluded
        winner: 1=black, 2=white, None if unknown or draw
    Raises:
        ValueError if not a 9×9 game
    """
    sz = _SIZE_RE.search(text)
    if not sz or int(sz.group(1)) != SIZE:
        raise ValueError("not 9×9")

    winner = None
    m = _RESULT_RE.search(text)
    if m:
        r = m.group(1).strip()
        if r.startswith("B+"):
            winner = 1
        elif r.startswith("W+"):
            winner = 2

    moves = []
    for m in _MOVE_RE.finditer(text):
        color = 1 if m.group(1) == "B" else 2
        coord = m.group(2)
        if len(coord) != 2:
            continue  # pass — skip
        col = ord(coord[0]) - ord("a")
        row = ord(coord[1]) - ord("a")
        if 0 <= row < SIZE and 0 <= col < SIZE:
            moves.append((color, row, col))

    return moves, winner


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

def _encode(board: Board, color: int) -> np.ndarray:
    """
    (NUM_INPUT_PLANES, 9, 9) float32. Planes 0/1 = current/opponent stones,
    plane 2 = colour, planes 3+ = zeros (history filled during self-play).
    """
    planes   = np.zeros((NUM_INPUT_PLANES, SIZE, SIZE), dtype=np.float32)
    opponent = 3 - color
    for r in range(SIZE):
        for c in range(SIZE):
            v = board.grid[r][c]
            if v == color:
                planes[0, r, c] = 1.0
            elif v == opponent:
                planes[1, r, c] = 1.0
    planes[2] = 1.0 if color == 1 else 0.0
    return planes


# ---------------------------------------------------------------------------
# Game replay → training examples
# ---------------------------------------------------------------------------

def _build_examples(
    path: Path,
) -> list[tuple[np.ndarray, int, float]]:
    """
    Replay one SGF file and produce (planes, move_idx, outcome) tuples.

    outcome = +1.0 if the player to move at that position won, else -1.0.
    Stops at the first illegal or out-of-order move.
    """
    try:
        moves, winner = _parse_sgf(path.read_text(errors="replace"))
    except ValueError:
        return []

    if not moves or winner is None:
        return []

    board    = Board()
    examples = []

    for color, row, col in moves:
        if board.turn != color:
            break
        if not is_legal(board, row, col, color):
            break

        planes   = _encode(board, color)
        move_idx = row * SIZE + col
        outcome  = 1.0 if winner == color else -1.0

        examples.append((planes, move_idx, outcome))
        board.place_stone(row, col)

    return examples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GoSupervisedDataset(Dataset):
    def __init__(self, examples: list[tuple[np.ndarray, int, float]]) -> None:
        self.planes   = torch.from_numpy(np.stack([e[0] for e in examples]))
        self.moves    = torch.tensor([e[1] for e in examples], dtype=torch.long)
        self.outcomes = torch.tensor([e[2] for e in examples], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.moves)

    def __getitem__(self, idx: int):
        return self.planes[idx], self.moves[idx], self.outcomes[idx]


def load_sgf_dataset(
    games_dir: str,
    max_games: Optional[int] = None,
    offset:    int = 0,
) -> GoSupervisedDataset:
    files = sorted(Path(games_dir).glob("**/*.sgf"))
    if not files:
        raise FileNotFoundError(f"No .sgf files found in {games_dir}")
    if offset:
        files = files[offset:]
    if max_games is not None:
        files = files[:max_games]

    all_examples: list = []
    skipped = 0
    for p in files:
        ex = _build_examples(p)
        if ex:
            all_examples.extend(ex)
        else:
            skipped += 1

    print(
        f"Loaded {len(all_examples)} positions from "
        f"{len(files) - skipped}/{len(files)} games ({skipped} skipped)"
    )
    return GoSupervisedDataset(all_examples)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_supervised(
    model:        GoNetwork,
    dataset:      GoSupervisedDataset,
    epochs:       int   = 20,
    batch_size:   int   = 512,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    device:       str   = "cpu",
    out_path:     str   = "models/supervised.pth",
) -> None:
    model.to(device).train()
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        total = policy_sum = value_sum = 0.0

        for planes, moves, outcomes in loader:
            planes, moves, outcomes = (
                planes.to(device), moves.to(device), outcomes.to(device)
            )
            logits, values = model(planes)
            values = values.squeeze(1)

            policy_loss = F.cross_entropy(logits, moves)
            value_loss  = F.mse_loss(values, outcomes)
            loss        = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = len(planes)
            total      += loss.item()       * n
            policy_sum += policy_loss.item() * n
            value_sum  += value_loss.item()  * n

        N = len(dataset)
        print(
            f"epoch {epoch}/{epochs}  "
            f"loss={total/N:.4f}  "
            f"policy={policy_sum/N:.4f}  "
            f"value={value_sum/N:.4f}"
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, out_path)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised pre-training on SGF games")
    parser.add_argument("--games",     type=str,   default="AI/training/data/Games/")
    parser.add_argument("--max-games", type=int,   default=None,
                        help="Number of SGF files to load in this chunk")
    parser.add_argument("--offset",    type=int,   default=0,
                        help="Skip the first N SGF files (for chunked training)")
    parser.add_argument("--resume",    type=str,   default=None,
                        help="Start from this checkpoint instead of a fresh network")
    parser.add_argument("--epochs",    type=int,   default=5)
    parser.add_argument("--batch",     type=int,   default=512)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--device",    type=str,   default="cpu")
    parser.add_argument("--out",       type=str,   default="models/supervised.pth")
    parser.add_argument(
        "--promote", action="store_true",
        help="Copy checkpoint to models/best_model.pth after training"
    )
    args = parser.parse_args()

    dataset = load_sgf_dataset(args.games, max_games=args.max_games, offset=args.offset)
    if args.resume:
        import torch
        model = GoNetwork()
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"Resumed from {args.resume}")
    else:
        model = GoNetwork()
    train_supervised(
        model, dataset,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        device=args.device, out_path=args.out,
    )

    if args.promote:
        dest = Path(args.out).parent / "best_model.pth"
        shutil.copy(args.out, dest)
        print(f"Promoted → {dest}")


if __name__ == "__main__":
    main()
