"""
eval.py — Pit two model checkpoints against each other.

Each game uses direct policy (greedy, no MCTS) so evaluation is fast.
Colors are alternated every game so neither model has a systematic advantage.

Usage:
    # Compare a specific pair
    python -m AI.training.eval --black AI/models/v1.pth --white AI/models/model_v10.pth

    # Find the best among all checkpoints in a folder
    python -m AI.training.eval --folder AI/models/ --games 20

    # Promote winner to v2.pth
    python -m AI.training.eval --folder AI/models/ --games 20 --promote v2
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from go_engine.board import Board, SIZE
from go_engine.network import load_model
from go_engine.rules import is_legal
from go_engine.scoring import get_winner

MAX_GAME_MOVES = 200


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def play_game(model_black, model_white, device: str = "cpu") -> int:
    """
    Play one game: model_black plays black, model_white plays white.
    Returns 1 if black wins, 2 if white wins.
    Uses greedy policy (argmax) for deterministic evaluation.
    """
    board  = Board()
    models = {1: model_black, 2: model_white}
    move_num = 0

    while not board.game_over and move_num < MAX_GAME_MOVES:
        color    = board.turn
        model    = models[color]
        opponent = 3 - color

        planes = np.zeros((17, SIZE, SIZE), dtype=np.float32)
        for r in range(SIZE):
            for c in range(SIZE):
                v = board.grid[r][c]
                if v == color:
                    planes[0, r, c] = 1.0
                elif v == opponent:
                    planes[1, r, c] = 1.0
        planes[2] = 1.0 if color == 1 else 0.0

        x = torch.from_numpy(planes).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(x)
        logits = logits.squeeze(0)

        for r in range(SIZE):
            for c in range(SIZE):
                if not is_legal(board, r, c, color):
                    logits[r * SIZE + c] = float("-inf")

        if logits.max().item() == float("-inf"):
            board.consecutive_passes += 1
            board.last_move = None
            board.turn = 3 - board.turn
            if board.consecutive_passes >= 2:
                board.game_over = True
            move_num += 1
            continue

        r, c = divmod(int(logits.argmax().item()), SIZE)
        board.place_stone(r, c)
        move_num += 1

    return 1 if get_winner(board)["winner"] == "black" else 2


# ---------------------------------------------------------------------------
# Head-to-head match
# ---------------------------------------------------------------------------

def head_to_head(
    path_a:    str,
    path_b:    str,
    num_games: int = 20,
    device:    str = "cpu",
) -> dict:
    """
    Play num_games between model A and model B, alternating colors.
    Returns win counts and win rate for A.
    """
    model_a = load_model(path_a, device=device)
    model_b = load_model(path_b, device=device)
    wins_a = wins_b = 0

    for i in range(1, num_games + 1):
        if i % 2 == 1:
            winner  = play_game(model_a, model_b, device)
            a_won   = winner == 1
        else:
            winner  = play_game(model_b, model_a, device)
            a_won   = winner == 2

        if a_won:
            wins_a += 1
        else:
            wins_b += 1
        print(f"  game {i}/{num_games} → {'A' if a_won else 'B'} wins  (A {wins_a} – {wins_b} B)")

    return {"wins_a": wins_a, "wins_b": wins_b, "win_rate_a": wins_a / num_games}


# ---------------------------------------------------------------------------
# Tournament across all checkpoints in a folder
# ---------------------------------------------------------------------------

def tournament(folder: str, num_games: int = 10, device: str = "cpu") -> Path:
    """
    Round-robin across all .pth files in folder.
    Returns the path of the model with the most match wins.
    """
    pth_files = sorted(Path(folder).glob("*.pth"))
    if len(pth_files) < 2:
        raise ValueError(f"Need at least 2 .pth files in {folder}, found {len(pth_files)}")

    print(f"Tournament: {len(pth_files)} models, {num_games} games per match\n")
    scores: dict[Path, int] = {p: 0 for p in pth_files}

    for i, path_a in enumerate(pth_files):
        for path_b in pth_files[i + 1:]:
            print(f"{path_a.name}  vs  {path_b.name}")
            result = head_to_head(str(path_a), str(path_b), num_games, device)
            if result["wins_a"] > result["wins_b"]:
                scores[path_a] += 1
                print(f"  → {path_a.name} wins ({result['wins_a']}-{result['wins_b']})\n")
            elif result["wins_b"] > result["wins_a"]:
                scores[path_b] += 1
                print(f"  → {path_b.name} wins ({result['wins_b']}-{result['wins_a']})\n")
            else:
                print(f"  → draw ({result['wins_a']}-{result['wins_b']})\n")

    print("=== Final standings ===")
    for path, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {path.name}: {score} match wins")

    best = max(scores, key=lambda p: scores[p])
    print(f"\nBest model: {best.name}")
    return best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare model checkpoints")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--folder", type=str,
                      help="Run round-robin tournament across all .pth files in this folder")
    mode.add_argument("--black",  type=str,
                      help="Path to model A (use with --white for a direct match)")
    parser.add_argument("--white",   type=str)
    parser.add_argument("--games",   type=int, default=20, help="Games per match")
    parser.add_argument("--device",  type=str, default="cpu")
    parser.add_argument("--promote", type=str, default=None,
                        help="Save best model under this version name, e.g. 'v2'")
    args = parser.parse_args()

    if args.folder:
        best = tournament(args.folder, num_games=args.games, device=args.device)
    else:
        if not args.white:
            parser.error("--black requires --white")
        print(f"A: {args.black}\nB: {args.white}\n")
        result = head_to_head(args.black, args.white, args.games, args.device)
        print(f"\nA win rate: {result['win_rate_a']:.1%}  ({result['wins_a']}-{result['wins_b']})")
        best = Path(args.black) if result["wins_a"] >= result["wins_b"] else Path(args.white)
        print(f"Better model: {best.name}")

    if args.promote:
        dest = Path(best).parent / f"{args.promote}.pth"
        shutil.copy(best, dest)
        print(f"Promoted → {dest}")


if __name__ == "__main__":
    main()
