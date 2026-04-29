"""
eval.py — Pit a new checkpoint against the current best model.

Protocol (AlphaGo Zero style):
  - Play NUM_EVAL_GAMES games, alternating colours.
  - If new model wins > WIN_THRESHOLD fraction, it becomes the new best.
  - Winning model is copied to models/best_model.pth.

Usage:
    python -m training.eval \
        --candidate models/model_v5.pth \
        --champion  models/best_model.pth \
        --games 20
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Optional

# from go_engine.network import GoNetwork, load_model
# from go_engine.board import Board
# from go_engine.rules import get_legal_moves
# from go_engine.scoring import get_winner

NUM_EVAL_GAMES  = 20
WIN_THRESHOLD   = 0.55   # new model must win >55% to replace champion


# ---------------------------------------------------------------------------
# Evaluation game
# ---------------------------------------------------------------------------

def play_eval_game(model_black, model_white, num_simulations: int = 200) -> int:
    """
    Play one game between two models.

    Returns:
        1  if black wins
        2  if white wins
        0  if draw (rare in Go with komi)
    """
    raise NotImplementedError("play_eval_game not yet implemented")


# ---------------------------------------------------------------------------
# Head-to-head evaluation
# ---------------------------------------------------------------------------

def evaluate(
    candidate_path: str,
    champion_path:  str,
    num_games:      int = NUM_EVAL_GAMES,
    num_simulations: int = 200,
    device: str = "cpu",
) -> bool:
    """
    Run `num_games` games between candidate and current champion.

    Returns True if candidate wins enough games to replace champion,
    and copies candidate → models/best_model.pth.
    """
    raise NotImplementedError("evaluate() not yet implemented")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a new checkpoint")
    parser.add_argument("--candidate", type=str, required=True,
                        help="Path to new checkpoint (.pth)")
    parser.add_argument("--champion",  type=str,
                        default="models/best_model.pth")
    parser.add_argument("--games",     type=int, default=NUM_EVAL_GAMES)
    parser.add_argument("--sims",      type=int, default=200)
    parser.add_argument("--device",    type=str, default="cpu")
    args = parser.parse_args()

    raise NotImplementedError(
        "eval.py is a placeholder. "
        "Implement play_eval_game() and evaluate() first."
    )


if __name__ == "__main__":
    main()
