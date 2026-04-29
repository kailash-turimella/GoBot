"""
self_play.py — Generate self-play games using network-guided PUCT MCTS.

Run after supervised.py has produced models/best_model.pth.
Each game yields (state_planes, mcts_probs, outcome) triples that are
appended to data/replay_buffer.npz and consumed by train.py.

MCTS uses the PUCT formula:
    score(child) = Q(child) + c_puct · P(child) · √N(parent) / (1 + N(child))

where Q is the expected value from the CURRENT NODE'S player's perspective
(signs alternate going up the tree) and P is the network policy prior.

Usage:
    python -m training.self_play --games 100 --sims 400 --model models/best_model.pth
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from go_engine.board import Board, SIZE
from go_engine.network import GoNetwork, load_model, NUM_INPUT_PLANES
from go_engine.rules import is_legal
from go_engine.scoring import get_winner

MAX_GAME_MOVES  = 200
TEMP_THRESHOLD  = 30    # use temperature=1 for first 30 moves, then argmax
DIR_ALPHA       = 0.3   # Dirichlet concentration for 9×9
DIR_EPS         = 0.25  # fraction of Dirichlet noise mixed into root priors


# ---------------------------------------------------------------------------
# Board encoding with history
# ---------------------------------------------------------------------------

def encode_board(
    board: Board,
    color: int,
    history: deque,   # deque of (grid_snapshot, color) — newest at index 0
) -> np.ndarray:
    """
    (NUM_INPUT_PLANES, 9, 9) float32.

    Planes 0–7:  current player's stones at T, T-1, …, T-7
    Planes 8–15: opponent's stones at T, T-1, …, T-7
    Plane 16:    1.0 if current player is black, else 0.0
    """
    planes   = np.zeros((NUM_INPUT_PLANES, SIZE, SIZE), dtype=np.float32)
    opponent = 3 - color
    snapshots = list(history)

    for t, (grid, _) in enumerate(snapshots[:8]):
        for r in range(SIZE):
            for c in range(SIZE):
                v = grid[r][c]
                if v == color:
                    planes[t, r, c] = 1.0
                elif v == opponent:
                    planes[t + 8, r, c] = 1.0

    planes[16] = 1.0 if color == 1 else 0.0
    return planes


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = (
        "color", "move", "parent",
        "children", "visits", "total_value", "prior",
        "_candidates",
    )

    def __init__(
        self,
        color: int,
        move: Optional[tuple[int, int]] = None,
        parent: Optional[_Node] = None,
        prior: float = 0.0,
    ) -> None:
        self.color       = color
        self.move        = move
        self.parent      = parent
        self.children: list[_Node] = []
        self.visits      = 0
        self.total_value = 0.0
        self.prior       = prior
        self._candidates: list[tuple[int, int]] = []

    @property
    def q(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0

    def puct(self, c_puct: float) -> float:
        parent_n = self.parent.visits if self.parent else 1
        # Q is from THIS node's player's perspective; parent selects by
        # maximising its own value = -child.q, so parent calls -child.puct().
        return self.q + c_puct * self.prior * math.sqrt(parent_n) / (1 + self.visits)


class _MCTS:
    def __init__(
        self,
        model:    GoNetwork,
        device:   str   = "cpu",
        num_sims: int   = 400,
        c_puct:   float = 1.5,
    ) -> None:
        self.model    = model
        self.device   = device
        self.num_sims = num_sims
        self.c_puct   = c_puct

    def _net_eval(
        self,
        board:   Board,
        color:   int,
        history: deque,
    ) -> tuple[np.ndarray, float]:
        """Returns (masked_probs [81], value) from color's perspective."""
        planes = encode_board(board, color, history)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(x)

        probs = F.softmax(logits.squeeze(0), dim=0).cpu().numpy()

        mask = np.array([
            1.0 if is_legal(board, r, c, color) else 0.0
            for r in range(SIZE) for c in range(SIZE)
        ], dtype=np.float32)
        probs = probs * mask
        s = probs.sum()
        probs = probs / s if s > 0 else mask / mask.sum() if mask.sum() > 0 \
                else np.ones(SIZE * SIZE, dtype=np.float32) / (SIZE * SIZE)

        return probs, float(value.item())

    def _expand_root(
        self,
        root:    _Node,
        board:   Board,
        history: deque,
        add_noise: bool,
    ) -> None:
        probs, _ = self._net_eval(board, root.color, history)
        if add_noise:
            noise = np.random.dirichlet([DIR_ALPHA] * (SIZE * SIZE))
            probs = (1 - DIR_EPS) * probs + DIR_EPS * noise

        for r in range(SIZE):
            for c in range(SIZE):
                if not is_legal(board, r, c, root.color):
                    continue
                child = _Node(
                    color=3 - root.color,
                    move=(r, c),
                    parent=root,
                    prior=float(probs[r * SIZE + c]),
                )
                root.children.append(child)

    def _select(self, node: _Node) -> _Node:
        # Descend until we find a node with unexpanded candidates or a leaf
        while node.children:
            # Parent selects the child that maximises the parent's value.
            # child.q is from child's perspective = -parent's perspective,
            # so parent maximises -child.q + exploration = min child.q - expl.
            # We implement this by negating: pick child with lowest -puct = max puct
            # but negate q to flip perspectives.
            node = max(
                node.children,
                key=lambda n: -n.q + self.c_puct * n.prior
                              * math.sqrt(node.visits or 1) / (1 + n.visits),
            )
        return node

    def _backprop(self, node: _Node, value: float) -> None:
        """value is from node.color's perspective; sign alternates going up."""
        current = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            value = -value
            current = current.parent

    def search(
        self,
        board:     Board,
        color:     int,
        history:   deque,
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Run MCTS and return visit-count distribution π over 81 moves.
        Moves with no legal child have π=0.
        """
        root = _Node(color=color)
        self._expand_root(root, board, history, add_noise)

        if not root.children:
            return np.zeros(SIZE * SIZE, dtype=np.float32)

        # Store board states alongside nodes using a separate dict keyed by id
        board_map: dict[int, Board] = {id(root): copy.deepcopy(board)}
        for child in root.children:
            cb = copy.deepcopy(board)
            cb.place_stone(*child.move)
            board_map[id(child)] = cb

        for _ in range(self.num_sims):
            # --- Selection ---
            node = self._select(root)

            # --- Expansion + Evaluation ---
            node_board = board_map[id(node)]
            if node_board.game_over or node is root:
                # Terminal or unexpanded root: evaluate in place
                probs, value = self._net_eval(node_board, node.color, history)
            else:
                # Expand one child
                probs, _ = self._net_eval(node_board, node.color, history)
                if probs.sum() == 0:
                    # No legal moves from this state — evaluate directly
                    _, value = self._net_eval(node_board, node.color, history)
                else:
                    # Pick a random legal move to expand
                    legal = [
                        (r, c) for r in range(SIZE) for c in range(SIZE)
                        if probs[r * SIZE + c] > 0
                        and not any(
                            ch.move == (r, c) for ch in node.children
                        )
                    ]
                    if not legal:
                        _, value = self._net_eval(node_board, node.color, history)
                    else:
                        r, c = random.choice(legal)
                        child = _Node(
                            color=3 - node.color,
                            move=(r, c),
                            parent=node,
                            prior=float(probs[r * SIZE + c]),
                        )
                        node.children.append(child)
                        cb = copy.deepcopy(node_board)
                        cb.place_stone(r, c)
                        board_map[id(child)] = cb
                        _, value = self._net_eval(cb, child.color, history)
                        value = -value  # flip: we got value from child's perspective
                        node = child

            # --- Backprop ---
            self._backprop(node, value)

        visits = np.zeros(SIZE * SIZE, dtype=np.float32)
        for child in root.children:
            r, c = child.move
            visits[r * SIZE + c] = child.visits
        s = visits.sum()
        return visits / s if s > 0 else visits


# ---------------------------------------------------------------------------
# Self-play game
# ---------------------------------------------------------------------------

class SelfPlayGame:
    def __init__(self, mcts: _MCTS, temp_threshold: int = TEMP_THRESHOLD) -> None:
        self.mcts           = mcts
        self.temp_threshold = temp_threshold

    def run(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """
        Play one complete game.
        Returns (planes, pi, outcome) per position.
        outcome = +1.0 if the player to move at that position won, -1.0 if not.
        """
        board   = Board()
        history: deque = deque(maxlen=8)
        history.appendleft(([row[:] for row in board.grid], board.turn))

        records: list[tuple[np.ndarray, np.ndarray, int]] = []
        move_num = 0

        while not board.game_over and move_num < MAX_GAME_MOVES:
            color = board.turn
            planes = encode_board(board, color, history)
            pi     = self.mcts.search(board, color, history, add_noise=True)

            if pi.sum() == 0:
                # No legal moves — pass
                board.consecutive_passes += 1
                board.last_move = None
                board.turn = 3 - board.turn
                if board.consecutive_passes >= 2:
                    board.game_over = True
                records.append((planes, pi, color))
                move_num += 1
                continue

            # Temperature-controlled move selection
            if move_num < self.temp_threshold:
                p = pi / pi.sum()
                move_idx = int(np.random.choice(SIZE * SIZE, p=p))
            else:
                move_idx = int(np.argmax(pi))

            records.append((planes, pi, color))

            r, c = divmod(move_idx, SIZE)
            board.place_stone(r, c)
            history.appendleft(([row[:] for row in board.grid], board.turn))
            move_num += 1

        result = get_winner(board)
        winner_color = 1 if result["winner"] == "black" else 2

        return [
            (planes, pi, 1.0 if color == winner_color else -1.0)
            for planes, pi, color in records
        ]


# ---------------------------------------------------------------------------
# Replay buffer I/O
# ---------------------------------------------------------------------------

def load_buffer(path: str) -> list:
    p = Path(path)
    if not p.exists():
        return []
    data = np.load(p, allow_pickle=True)
    return list(zip(data["planes"], data["probs"], data["outcomes"]))


def save_buffer(examples: list, path: str) -> None:
    planes   = np.array([e[0] for e in examples], dtype=np.float32)
    probs    = np.array([e[1] for e in examples], dtype=np.float32)
    outcomes = np.array([e[2] for e in examples], dtype=np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    existing = load_buffer(path)
    if existing:
        planes   = np.concatenate([np.array([e[0] for e in existing]), planes])
        probs    = np.concatenate([np.array([e[1] for e in existing]), probs])
        outcomes = np.concatenate([np.array([e[2] for e in existing]), outcomes])
    np.savez(path, planes=planes, probs=probs, outcomes=outcomes)
    print(f"Buffer: {len(outcomes)} total positions → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate self-play games")
    parser.add_argument("--games",  type=int, default=100)
    parser.add_argument("--sims",   type=int, default=400)
    parser.add_argument("--model",  type=str, default="models/best_model.pth")
    parser.add_argument("--buffer", type=str, default="data/replay_buffer.npz")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"Loading model from {args.model}")
    model = load_model(args.model, device=args.device)
    mcts  = _MCTS(model, device=args.device, num_sims=args.sims)
    game  = SelfPlayGame(mcts)

    all_examples: list = []
    for i in range(1, args.games + 1):
        examples = game.run()
        all_examples.extend(examples)
        print(f"game {i}/{args.games}: {len(examples)} positions")

    save_buffer(all_examples, args.buffer)


if __name__ == "__main__":
    main()
