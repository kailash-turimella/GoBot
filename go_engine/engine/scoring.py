"""
scoring.py — Territory counting and final score calculation for a 9x9 Go game.

Implements Chinese scoring: a player's score is the number of their stones on
the board plus the number of empty intersections whose only adjacent stones all
belong to that player (territory).

Key design decisions:
  - find_territory uses BFS over connected empty regions.  For each region it
    inspects every adjacent non-empty cell.  If all adjacent stones belong to
    exactly one color, the region is that color's territory.  If both colors
    touch the region, it is contested and counts for neither player.
  - Komi (compensation for white going second) is 2.5 under standard Chinese
    rules.  The fractional 0.5 guarantees no draw.
  - calculate_score and find_territory are pure functions: they read the board
    but never mutate it, so they can be called at any point (e.g., for live
    score estimates before the game ends).
  - The AI would call calculate_score as the terminal evaluation function at
    the leaf nodes of its MCTS simulations.

Known edge cases:
  - Seki (mutual-life groups with shared liberties but no two eyes) is not
    specially handled.  Stones in seki are counted as live; the shared empty
    intersections between them are contested (bordered by both colors), which
    is the correct treatment under Chinese rules.
  - Dead stones inside an opponent's territory are not automatically removed.
    A complete implementation would include a "mark dead stones" phase after
    both players pass before calling calculate_score.
  - An entirely empty board gives territory = 0 for both players because the
    single connected empty region is bordered by no stones at all, making
    border_colors an empty set (len != 1), so it is classified as contested.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.board import Board

SIZE = 9
KOMI = 2.5


def find_territory(board: "Board") -> dict[tuple[int, int], int]:
    """
    Returns a mapping (row, col) -> owner for every empty intersection.
    owner: 1 = black territory, 2 = white territory, 0 = contested.
    """
    territory: dict[tuple[int, int], int] = {}
    visited: set[tuple[int, int]] = set()

    for start_r in range(SIZE):
        for start_c in range(SIZE):
            if board.grid[start_r][start_c] != 0 or (start_r, start_c) in visited:
                continue

            region: list[tuple[int, int]] = []
            border_colors: set[int] = set()
            queue: deque[tuple[int, int]] = deque([(start_r, start_c)])

            while queue:
                r, c = queue.popleft()
                if (r, c) in visited:
                    continue
                visited.add((r, c))
                region.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < SIZE and 0 <= nc < SIZE):
                        continue
                    cell = board.grid[nr][nc]
                    if cell == 0:
                        if (nr, nc) not in visited:
                            queue.append((nr, nc))
                    else:
                        border_colors.add(cell)

            owner = border_colors.pop() if len(border_colors) == 1 else 0
            for pos in region:
                territory[pos] = owner

    return territory


def calculate_score(board: "Board") -> dict[str, float]:
    """
    Chinese scoring: stones on board + territory.
    Returns {"black": float, "white": float}; white's total includes komi.
    """
    territory = find_territory(board)

    black_stones = sum(1 for r in range(SIZE) for c in range(SIZE) if board.grid[r][c] == 1)
    white_stones = sum(1 for r in range(SIZE) for c in range(SIZE) if board.grid[r][c] == 2)
    black_territory = sum(1 for v in territory.values() if v == 1)
    white_territory = sum(1 for v in territory.values() if v == 2)

    return {
        "black": float(black_stones + black_territory),
        "white": float(white_stones + white_territory + KOMI),
    }


def get_winner(board: "Board") -> dict:
    """
    Determine winner by Chinese scoring with 2.5 komi.
    Returns {"winner": str, "scores": {"black": float, "white": float}}.
    """
    scores = calculate_score(board)
    winner = "black" if scores["black"] > scores["white"] else "white"
    return {"winner": winner, "scores": scores}
