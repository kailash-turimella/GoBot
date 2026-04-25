"""
board.py — Core board representation for a 9x9 Go game.

This module owns all mutable game state: the 9×9 grid, whose turn it is,
captured-stone counts, and the single-move-back snapshot needed for Ko
detection.

Key design decisions:
  - Grid encoding: 0 = empty, 1 = black, 2 = white.  Integer values are
    compact, JSON-serializable without any mapping, and fast to compare.
  - Separation of concerns: Board does NOT validate moves.  It trusts its
    callers (rules.py or app.py) to call is_legal first.  This lets rules
    be tested without going through place_stone, and keeps each module small.
  - Captures live inside place_stone because the board is the sole owner of
    the grid.  Rules checks that need to simulate captures work on a deep
    copy (copy.deepcopy) rather than the live board, so there are no
    accidental side effects.
  - previous_state is stored as a tuple-of-tuples (immutable, hashable).
    It records the board BEFORE the last move, which is exactly what
    rules.is_ko needs to compare against the state that would result from a
    candidate move.
  - get_group and _count_liberties use BFS, which avoids Python recursion
    limits that a DFS approach would hit on large boards or tight groups.

Known edge cases:
  - Multiple opponent groups can be captured in a single move (e.g., a
    stone that simultaneously fills the last liberty of two separate groups).
    This is handled correctly because we check all four neighbors before
    removing any stones.
  - Ko detection only looks one move back (simple Ko).  Super-Ko—comparing
    against the full position history—is not implemented.  To add it, change
    previous_state to a set of past snapshots.
  - consecutive_passes and game_over are tracked here so that get_board_state
    can include them in every API response without extra state in app.py.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Optional

SIZE = 9


class Board:
    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.grid: list[list[int]] = [[0] * SIZE for _ in range(SIZE)]
        self.turn: int = 1  # 1 = black goes first
        self.captured: dict[str, int] = {"black": 0, "white": 0}
        self.previous_state: Optional[tuple] = None
        self.last_move: Optional[tuple[int, int]] = None
        self.consecutive_passes: int = 0
        self.game_over: bool = False
        self.winner: Optional[str] = None
        self.scores: Optional[dict] = None

    # ------------------------------------------------------------------
    # Core mutation
    # ------------------------------------------------------------------

    def place_stone(self, row: int, col: int) -> bool:
        """
        Place the current player's stone at (row, col).

        Assumes the caller has already verified legality with rules.is_legal.
        Performs opponent captures, updates the turn, and resets consecutive
        passes.  Returns False if the cell is non-empty (last-resort guard).
        """
        if self.grid[row][col] != 0:
            return False

        self.previous_state = self._snapshot()

        self.grid[row][col] = self.turn
        opponent = 3 - self.turn

        captured_count = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < SIZE and 0 <= nc < SIZE and self.grid[nr][nc] == opponent:
                group = self.get_group(nr, nc)
                if self._count_liberties(group) == 0:
                    for gr, gc in group:
                        self.grid[gr][gc] = 0
                        captured_count += 1

        capturer = "black" if self.turn == 1 else "white"
        self.captured[capturer] += captured_count
        self.last_move = (row, col)
        self.consecutive_passes = 0
        self.turn = opponent
        return True

    # ------------------------------------------------------------------
    # Group / liberty queries (used by rules.py and capture logic)
    # ------------------------------------------------------------------

    def get_group(self, row: int, col: int) -> set[tuple[int, int]]:
        """BFS flood-fill: all stones connected to (row, col) of the same color."""
        color = self.grid[row][col]
        visited: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque([(row, col)])
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < SIZE
                    and 0 <= nc < SIZE
                    and self.grid[nr][nc] == color
                    and (nr, nc) not in visited
                ):
                    queue.append((nr, nc))
        return visited

    def _count_liberties(self, group: set[tuple[int, int]]) -> int:
        """Count distinct empty intersections adjacent to every stone in group."""
        liberties: set[tuple[int, int]] = set()
        for r, c in group:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < SIZE and 0 <= nc < SIZE and self.grid[nr][nc] == 0:
                    liberties.add((nr, nc))
        return len(liberties)

    def get_liberties(self, row: int, col: int) -> int:
        """Liberty count for the group containing (row, col). Returns 0 for empty cells."""
        if self.grid[row][col] == 0:
            return 0
        return self._count_liberties(self.get_group(row, col))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _snapshot(self) -> tuple:
        """Immutable grid snapshot for Ko comparison."""
        return tuple(tuple(row) for row in self.grid)

    def get_board_state(self) -> dict:
        """Full serializable state dict for JSON responses."""
        return {
            "board": [row[:] for row in self.grid],
            "turn": "black" if self.turn == 1 else "white",
            "captured": dict(self.captured),
            "last_move": list(self.last_move) if self.last_move else None,
            "game_over": self.game_over,
            "winner": self.winner,
            "scores": self.scores,
        }
