"""
rules.py — Move legality enforcement for a 9x9 Go game.

This module is a pure-function layer over Board.  No function here mutates
the live board; all simulation is done on deep copies so callers can probe
legality at will without side effects.

Key design decisions:
  - is_legal is the single public entry point for move validation.  It chains
    individual checks in the order that matters for correctness:
      bounds → occupied → suicide → ko
    Suicide must be evaluated AFTER simulating captures because a move that
    looks suicidal can be legal if it first captures an opponent group whose
    removal opens liberties for the placed stone.
  - Ko detection compares the board snapshot that would result from a move
    against board.previous_state (set by Board.place_stone on the last turn).
    This implements simple Ko.  Super-Ko would require Board to store a list
    of all past snapshots and compare against each of them.
  - None of these functions advance board.turn, so they can safely be called
    for either color without touching game state.
  - get_legal_moves iterates all 81 intersections and filters through
    is_legal.  For a 9×9 board this is fast enough (≤81 deep-copy checks);
    a larger board or tighter performance budget would warrant caching.

Known edge cases:
  - A "capture-suicide" (placing a stone that has zero liberties but
    simultaneously captures an opponent group, which then frees liberties)
    is correctly classified as LEGAL because is_suicide simulates captures
    first and then checks the resulting liberties.
  - An empty board has previous_state = None, so is_ko short-circuits to
    False and never compares snapshots.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.board import Board

SIZE = 9


def is_in_bounds(row: int, col: int) -> bool:
    return 0 <= row < SIZE and 0 <= col < SIZE


def is_legal(board: "Board", row: int, col: int, color: int) -> bool:
    """Master legality check.  Returns True iff the move is fully valid."""
    if not is_in_bounds(row, col):
        return False
    if board.grid[row][col] != 0:
        return False
    if is_suicide(board, row, col, color):
        return False
    if is_ko(board, row, col, color):
        return False
    return True


def is_suicide(board: "Board", row: int, col: int, color: int) -> bool:
    """
    Returns True if placing color at (row, col) would leave the resulting
    group with zero liberties after all opponent captures are resolved.
    """
    sim = copy.deepcopy(board)
    sim.grid[row][col] = color
    opponent = 3 - color

    # Simulate opponent captures caused by this placement
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < SIZE and 0 <= nc < SIZE and sim.grid[nr][nc] == opponent:
            group = sim.get_group(nr, nc)
            if sim._count_liberties(group) == 0:
                for gr, gc in group:
                    sim.grid[gr][gc] = 0

    return sim._count_liberties(sim.get_group(row, col)) == 0


def is_ko(board: "Board", row: int, col: int, color: int) -> bool:
    """
    Returns True if placing color at (row, col) would recreate the board
    state that existed one move ago (simple Ko rule).
    """
    if board.previous_state is None:
        return False

    sim = copy.deepcopy(board)
    sim.grid[row][col] = color
    opponent = 3 - color

    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < SIZE and 0 <= nc < SIZE and sim.grid[nr][nc] == opponent:
            group = sim.get_group(nr, nc)
            if sim._count_liberties(group) == 0:
                for gr, gc in group:
                    sim.grid[gr][gc] = 0

    return sim._snapshot() == board.previous_state


def get_legal_moves(board: "Board", color: int) -> list[tuple[int, int]]:
    """Return every legal (row, col) for color on the current board."""
    return [
        (r, c)
        for r in range(SIZE)
        for c in range(SIZE)
        if is_legal(board, r, c, color)
    ]
