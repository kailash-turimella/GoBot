"""
test_rules.py — Unit tests for engine/rules.py.

Covers boundary checks, occupied-cell detection, Ko, suicide rules, and
get_legal_moves filtering.
"""

import pytest
from go_engine.board import Board, SIZE
from go_engine.rules import is_legal, is_in_bounds, is_suicide, is_ko, get_legal_moves


# -----------------------------------------------------------------------
# Bounds
# -----------------------------------------------------------------------

def test_in_bounds_center():
    assert is_in_bounds(4, 4)


def test_out_of_bounds_negative_row():
    assert not is_in_bounds(-1, 0)


def test_out_of_bounds_negative_col():
    assert not is_in_bounds(0, -1)


def test_out_of_bounds_row_too_large():
    assert not is_in_bounds(SIZE, 0)


def test_out_of_bounds_col_too_large():
    assert not is_in_bounds(0, SIZE)


def test_is_legal_rejects_out_of_bounds():
    board = Board()
    assert not is_legal(board, -1, 0, 1)
    assert not is_legal(board, 0, SIZE, 1)


# -----------------------------------------------------------------------
# Occupied cell
# -----------------------------------------------------------------------

def test_is_legal_rejects_occupied_cell():
    board = Board()
    board.place_stone(4, 4)          # black places
    assert not is_legal(board, 4, 4, 2)  # white can't play there


# -----------------------------------------------------------------------
# Suicide
# -----------------------------------------------------------------------

def test_suicide_in_corner_is_illegal():
    """
    Black stones at (0,1) and (1,0) leave (0,0) with zero liberties for white.
    """
    board = Board()
    board.grid[0][1] = 1  # B
    board.grid[1][0] = 1  # B
    board.turn = 2         # white to play
    assert is_suicide(board, 0, 0, 2)
    assert not is_legal(board, 0, 0, 2)


def test_open_move_is_not_suicide():
    board = Board()
    assert not is_suicide(board, 4, 4, 1)


def test_suicide_allowed_if_it_captures_opponent():
    """
    Black plays at (1,1) surrounded by 4 white stones, each with no other
    liberty.  This looks suicidal but is legal because all 4 white stones
    are captured first, giving the placed stone liberties.

    Board layout (B=1, W=2):
      B W B . .   row 0
      W . W . .   row 1
      B W B . .   row 2
    Black plays at (1,1).
    """
    board = Board()
    board.grid[0][0] = 1
    board.grid[0][1] = 2
    board.grid[0][2] = 1
    board.grid[1][0] = 2
    board.grid[1][2] = 2
    board.grid[2][0] = 1
    board.grid[2][1] = 2
    board.grid[2][2] = 1
    board.turn = 1  # black to play

    assert not is_suicide(board, 1, 1, 1), "Should be legal — captures opponent first"
    assert is_legal(board, 1, 1, 1)


def test_suicide_with_no_captures_remains_illegal():
    """Placing a stone that has no liberties and makes no captures is suicide."""
    board = Board()
    # Surround (1,1) with black stones; white tries to play there
    board.grid[0][1] = 1
    board.grid[1][0] = 1
    board.grid[1][2] = 1
    board.grid[2][1] = 1
    board.turn = 2
    assert is_suicide(board, 1, 1, 2)
    assert not is_legal(board, 1, 1, 2)


# -----------------------------------------------------------------------
# Ko
# -----------------------------------------------------------------------

def test_ko_detected():
    """
    Classic single-stone Ko position.

    State after black captured white at (1,1):
      . B W .   row 0
      B . B W   row 1   ← (1,2) is black stone that just captured
      . B W .   row 2

    previous_state had white at (1,1) and no black at (1,2).
    White playing at (1,1) would capture black at (1,2) and recreate
    the previous_state — a Ko violation.
    """
    board = Board()
    board.grid[0][1] = 1
    board.grid[0][2] = 2
    board.grid[1][0] = 1
    board.grid[1][2] = 1  # stone that just captured
    board.grid[1][3] = 2
    board.grid[2][1] = 1
    board.grid[2][2] = 2
    board.turn = 2  # white's turn

    prev = [[0] * SIZE for _ in range(SIZE)]
    prev[0][1] = 1
    prev[0][2] = 2
    prev[1][0] = 1
    prev[1][1] = 2  # white stone that was captured
    prev[1][3] = 2
    prev[2][1] = 1
    prev[2][2] = 2
    board.previous_state = tuple(tuple(row) for row in prev)

    assert is_ko(board, 1, 1, 2), "Should detect Ko"
    assert not is_legal(board, 1, 1, 2), "Ko move must be illegal"


def test_no_ko_without_previous_state():
    """On the very first move (no previous_state), Ko can never apply."""
    board = Board()
    board.previous_state = None
    assert not is_ko(board, 4, 4, 1)


def test_non_ko_move_is_not_flagged():
    """A move that does not recreate the previous board state is not Ko."""
    board = Board()
    board.grid[0][0] = 1
    board.previous_state = tuple(tuple(row) for row in board.grid)
    # Playing anywhere that doesn't restore previous_state
    board.grid[0][0] = 0  # mutate so state differs from previous
    assert not is_ko(board, 4, 4, 1)


# -----------------------------------------------------------------------
# get_legal_moves
# -----------------------------------------------------------------------

def test_legal_moves_full_empty_board():
    board = Board()
    moves = get_legal_moves(board, 1)
    assert len(moves) == SIZE * SIZE  # all 81 intersections are legal on empty board


def test_legal_moves_excludes_occupied():
    board = Board()
    board.place_stone(4, 4)  # black
    moves = get_legal_moves(board, 2)  # white's legal moves
    assert (4, 4) not in moves
    assert len(moves) == SIZE * SIZE - 1


def test_legal_moves_excludes_suicide():
    board = Board()
    board.grid[0][1] = 2  # W
    board.grid[1][0] = 2  # W
    # (0,0) is suicidal for black (surrounded by W and edges)
    moves = get_legal_moves(board, 1)
    assert (0, 0) not in moves


def test_legal_moves_excludes_ko(monkeypatch):
    """Ko position from above: (1,1) must not appear in white's legal moves."""
    board = Board()
    board.grid[0][1] = 1
    board.grid[0][2] = 2
    board.grid[1][0] = 1
    board.grid[1][2] = 1
    board.grid[1][3] = 2
    board.grid[2][1] = 1
    board.grid[2][2] = 2
    board.turn = 2

    prev = [[0] * SIZE for _ in range(SIZE)]
    prev[0][1] = 1; prev[0][2] = 2; prev[1][0] = 1
    prev[1][1] = 2; prev[1][3] = 2; prev[2][1] = 1; prev[2][2] = 2
    board.previous_state = tuple(tuple(row) for row in prev)

    moves = get_legal_moves(board, 2)
    assert (1, 1) not in moves
