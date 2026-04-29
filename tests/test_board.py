"""
test_board.py — Unit tests for engine/board.py.

Covers initial state, turn order, stone placement, capture logic, and group
capture counts.
"""

import pytest
from go_engine.board import Board, SIZE


# -----------------------------------------------------------------------
# Initial state
# -----------------------------------------------------------------------

def test_initial_board_all_zeros():
    board = Board()
    for r in range(SIZE):
        for c in range(SIZE):
            assert board.grid[r][c] == 0


def test_black_plays_first():
    board = Board()
    assert board.turn == 1  # 1 = black


def test_initial_captured_counts_zero():
    board = Board()
    assert board.captured["black"] == 0
    assert board.captured["white"] == 0


# -----------------------------------------------------------------------
# Stone placement
# -----------------------------------------------------------------------

def test_stone_placed_at_correct_coordinate():
    board = Board()
    board.place_stone(3, 4)
    assert board.grid[3][4] == 1  # black placed


def test_turn_alternates_after_move():
    board = Board()
    assert board.turn == 1
    board.place_stone(0, 0)
    assert board.turn == 2  # now white
    board.place_stone(1, 1)
    assert board.turn == 1  # back to black


def test_cannot_place_on_occupied_cell():
    board = Board()
    board.place_stone(4, 4)  # black at (4,4)
    result = board.place_stone(4, 4)  # attempt to overwrite
    assert result is False
    assert board.grid[4][4] == 1  # unchanged


def test_last_move_recorded():
    board = Board()
    board.place_stone(2, 3)
    assert board.last_move == (2, 3)


# -----------------------------------------------------------------------
# Captures
# -----------------------------------------------------------------------

def test_single_stone_captured():
    """
    White stone at (0,0) surrounded by black on all sides.
    After black plays the final surrounding move, white at (0,0) is gone.

    Layout before final black move (black=1, white=2):
      W B .   row 0
      B . .   row 1
    Black plays (1,1) — wait, let's use a corner squeeze instead.

    Simpler: place white at (4,4), surround with black on all 4 sides.
    """
    board = Board()
    # Place white at centre (4,4) — need to alternate turns
    # Turn: B W B W B
    board.place_stone(0, 0)  # B somewhere far away
    board.place_stone(4, 4)  # W at target
    board.place_stone(3, 4)  # B north
    board.place_stone(8, 8)  # W far away (just to alternate)
    board.place_stone(5, 4)  # B south
    board.place_stone(8, 7)  # W far away
    board.place_stone(4, 3)  # B west
    board.place_stone(8, 6)  # W far away
    board.place_stone(4, 5)  # B east — this captures white at (4,4)

    assert board.grid[4][4] == 0  # white removed
    assert board.captured["black"] == 1


def test_captured_stones_removed_from_board():
    """A group of two white stones surrounded by black is fully removed."""
    board = Board()
    # Place white at (0,0) and (0,1), then surround them with black.
    # Sequence chosen to satisfy strict turn alternation:
    # Initial turn: B(1)
    board.place_stone(1, 0)  # B south of (0,0)
    board.place_stone(0, 0)  # W
    board.place_stone(1, 1)  # B south of (0,1)
    board.place_stone(0, 1)  # W — group {(0,0),(0,1)}
    board.place_stone(0, 2)  # B east of (0,1)
    board.place_stone(8, 8)  # W elsewhere
    # Only liberty left for white group is at (-1,x) — out of bounds — and
    # (0,-1) — out of bounds.  White group actually has no liberties after
    # black at (0,2) because (0,0) liberties: north OOB, west OOB, south=B,
    # east=(0,1)=W(internal). (0,1) liberties: north OOB, east=(0,2)=B,
    # south=B, west=(0,0) internal. → 0 liberties → already captured after (0,2).
    assert board.grid[0][0] == 0
    assert board.grid[0][1] == 0
    assert board.captured["black"] == 2


def test_group_captured_when_fully_surrounded():
    """Four white stones in a line captured when black removes last liberty."""
    board = Board()
    # White chain along row 1: (1,0),(1,1),(1,2),(1,3)
    # Black will surround from row 0, row 2, and the ends.
    # Turn sequence: B W B W B W B W B W B W B
    moves = [
        (0, 0, 1), (1, 0, 2),
        (0, 1, 1), (1, 1, 2),
        (0, 2, 1), (1, 2, 2),
        (0, 3, 1), (1, 3, 2),  # white chain complete
        (2, 0, 1), (8, 8, 2),
        (2, 1, 1), (8, 7, 2),
        (2, 2, 1), (8, 6, 2),
        (2, 3, 1), (8, 5, 2),  # south side closed
    ]
    b = Board()
    for r, c, expected_color in moves:
        assert b.turn == expected_color, f"Expected turn {expected_color} but got {b.turn} at ({r},{c})"
        b.place_stone(r, c)

    # Last liberty of the white chain is at (1,-1) OOB and (1,4).
    # Place black at (1,4) to capture.
    assert b.turn == 1  # black's turn
    b.place_stone(1, 4)

    for c in range(4):
        assert b.grid[1][c] == 0, f"White at (1,{c}) should have been captured"
    assert b.captured["black"] == 4


def test_capture_count_increments_correctly():
    board = Board()
    # Capture one white stone, check count goes from 0 to 1.
    board.place_stone(3, 4)   # B
    board.place_stone(4, 4)   # W target
    board.place_stone(4, 3)   # B west
    board.place_stone(8, 0)   # W elsewhere
    board.place_stone(4, 5)   # B east
    board.place_stone(8, 1)   # W elsewhere
    board.place_stone(3, 4)   # already occupied — skip, use different coords
    # Redo: use a fresh board with a clean capture scenario
    b = Board()
    b.place_stone(0, 1)  # B north
    b.place_stone(0, 0)  # W at corner
    b.place_stone(1, 0)  # B west — captures (0,0) which has 0 lib
    assert b.captured["black"] == 1
    b.place_stone(8, 8)  # W somewhere
    b.place_stone(5, 5)  # B somewhere
    assert b.captured["black"] == 1  # still 1, no new captures


def test_reset_clears_all_state():
    board = Board()
    board.place_stone(0, 0)
    board.place_stone(1, 1)
    board.reset()
    assert board.turn == 1
    assert board.captured == {"black": 0, "white": 0}
    assert board.last_move is None
    assert board.game_over is False
    for r in range(SIZE):
        for c in range(SIZE):
            assert board.grid[r][c] == 0
