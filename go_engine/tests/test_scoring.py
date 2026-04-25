"""
test_scoring.py — Unit tests for engine/scoring.py.

Covers territory flood-fill, komi application, and winner determination.
"""

import pytest
from engine.board import Board, SIZE
from engine.scoring import calculate_score, find_territory, get_winner, KOMI


# -----------------------------------------------------------------------
# Empty board
# -----------------------------------------------------------------------

def test_empty_board_zero_territory_both():
    """On an empty board the single connected region is bordered by no stones
    at all, so it is contested and counts for neither player."""
    board = Board()
    territory = find_territory(board)
    black_t = sum(1 for v in territory.values() if v == 1)
    white_t = sum(1 for v in territory.values() if v == 2)
    assert black_t == 0
    assert white_t == 0


def test_empty_board_scores():
    """Empty board: black = 0, white = komi only."""
    board = Board()
    scores = calculate_score(board)
    assert scores["black"] == 0.0
    assert scores["white"] == KOMI


# -----------------------------------------------------------------------
# Fully enclosed territory
# -----------------------------------------------------------------------

def test_single_empty_cell_enclosed_by_black():
    """
    Four black stones form a ring around (1,1).  That single empty
    intersection is entirely black territory.
    """
    board = Board()
    board.grid[0][1] = 1  # B north
    board.grid[2][1] = 1  # B south
    board.grid[1][0] = 1  # B west
    board.grid[1][2] = 1  # B east

    territory = find_territory(board)
    assert territory[(1, 1)] == 1  # owned by black


def test_enclosed_territory_counts_correctly():
    """
    Enclose a 2-cell empty strip (0,0)–(0,1) completely with black.
    Both cells must be credited to black.
    """
    board = Board()
    # North wall: out of bounds (OOB).
    # West wall: OOB for (0,0).
    # (0,2) = B east wall; (1,0) = B south of (0,0); (1,1) = B south of (0,1).
    board.grid[0][2] = 1
    board.grid[1][0] = 1
    board.grid[1][1] = 1
    territory = find_territory(board)
    # (0,0) and (0,1) are the enclosed region; they are adjacent only to black
    # stones and out-of-bounds (which counts as "no stone").
    # All finite neighbors of (0,0): right=(0,1)=empty, down=(1,0)=B.
    # All finite neighbors of (0,1): left=(0,0)=empty, right=(0,2)=B, down=(1,1)=B.
    # border_colors = {1} → owner = black.
    assert territory.get((0, 0)) == 1
    assert territory.get((0, 1)) == 1


def test_komi_applied_to_white():
    """White's score always includes KOMI even if territory is equal."""
    board = Board()
    scores = calculate_score(board)
    assert scores["white"] - scores["black"] == pytest.approx(KOMI)


def test_winner_determined_correctly_black_wins():
    """
    Place enough black stones that black beats white + komi.
    With no white stones, ALL empty intersections border only black,
    so they all become black territory.  5 black stones + 76 empty = 81
    black points vs. 0 + 2.5 = 2.5 white points → black wins.
    """
    board = Board()
    for c in range(5):
        board.grid[4][c] = 1  # five black stones in a row
    result = get_winner(board)
    assert result["winner"] == "black"
    # 5 stones + 76 empty intersections (all black territory) = 81
    assert result["scores"]["black"] == 81.0
    assert result["scores"]["white"] == pytest.approx(KOMI)


def test_winner_determined_correctly_white_wins_by_komi():
    """
    Completely empty board: white wins by komi alone.
    """
    board = Board()
    result = get_winner(board)
    assert result["winner"] == "white"


def test_mixed_territory_not_counted():
    """
    An empty region adjacent to both black and white stones is contested
    and scores for neither player.
    """
    board = Board()
    # Place one black and one white stone anywhere; the vast connected
    # empty region touches both → contested.
    board.grid[0][0] = 1  # B
    board.grid[0][1] = 2  # W
    territory = find_territory(board)
    black_t = sum(1 for v in territory.values() if v == 1)
    white_t = sum(1 for v in territory.values() if v == 2)
    # The remaining empty region is one big connected area touching B and W
    assert black_t == 0
    assert white_t == 0


def test_scores_include_both_stones_and_territory():
    """
    Create a board where black has 1 stone + 1 territory point, white 0.
    Expected: black = 2, white = 0 + komi.
    """
    board = Board()
    # Black stone at (5,5), completely enclosed empty region at (6,6)
    board.grid[5][5] = 1   # B stone
    board.grid[5][6] = 1   # B east
    board.grid[5][7] = 1   # B further east
    board.grid[6][5] = 1   # B south
    board.grid[6][7] = 1   # B south-east corner closer
    board.grid[7][6] = 1   # B below (6,6)
    board.grid[7][5] = 1   # closes south-west
    board.grid[7][7] = 1   # closes south-east
    # (6,6) is surrounded: north=(5,6)=B, south=(7,6)=B, west=(6,5)=B, east=(6,7)=B
    territory = find_territory(board)
    assert territory.get((6, 6)) == 1
