"""
test_interface.py — Integration tests for the Flask REST API (app.py).

Uses Flask's built-in test client.  The module-level `board` object from
app.py is reset before each test via the `client` fixture.
"""

import json
import pytest
from app import app as flask_app, board


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    board.reset()
    with flask_app.test_client() as c:
        yield c


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def post_json(client, url, data=None):
    return client.post(
        url,
        data=json.dumps(data or {}),
        content_type="application/json",
    )


def get_state(client):
    return client.get("/state").get_json()


# -----------------------------------------------------------------------
# /new_game
# -----------------------------------------------------------------------

def test_new_game_returns_initial_state(client):
    # Place a stone first so state is dirty
    post_json(client, "/move", {"row": 4, "col": 4})
    res = post_json(client, "/new_game")
    assert res.status_code == 200
    data = res.get_json()
    assert data["turn"] == "black"
    assert data["game_over"] is False
    assert data["winner"] is None
    flat = [cell for row in data["board"] for cell in row]
    assert all(c == 0 for c in flat)


# -----------------------------------------------------------------------
# /state
# -----------------------------------------------------------------------

def test_state_returns_correct_shape(client):
    data = get_state(client)
    assert "board" in data
    assert "turn" in data
    assert "captured" in data
    assert "last_move" in data
    assert "game_over" in data
    assert "winner" in data
    assert "scores" in data
    assert len(data["board"]) == 9
    assert all(len(row) == 9 for row in data["board"])


# -----------------------------------------------------------------------
# /move
# -----------------------------------------------------------------------

def test_move_updates_board(client):
    res = post_json(client, "/move", {"row": 3, "col": 3})
    assert res.status_code == 200
    data = res.get_json()
    assert data["board"][3][3] == 1   # black stone placed
    assert data["turn"] == "white"    # turn advanced


def test_move_returns_error_on_illegal_move(client):
    post_json(client, "/move", {"row": 4, "col": 4})  # black
    res = post_json(client, "/move", {"row": 4, "col": 4})  # occupied → illegal
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data


def test_move_returns_error_when_game_over(client):
    # End game via two consecutive passes
    post_json(client, "/pass")
    post_json(client, "/pass")
    res = post_json(client, "/move", {"row": 0, "col": 0})
    assert res.status_code == 400


def test_move_missing_parameters(client):
    res = post_json(client, "/move", {"row": 3})
    assert res.status_code == 400


# -----------------------------------------------------------------------
# /pass
# -----------------------------------------------------------------------

def test_single_pass_advances_turn(client):
    res = post_json(client, "/pass")
    assert res.status_code == 200
    data = res.get_json()
    assert data["turn"] == "white"
    assert data["game_over"] is False


def test_two_consecutive_passes_end_game(client):
    post_json(client, "/pass")   # black passes
    res = post_json(client, "/pass")  # white passes → game over
    assert res.status_code == 200
    data = res.get_json()
    assert data["game_over"] is True
    assert data["winner"] in ("black", "white")
    assert data["scores"] is not None


def test_pass_after_game_over_returns_error(client):
    post_json(client, "/pass")
    post_json(client, "/pass")
    res = post_json(client, "/pass")
    assert res.status_code == 400


# -----------------------------------------------------------------------
# /legal_moves
# -----------------------------------------------------------------------

def test_legal_moves_returns_list(client):
    res = client.get("/legal_moves")
    assert res.status_code == 200
    data = res.get_json()
    assert "legal_moves" in data
    assert isinstance(data["legal_moves"], list)


def test_legal_moves_all_on_empty_board(client):
    res = client.get("/legal_moves")
    data = res.get_json()
    assert len(data["legal_moves"]) == 81


def test_legal_moves_decreases_after_placement(client):
    post_json(client, "/move", {"row": 4, "col": 4})  # black placed
    res = client.get("/legal_moves")
    data = res.get_json()
    assert len(data["legal_moves"]) <= 80


# -----------------------------------------------------------------------
# /ai_move
# -----------------------------------------------------------------------

def test_ai_move_returns_valid_state_or_503(client):
    res = post_json(client, "/ai_move")
    if res.status_code == 503:
        # No trained model present — expected in CI / pre-training
        assert "error" in res.get_json()
    else:
        assert res.status_code == 200
        data = res.get_json()
        assert "board" in data
        assert "turn" in data
