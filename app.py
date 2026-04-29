"""
app.py — Flask server and REST API for the 9x9 Go game.

All game state lives in a single module-level Board instance.  This is
intentionally simple: Go is a two-player turn-based game with no concurrent
requests in the intended single-session use case.  A multi-session deployment
would move state into a server-side session or database.

Key design decisions:
  - Every route returns JSON.  Error responses carry an "error" key and an
    appropriate HTTP status code so the frontend can distinguish success from
    failure without inspecting body text.
  - The engine (board, rules, scoring) is imported directly; Flask knows
    nothing about Go rules.  This means all engine logic can be tested with
    pytest without starting the server.
  - /pass increments consecutive_passes on the Board and switches the turn.
    When two consecutive passes occur the scoring module determines the winner
    and the result is written back to the Board so every subsequent /state
    call reflects game over.
  - /ai_move runs MCTS via AIPlayer and applies the chosen move.

Known edge cases or future work:
  - concurrent players would require per-session Board instances (Flask
    sessions or a keyed store).
  - The board is not persisted across server restarts; add a /save and /load
    route (or SQLite serialization) if persistence is needed.
"""

from flask import Flask, jsonify, request, render_template

from engine.board import Board
from engine.rules import is_legal, get_legal_moves
from engine.scoring import get_winner
from engine.ai import AIPlayer, ModelNotFoundError

MAX_MOVES = 150  # 9×9 games rarely exceed 80 moves; 150 catches runaway loops

app = Flask(__name__)
board = Board()

try:
    _ai = AIPlayer()
except ModelNotFoundError as _e:
    _ai = None
    print(f"[warning] AI disabled: {_e}")


def _check_move_limit():
    """End the game by scoring if the move cap is reached."""
    if not board.game_over and board.move_count >= MAX_MOVES:
        result = get_winner(board)
        board.game_over = True
        board.winner = result["winner"]
        board.scores = result["scores"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/new_game", methods=["POST"])
def new_game():
    board.reset()
    return jsonify(board.get_board_state())


@app.route("/move", methods=["POST"])
def move():
    data = request.get_json(force=True) or {}
    row = data.get("row")
    col = data.get("col")

    if row is None or col is None:
        return jsonify({"error": "Missing row or col"}), 400

    if board.game_over:
        return jsonify({"error": "Game is already over"}), 400

    if not is_legal(board, int(row), int(col), board.turn):
        return jsonify({"error": "Illegal move"}), 400

    board.place_stone(int(row), int(col))
    _check_move_limit()
    return jsonify(board.get_board_state())


@app.route("/state", methods=["GET"])
def state():
    return jsonify(board.get_board_state())


@app.route("/pass", methods=["POST"])
def pass_turn():
    if board.game_over:
        return jsonify({"error": "Game is already over"}), 400

    passing_player = board.turn
    board.consecutive_passes += 1
    board.player_passes[passing_player] += 1
    board.last_move = None
    board.turn = 3 - board.turn

    if board.consecutive_passes >= 2 or board.player_passes[passing_player] >= 3:
        result = get_winner(board)
        board.game_over = True
        board.winner = result["winner"]
        board.scores = result["scores"]

    return jsonify(board.get_board_state())


@app.route("/legal_moves", methods=["GET"])
def legal_moves():
    moves = get_legal_moves(board, board.turn)
    return jsonify({"legal_moves": [list(m) for m in moves]})


@app.route("/ai_move", methods=["POST"])
def ai_move():
    if _ai is None:
        return jsonify({"error": "No trained model found. Train the network first and save it to models/v1.pth."}), 503

    if board.game_over:
        return jsonify({"error": "Game is already over"}), 400

    move = _ai.select_move(board, board.turn)

    if move is None:
        passing_player = board.turn
        board.consecutive_passes += 1
        board.player_passes[passing_player] += 1
        board.last_move = None
        board.turn = 3 - board.turn
        if board.consecutive_passes >= 2 or board.player_passes[passing_player] >= 3:
            result = get_winner(board)
            board.game_over = True
            board.winner    = result["winner"]
            board.scores    = result["scores"]
    else:
        board.place_stone(*move)
        _check_move_limit()

    return jsonify(board.get_board_state())


if __name__ == "__main__":
    app.run(debug=True)
