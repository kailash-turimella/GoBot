"""
app.py — Flask server and REST API for the 9x9 Go game.
"""

from flask import Flask, jsonify, request, render_template

from engine.board import Board
from engine.rules import is_legal, get_legal_moves
from engine.scoring import get_winner

app = Flask(__name__)
board = Board()


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
    return jsonify(board.get_board_state())


@app.route("/state", methods=["GET"])
def state():
    return jsonify(board.get_board_state())


@app.route("/pass", methods=["POST"])
def pass_turn():
    if board.game_over:
        return jsonify({"error": "Game is already over"}), 400

    board.consecutive_passes += 1
    board.last_move = None
    board.turn = 3 - board.turn

    if board.consecutive_passes >= 2:
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
    return jsonify({"error": "AI not yet implemented"}), 501


if __name__ == "__main__":
    app.run(debug=True)
