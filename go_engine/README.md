# 9×9 Go — Python/Flask Implementation

A complete browser-playable 9×9 Go game with a Python/Flask backend and
canvas-based HTML/JS frontend. All rules are implemented from scratch:
stone capture, Ko, suicide prevention, and Chinese scoring with komi.

## Project structure

```
go_engine/
├── app.py              Flask server + REST API
├── engine/
│   ├── board.py        Board state, stone placement, capture logic
│   ├── rules.py        Ko, suicide, legal-move validation
│   ├── scoring.py      Chinese scoring, territory flood-fill
│   └── ai.py           AI stub (NotImplementedError — MCTS interface)
├── static/
│   ├── js/game.js      Canvas rendering + click handling
│   └── css/style.css   Board and stone styling
├── templates/
│   └── index.html      Single-page shell
└── tests/              pytest test suite
```

## Setup

### 1 — Create and activate the virtual environment

```bash
cd go_engine
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Run the server

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## Running tests

```bash
pytest tests/ -v
```

## REST API

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Serve `index.html` |
| POST | `/new_game` | Reset board, return initial state |
| POST | `/move` | `{"row": int, "col": int}` — place a stone |
| GET | `/state` | Current board state as JSON |
| POST | `/pass` | Current player passes; two consecutive passes end the game |
| GET | `/legal_moves` | All legal moves for the current player |
| POST | `/ai_move` | 501 Not Implemented (stub for future MCTS) |

### State response shape

```json
{
  "board": [[0, 0, ...], ...],
  "turn": "black",
  "captured": {"black": 0, "white": 0},
  "last_move": [row, col],
  "game_over": false,
  "winner": null,
  "scores": null
}
```

`board` values: `0` = empty, `1` = black, `2` = white.

## Rules implemented

- **Capture**: a group with no liberties after a placement is removed.
- **Suicide**: placing a stone that leaves your own group with zero liberties is illegal, *unless* it simultaneously captures an opponent group (which restores liberties).
- **Ko**: you may not play a move that recreates the board state that existed immediately before your opponent's last move (simple Ko).
- **Scoring**: Chinese rules — stones on board + territory (empty intersections bordered exclusively by one color). White receives 2.5 komi.

## Future AI

`engine/ai.py` defines the `AIPlayer` interface. The docstring explains
exactly how a Monte Carlo Tree Search agent would plug in. Replace the
`raise NotImplementedError` in `select_move` with the MCTS loop.
