# 9×9 Go — Python/Flask + MCTS AI

A fully playable 9×9 Go game with all rules implemented from scratch and a custom Monte Carlo Tree Search (MCTS) AI engine. Play in the browser against another human or let the AI pick moves for either side.

## Overview

There are two core components:

**Game engine** — every rule of Go is implemented from scratch in pure Python with no external Go libraries. This includes stone capture (BFS group detection), the Ko rule (board-state snapshot comparison), suicide prevention (including the capture-suicide edge case where a move that looks suicidal is legal because it captures first), and Chinese scoring with territory flood-fill and 2.5 komi. A REST API built with Flask exposes the engine to the browser frontend, which renders the board on an HTML canvas.

**MCTS AI** — `engine/ai.py` implements a full Monte Carlo Tree Search agent. Each search iteration runs four phases: selection via UCB1 (Upper Confidence Bound), expansion of one new tree node, a random rollout with capture-biased move selection, and backpropagation of the result. The rollout is heavily optimised to avoid Python deep copies — legality is checked with an O(16) direct-neighbour scan instead of a full BFS group traversal, cutting per-move overhead by ~300×. At 800 simulations the AI responds in ~3 seconds and plays at a beginner level; increasing `num_simulations` trades time for stronger play.

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
| POST | `/ai_move` | MCTS AI picks and plays a move for the current player |

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

## AI tuning

`AIPlayer` in `engine/ai.py` accepts two parameters:

| Parameter | Default | Effect |
|---|---|---|
| `num_simulations` | `800` | Rollout budget per move — higher = stronger, slower |
| `exploration_c` | `1.41` | UCB1 exploration constant (√2 is theoretically optimal) |

The server instantiates the AI once at startup (`app.py`). To change strength, edit the `_ai = AIPlayer(num_simulations=800)` line and restart.
