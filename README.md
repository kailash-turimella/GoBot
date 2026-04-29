# 9×9 Go — Python/Flask + AlphaGo Zero-Style Neural Network AI

A fully playable 9×9 Go game with all rules implemented from scratch and a neural network AI trained via supervised learning on real games followed by AlphaGo Zero-style self-play. Play in the browser against another human or let the AI pick moves.

## Overview

There are two core components:

**Game engine** — every rule of Go is implemented from scratch in pure Python with no external Go libraries. This includes stone capture (BFS group detection), the Ko rule (board-state snapshot comparison), suicide prevention (including the capture-suicide edge case where a move that looks suicidal is legal because it captures first), and Chinese scoring with territory flood-fill and 2.5 komi. A REST API built with Flask exposes the engine to the browser frontend, which renders the board on an HTML canvas.

**Neural network AI** — `engine/ai.py` loads a trained ResNet policy+value network (`models/model`) and picks moves in a single forward pass: the board is encoded as feature planes, the policy head outputs probabilities over all 81 intersections, illegal moves are masked out, and the highest-probability legal move is played. The network is trained in two stages — supervised pre-training on human/computer games to learn basic shape and joseki, followed by AlphaGo Zero-style self-play where the network improves by playing against itself using PUCT Monte Carlo Tree Search. Each self-play iteration the network generates training data, trains on it, and is evaluated against the previous best checkpoint before being promoted.

## Project structure

```
GoBot/
├── app.py                  Flask server + REST API
├── engine/
│   ├── board.py            Board state, stone placement, capture logic
│   ├── rules.py            Ko, suicide, legal-move validation
│   ├── scoring.py          Chinese scoring, territory flood-fill
│   ├── network.py          ResNet architecture (policy + value heads)
│   └── ai.py               Loads trained model, selects moves
├── training/
│   ├── supervised.py       Pre-train on SGF game dataset
│   ├── self_play.py        Generate self-play games → replay buffer
│   ├── train.py            Fine-tune on replay buffer
│   └── eval.py             Pit new checkpoint vs. current best
├── models/            Saved model weights (.pth files)
├── data/
│   ├── Games/              SGF game archives for supervised training
│   └── replay_buffer.npz   Self-play training data
├── static/
│   ├── js/game.js          Canvas rendering + click handling
│   └── css/style.css       Board and stone styling
├── templates/
│   └── index.html          Single-page shell
└── tests/                  pytest test suite
```

## Setup

### 1 — Create and activate the virtual environment

```bash
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

The AI move button requires a trained model at `models/model`. Without it the `/ai_move` route returns a `503` with a clear error message.

## Training the AI

### Step 1 — Supervised pre-training

Get 9×9 SGF game files (CGOS archives work well) and place them under `data/Games/`. Then run:

```bash
python -m training.supervised --max-games 50000 --epochs 20 --device mps --promote
```

`--promote` copies the result to `models/best_model.pth` automatically.
Use `--device mps` on Apple Silicon, `--device cuda` on NVIDIA GPU, `--device cpu` otherwise.

### Step 2 — Self-play loop

```bash
# Generate self-play games using the current best model
python -m training.self_play --games 200 --sims 400 --device mps

# Fine-tune on the new data
python -m training.train --epochs 10 --model models/best_model.pth --device mps

# Promote if the new checkpoint wins >55% of evaluation games
python -m training.eval --candidate models/model_v10.pth
```

Repeat from `self_play` with the promoted `best_model.pth`. Each iteration the model improves.

### Run overnight (macOS)

```bash
nohup python -m training.supervised --max-games 50000 --epochs 20 --device mps --promote \
    > training.log 2>&1 &

tail -f training.log   # check progress
```

## Network architecture

`engine/network.py` defines `GoNetwork` — a dual-head ResNet:

| Component | Detail |
|---|---|
| Input | 17 × 9 × 9 feature planes |
| Trunk | 5 residual blocks, 64 filters, 3×3 convolutions |
| Policy head | Conv → flatten → linear → 81 logits (one per intersection) |
| Value head | Conv → flatten → linear → tanh → scalar ∈ [−1, 1] |

Input planes: current player's stones, opponent's stones, color-to-move, plus 14 history planes (AlphaGo Zero style).

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
| POST | `/ai_move` | Neural network picks and plays a move (503 if no model) |

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
