# 9×9 Go — Python/Flask + Neural Network AI

A fully playable 9×9 Go game with all rules implemented from scratch and a neural network AI trained via supervised learning on real games followed by self-play, using a similar strategy to AlphaGo. Play in the browser against another human or let the AI pick moves.

---

## Overview

There are two core components:

**Game engine** — every rule of Go is implemented from scratch in pure Python with no external Go libraries. This includes stone capture (BFS group detection), the Ko rule (board-state snapshot comparison), suicide prevention (including the capture-suicide edge case where a move that looks suicidal is legal because it captures first), and Chinese scoring with territory flood-fill and 2.5 komi.

**Neural network AI** — a ResNet policy+value network trained in two stages: supervised pre-training on 500,000 real computer-vs-computer games from CGOS, followed by self-play where the network improves by playing against itself. At game time the AI encodes the board into feature planes, runs a forward pass through the policy head, masks illegal moves, and plays the highest-probability legal move.

---

## Project structure

```
GoBot/
├── app.py                      Flask server + REST API
├── requirements.txt
├── frontend/                   Browser client
│   ├── static/js/game.js       Canvas rendering, click handling, API calls
│   ├── static/css/style.css
│   └── templates/index.html
├── go_engine/                  Pure Go rules, no AI
│   ├── board.py                Board state, stone placement, capture (BFS)
│   ├── rules.py                Ko, suicide, legal-move validation
│   ├── scoring.py              Chinese scoring, territory flood-fill
│   └── network.py              ResNet architecture (policy + value heads)
├── AI/
│   ├── ai.py                   Loads trained model, selects moves
│   ├── models/                 Saved checkpoints (.pth files)
│   │   ├── v1.pth              Supervised baseline (48k games)
│   │   ├── v2.pth              After self-play fine-tuning
│   │   ├── v3.pth              After 500k supervised training (current)
│   │   └── trials/             All intermediate epoch snapshots
│   └── training/
│       ├── supervised.py       Pre-train on SGF game dataset
│       ├── self_play.py        Generate self-play games → replay buffer
│       ├── train.py            Fine-tune on replay buffer
│       ├── eval.py             Round-robin tournament between checkpoints
│       └── data/
│           ├── README.md       Dataset source and setup instructions
│           ├── Games/          Downloaded CGOS archives (not tracked)
│           └── *.sgf           10 example SGF files
└── tests/                      pytest suite (52 tests)
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

The AI move button requires a trained model at `AI/models/v3.pth` (included). Without it the `/ai_move` route returns a `503`.

---

## AI strategy

### Network architecture

`go_engine/network.py` defines `GoNetwork` — a dual-head ResNet:

| Component | Detail |
|---|---|
| Input | 17 × 9 × 9 feature planes |
| Trunk | 5 residual blocks, 64 filters, 3×3 convolutions with batch norm |
| Policy head | Conv → flatten → linear → 81 logits (one per intersection) |
| Value head | Conv → flatten → linear → tanh → scalar ∈ [−1, 1] |

Input planes: current player's stones (plane 0), opponent's stones (plane 1), colour-to-move (plane 2), plus 14 history planes — same layout as AlphaGo Zero.

Both heads share the ResNet trunk so the network simultaneously learns to evaluate positions (value) and predict moves (policy).

### Training pipeline

**Stage 1 — Supervised pre-training (`training/supervised.py`)**

The network is trained on real games from CGOS (see `AI/training/data/README.md`). For each position in each game it learns:
- **Policy**: predict which move was actually played (cross-entropy loss)
- **Value**: predict who won the game (MSE loss)

500,000 games were used, processed in chunks of 40,000 to fit in memory. Each chunk starts from the previous checkpoint so knowledge accumulates across chunks. The `train_500k.sh` script automates the full sequence.

**Stage 2 — Self-play (`training/self_play.py`)**

The network plays games against itself, sampling moves from the policy head with temperature (exploratory for the first 30 moves, near-greedy after). Each position is labelled with the actual game outcome. These games are saved to `AI/training/data/replay_buffer.npz`.

**Stage 3 — Fine-tuning (`training/train.py`)**

The network trains on the self-play buffer with soft policy targets (the full temperature-scaled distribution, not just the move played) and MSE value loss.

**Stage 4 — Evaluation (`training/eval.py`)**

All checkpoints play a round-robin tournament, 20 games per match, alternating colours. The winning checkpoint is promoted.

### Move selection at game time

`AI/ai.py` loads the trained model and at each turn:
1. Encodes the board as a `(17, 9, 9)` float32 tensor
2. Runs a forward pass → 81 policy logits
3. Sets logits for illegal moves to `−∞`
4. Returns `argmax` of the remaining logits as `(row, col)`

No search — single forward pass, ~5ms per move.

---

## Training the AI yourself

### Get the data

Download 9×9 game archives from **http://www.yss-aya.com/cgos/**, place them in `AI/training/data/Games/`, and extract:

```bash
cd AI/training/data/Games
for f in *.tar.bz2; do tar -xjf "$f"; done
```

### Run supervised pre-training

```bash
# Quick start (50k games, ~30 min on M1)
python -m AI.training.supervised --games AI/training/data/Games/ --max-games 50000 --epochs 20 --device mps --promote

# Full 500k game run in chunks (automated)
bash train_500k.sh
```

### Self-play loop 

```bash
# Generate self-play games
python -m AI.training.self_play --games 500 --sims 0 --model AI/models/v3.pth --device mps

# Fine-tune on self-play data
python -m AI.training.train --buffer AI/training/data/replay_buffer.npz --model AI/models/v3.pth --epochs 10 --device mps

# Run tournament to find the best checkpoint
python -m AI.training.eval --folder AI/models/ --games 20 --device cpu --promote v4
```

Use `--device mps` on Apple Silicon, `--device cuda` on NVIDIA GPU, `--device cpu` otherwise.

---

## Running tests

```bash
pytest tests/ -v
```

52 tests covering board logic, rules (Ko, suicide, capture-suicide), scoring, and the REST API.

---

## REST API

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Serve the browser client |
| POST | `/new_game` | Reset board, return initial state |
| POST | `/move` | `{"row": int, "col": int}` — place a stone |
| GET | `/state` | Current board state as JSON |
| POST | `/pass` | Current player passes |
| GET | `/legal_moves` | All legal moves for the current player |
| POST | `/ai_move` | Neural network picks and plays a move |

### State response

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

---

## Rules implemented

- **Capture**: a group with no liberties after a placement is removed
- **Suicide**: illegal unless the move simultaneously captures an opponent group
- **Ko**: you may not recreate the board state from your opponent's last move
- **Scoring**: Chinese rules — stones + territory, 2.5 komi for white
- **Game end**: two consecutive passes, same player passes 3 times, or 150 move limit
