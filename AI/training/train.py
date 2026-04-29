"""
train.py — Fine-tune GoNetwork on self-play data from the replay buffer.

Combined loss:
    L = cross_entropy(π, softmax(logits)) + MSE(v, z)

  π   = MCTS visit-count distribution (soft target from self_play.py)
  z   = actual game outcome ∈ {+1, -1}
  v   = network value head output

Saves models/model_v{N}.pth after each epoch.
Run eval.py afterward to promote the best one to best_model.pth.

Usage:
    python -m training.train --buffer data/replay_buffer.npz --epochs 10
    python -m training.train --model models/best_model.pth --epochs 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from go_engine.network import GoNetwork, save_model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(buffer_path: str) -> TensorDataset:
    data     = np.load(buffer_path, allow_pickle=True)
    planes   = torch.from_numpy(data["planes"].astype(np.float32))
    probs    = torch.from_numpy(data["probs"].astype(np.float32))
    outcomes = torch.from_numpy(data["outcomes"].astype(np.float32))
    print(f"Loaded {len(outcomes)} examples from {buffer_path}")
    return TensorDataset(planes, probs, outcomes)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model:         GoNetwork,
    dataset:       TensorDataset,
    epochs:        int   = 10,
    batch_size:    int   = 256,
    lr:            float = 1e-3,
    weight_decay:  float = 1e-4,
    device:        str   = "cpu",
    checkpoint_dir: str  = "models",
) -> None:
    model.to(device).train()
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ckpt_dir  = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        total = policy_sum = value_sum = 0.0
        n_samples = 0

        for planes, probs, outcomes in loader:
            planes, probs, outcomes = (
                planes.to(device), probs.to(device), outcomes.to(device)
            )
            logits, values = model(planes)
            values = values.squeeze(1)

            # Soft cross-entropy: -sum(π · log p)
            log_p       = F.log_softmax(logits, dim=1)
            policy_loss = -(probs * log_p).sum(dim=1).mean()
            value_loss  = F.mse_loss(values, outcomes)
            loss        = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = len(planes)
            total      += loss.item()        * n
            policy_sum += policy_loss.item() * n
            value_sum  += value_loss.item()  * n
            n_samples  += n

        print(
            f"epoch {epoch}/{epochs}  "
            f"loss={total/n_samples:.4f}  "
            f"policy={policy_sum/n_samples:.4f}  "
            f"value={value_sum/n_samples:.4f}"
        )
        save_model(model, ckpt_dir / f"model_v{epoch}.pth")

    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train GoNetwork on self-play data")
    parser.add_argument("--epochs",  type=int,   default=10)
    parser.add_argument("--batch",   type=int,   default=256)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--buffer",  type=str,   default="data/replay_buffer.npz")
    parser.add_argument("--model",   type=str,   default=None,
                        help="Checkpoint to resume from (omit to start fresh)")
    parser.add_argument("--out",     type=str,   default="models")
    parser.add_argument("--device",  type=str,   default="cpu")
    args = parser.parse_args()

    model = GoNetwork()
    if args.model:
        model.load_state_dict(torch.load(args.model, map_location=args.device))
        print(f"Resumed from {args.model}")

    dataset = load_dataset(args.buffer)
    train(
        model, dataset,
        epochs=args.epochs, batch_size=args.batch,
        lr=args.lr, weight_decay=1e-4,
        device=args.device, checkpoint_dir=args.out,
    )


if __name__ == "__main__":
    main()
