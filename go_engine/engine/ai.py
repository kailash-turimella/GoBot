"""
ai.py — AI player stub for a 9x9 Go game.

This file defines the public AIPlayer interface.  select_move currently raises
NotImplementedError.  The extensive comments below explain exactly how a Monte
Carlo Tree Search (MCTS) agent would be implemented here when the time comes.

=============================================================================
INTENDED FUTURE ARCHITECTURE: Monte Carlo Tree Search (MCTS)
=============================================================================

MCTS builds a partial game tree iteratively.  Each node stores a board state,
the move that produced it, visit count N, and cumulative score Q.  The
algorithm runs four phases per iteration:

  1. SELECTION
     From the root (current board state), descend the tree by always choosing
     the child with the highest Upper Confidence Bound (UCB1):

         UCB1(v) = Q(v)/N(v)  +  C * sqrt( ln(N(parent)) / N(v) )

     where C ≈ 1.41 balances exploration vs. exploitation.  Stop at a node
     that still has unexpanded children (i.e., legal moves not yet in tree).

  2. EXPANSION
     Pick one unexpanded legal move from the stopped node (use
     rules.get_legal_moves), create a new child node, place the stone on a
     deep-copied Board.

  3. SIMULATION  (rollout)
     From the new node, play random legal moves for alternating colors until
     both players pass consecutively.  Score the terminal position with
     scoring.calculate_score().  Random rollouts can be replaced with a
     lightweight neural-network policy for stronger play (AlphaGo-style).

  4. BACKPROPAGATION
     Walk from the new node back to the root, incrementing N by 1 and adding
     the rollout result to Q for every ancestor.  The result is from the
     perspective of the node's color (1.0 = win, 0.0 = loss, 0.5 = draw).

After the iteration budget (num_simulations), return the move of the root
child with the highest visit count N (most-visited = most-confident).

Plugging the AI in:
  - Replace `raise NotImplementedError` with the MCTS loop below.
  - The only engine APIs needed are:
      rules.get_legal_moves(board, color)   → list of candidate moves
      board.place_stone(row, col)           → apply a move on the copy
      scoring.calculate_score(board)        → evaluate terminal states
  - Pass (return None) must be included as a legal option in the tree.
  - Always work on copy.deepcopy(board) so the live game state is untouched.
  - For 9×9, pure Python MCTS at 1 000 simulations produces a weak but
    legal-playing opponent in ~1 second per move.

Performance path:
  - 1 000 rollouts   → beginner strength, ~1 s on a modern laptop
  - 10 000 rollouts  → intermediate, ~10 s
  - Neural policy    → strong amateur with modest rollout budget
  - GTP bridge       → delegate to Leela Zero / Katago for expert play
=============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from engine.board import Board


class AIPlayer:
    """
    Stub AI player.  Every call to select_move raises NotImplementedError.

    When MCTS is implemented, __init__ accepts tuning parameters:
      num_simulations  — rollout budget per move (higher = stronger / slower)
      exploration_c    — C constant in UCB1 (default ≈ 1.41)
    """

    def __init__(self, num_simulations: int = 1000, exploration_c: float = 1.41) -> None:
        self.num_simulations = num_simulations
        self.exploration_c = exploration_c

    def select_move(self, board: "Board", color: int) -> Optional[tuple[int, int]]:
        """
        Select the best move for color on the given board.

        Args:
            board: Current Board instance.  Will NOT be mutated.
            color: 1 for black, 2 for white.

        Returns:
            (row, col) of the chosen move, or None to pass.

        Raises:
            NotImplementedError: Always — until MCTS is implemented.

        Skeleton of the future implementation:
            import copy
            from engine.rules import get_legal_moves
            from engine.scoring import calculate_score

            root = _MCTSNode(copy.deepcopy(board), color)
            for _ in range(self.num_simulations):
                node = self._select(root)
                child = self._expand(node)
                result = self._simulate(child)
                self._backpropagate(child, result)
            best = max(root.children, key=lambda n: n.visits)
            return best.move
        """
        raise NotImplementedError(
            "AI not yet implemented. Use MCTS or similar in future."
        )
