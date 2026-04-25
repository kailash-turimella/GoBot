from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.board import Board


class AIPlayer:
    """
    Stub AI player. select_move raises NotImplementedError.
    Replace with a Monte Carlo Tree Search implementation.
    """

    def __init__(self, num_simulations: int = 800, exploration_c: float = 1.41) -> None:
        self.num_simulations = num_simulations
        self.exploration_c   = exploration_c

    def select_move(self, board: "Board", color: int) -> Optional[tuple[int, int]]:
        raise NotImplementedError("AI not yet implemented. Use MCTS or similar in future.")
