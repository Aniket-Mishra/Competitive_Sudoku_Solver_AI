import random
import copy
from competitive_sudoku.sudoku import (
    GameState,
    Move,
    TabooMove,
    SudokuBoard,
)
import competitive_sudoku.sudokuai
from A3_MCTS.helper_functions import get_valid_moves, naked_singles
import time
from A3_MCTS.MCTS import MonteCarloTree
from A3_MCTS.MCTS2 import MonteCarloTree2
from typing import List, Tuple, Dict, Optional, Any


class SudokuAI:
    """
    Sudoku AI that computes a move for the AI using Monte Carlo Tree Search,
    with Minimax-based rollouts at deeper levels.
    """

    def __init__(self):
        pass

    def compute_best_move(self, game_state: GameState) -> Optional[Move]:
        """
        Main entry point. Builds the MCTS tree, iterates search, and returns
        the best move. In a real environment, you might 'propose_move' instead
        of returning it.
        """
        ai_player_index = game_state.current_player - 1

        # Initialize MCTS
        mcts = MonteCarloTree(game_state, ai_player_index)

        # Expand at least once so we have children
        if not mcts.root.children:
            mcts.expand(mcts.root)

        # Fallback / initial move if something goes wrong
        moves_dict = get_valid_moves(game_state)
        initial_moves = []
        for (r, c), vals in moves_dict.items():
            for v in vals:
                initial_moves.append(Move((r, c), v))
        if not initial_moves:
            return None  # no valid moves

        # Example search loop with a time cutoff
        time_limit = 2.0
        start_time = time.time()
        iteration = 0

        # Periodically track the best move to return
        best_move = random.choice(initial_moves)

        while (time.time() - start_time) < time_limit:
            # Selection
            leaf = mcts.visit()
            # Expansion
            if leaf.visit_count == 0:
                selected_node = leaf
            else:
                selected_node = mcts.expand(leaf)
            # Simulation
            result = mcts.simulate_playout(selected_node)
            # Backpropagation
            mcts.backpropagate(selected_node, result)

            # Every few iterations, update best_move
            if iteration % 5 == 0:
                # or you can directly call mcts.best_move()
                maybe_best = mcts.best_move()
                if maybe_best is not None:
                    best_move = maybe_best
            iteration += 1

        return best_move


# python .\simulate_game.py --first=A3_MCTS --second=greedy_player --board=boards/empty-3x3.txt
