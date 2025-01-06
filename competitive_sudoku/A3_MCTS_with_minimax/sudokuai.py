import random
import copy
from competitive_sudoku.sudoku import (
    GameState,
    Move,
    TabooMove,
    SudokuBoard,
)
import competitive_sudoku.sudokuai
from A3_MCTS.helper_functions import get_valid_moves
from A3_MCTS.taboo_helpers import naked_singles, hidden_singles

import time
from A3_MCTS.MCTS import MonteCarloTree


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration using Minimax.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Computes the best move for the AI using Monte Carlo Tree Search.
        """
        ai_player_index = game_state.current_player - 1

        mcts = MonteCarloTree(game_state, ai_player_index)

        if not mcts.root.children:
            mcts.expand(mcts.root)

        initial_moves = get_valid_moves(game_state)
        initial_moves = naked_singles(game_state, initial_moves)
        initial_moves = [
            Move((row, col), value)
            for (row, col), values in initial_moves.items()
            for value in values
        ]
        if not initial_moves:
            return  # No valid moves, do nothing
        self.propose_move(random.choice(initial_moves))

        iteration = 0
        while True:
            best_leaf_node = mcts.visit()

            if best_leaf_node.visit_count == 0:
                selected_node = best_leaf_node
            else:
                selected_node = mcts.expand(best_leaf_node)
            result = mcts.simulate_playout(selected_node)

            mcts.backpropagate(selected_node, result)

            if iteration % 5 == 0:
                best_move = self.get_best_move_from_tree(mcts, initial_moves)
                self.propose_move(best_move)

            iteration += 1

    def get_best_move_from_tree(
        self, tree: MonteCarloTree, fallback_moves: list[Move]
    ) -> Move:
        """
        Returns the best move from the MCTS tree based on visit count.
        """
        best_child = None
        max_visits = -1

        for child in tree.root.children:
            if child.visit_count > max_visits:
                max_visits = child.visit_count
                best_child = child

        return best_child.move if best_child else random.choice(fallback_moves)


# python .\simulate_game.py --first=A3_MCTS --second=greedy_player --board=boards/empty-3x3.txt
