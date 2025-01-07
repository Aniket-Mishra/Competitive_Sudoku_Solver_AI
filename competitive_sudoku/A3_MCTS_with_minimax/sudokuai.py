import random
from competitive_sudoku.sudoku import (
    GameState,
    Move,
)
import competitive_sudoku.sudokuai
from A3_MCTS_with_minimax.helper_functions import get_valid_moves
from A3_MCTS_with_minimax.taboo_helpers import naked_singles
from A3_MCTS_with_minimax.MCTS import MonteCarloTree


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
            return

        self.propose_move(random.choice(initial_moves))

        while True:
            best_leaf_node = mcts.visit()

            if best_leaf_node.visit_count == 0:
                selected_node = best_leaf_node
            else:
                selected_node = mcts.expand(best_leaf_node)
            result = mcts.simulate_playout(selected_node)

            mcts.backpropagate(selected_node, result)

            best_move = self.get_best_move_from_tree(mcts, initial_moves)
            self.propose_move(best_move)

    def get_best_move_from_tree(self, tree: MonteCarloTree, fallback_moves: list[Move]) -> Move:
        """
        Identifies the best move based on visit counts of child nodes.

        This function determines the move leading to the most explored node, indicating 
        the most promising move.

        Returns:
        - Move | None: The best move or None if no moves are available.
        """

        best_child = None
        max_visits = -1

        for child in tree.root.children:
            if child.visit_count > max_visits:
                max_visits = child.visit_count
                best_child = child

        return best_child.move if best_child else random.choice(fallback_moves)
