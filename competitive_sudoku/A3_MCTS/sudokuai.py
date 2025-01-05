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

        # Initialize the Monte Carlo Tree with the current game state
        mcts = MonteCarloTree(game_state, ai_player_index)

        # Ensure the root node has children
        if not mcts.root.children:
            mcts.expand(mcts.root)

        # Propose a random initial move in case computation is interrupted
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

        # Perform MCTS iterations
        iteration = 0
        while True:
            # Selection
            best_leaf_node = mcts.visit()

            # Expansion
            if best_leaf_node.visit_count == 0:
                selected_node = best_leaf_node
            else:
                selected_node = mcts.expand(best_leaf_node)

            # Simulation
            result = mcts.simulate_playout(selected_node)

            # Backpropagation
            mcts.backpropagate(selected_node, result)

            # Propose the best move periodically
            if iteration % 5 == 0:
                best_move = self.get_best_move_from_tree(mcts, initial_moves)
                self.propose_move(best_move)

            iteration += 1


    def get_best_move_from_tree(self, tree: MonteCarloTree, fallback_moves: list[Move]) -> Move:
        """
        Returns the best move from the MCTS tree based on visit count.
        """
        best_child = None
        max_visits = -1

        for child in tree.root.children:
            if child.visit_count > max_visits:
                max_visits = child.visit_count
                best_child = child

        # Fallback to a random valid move if no best child found
        return best_child.move if best_child else random.choice(fallback_moves)



# python .\simulate_game.py --first=A3_MCTS --second=greedy_player --board=boards/empty-3x3.txt
