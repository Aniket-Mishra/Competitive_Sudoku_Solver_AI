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
        Minimax with iterative deepening depth  # ToDo - Caching if needed
        """
        initial_moves = get_valid_moves(game_state)
        initial_moves = naked_singles(game_state, initial_moves)
        initial_moves = [
            Move((row, col), value)
            for (row, col), values in initial_moves.items()
            for value in values
        ]

        self.propose_move(random.choice(initial_moves))

        our_agent = game_state.current_player - 1

        mcts = MonteCarloTree(game_state)

        result = mcts.simulate_playout(mcts.root)
        mcts.backpropagate(mcts.root, result, our_agent)

        iteration = 0

        while True:
            best_leaf_node = mcts.visit()
            if best_leaf_node.visit_count == 0:
                selected_node = best_leaf_node
            else:
                selected_node = mcts.expand(best_leaf_node)
            
            result = mcts.simulate_playout(selected_node)
            mcts.backpropagate(selected_node, result, our_agent)

            if iteration % 5 == 0:
                best_move = get_best_move(mcts, False)
                self.propose_move(best_move)
            
            iteration += 1
    
def get_best_move(tree: MonteCarloTree, is_robust: bool = False):
    root = tree.root
    bestScore = float('-inf')
    bestChild = None
    
    for child in root.children:
        if child.visit_count == 0:
            continue
        score = child.visit_count if is_robust else (child.player_1_wins / child.visit_count)

        if score > bestScore:
            bestScore = score
            bestChild = child
        
    return bestChild.move


# python .\simulate_game.py --first=A3_MCTS --second=greedy_player --board=boards/empty-3x3.txt
