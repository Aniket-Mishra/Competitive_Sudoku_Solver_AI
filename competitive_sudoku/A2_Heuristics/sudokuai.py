import random
import copy
from competitive_sudoku.sudoku import (
    GameState,
    Move,
    TabooMove,
    SudokuBoard,
    pretty_print_sudoku_board,
    pretty_print_game_state,
)
import competitive_sudoku.sudokuai
from typing import Tuple
from A2_Heuristics.helper_functions import simulate_move, get_valid_moves
from A2_Heuristics.minimax import minimax


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration using Minimax.
    """

    def __init__(self):
        super().__init__()

    def custom_sort_generic(self, data):
        """
        Sorts dictionary values based on the following rules:
        1. Lists with exactly 1 entry are prioritized to the front.
        2. Lists with exactly 2 entries are de-prioritized to the back.
        3. Other lists are sorted by length in ascending order (smallest first).
        
        :param data: Dictionary where keys are arbitrary, and values are lists.
        :return: Flattened list of sorted values.
        """
        sorted_items = sorted(
            data.items(),
            key=lambda item: (
                1 if len(item[1]) == 1 else 3 if len(item[1]) == 2 else 2,  # Primary priority rule
                len(item[1])  # Secondary sorting rule for list length
            )
        )

        # Flatten the sorted dictionary into a single list of values
        sorted_values = [value for _, values in sorted_items for value in values]
        return sorted_values


    def compute_best_move(self, game_state: GameState) -> None:
        """
        Minimax with iterative deepening depth  # ToDo - Caching if needed
        """
        ai_player_index = game_state.current_player - 1
        depth = (game_state.board.board_height() * game_state.board.board_width()) - (
            len(game_state.occupied_squares1) +
            len(game_state.occupied_squares2)
        )  # Total number of unoccupied squares

        valid_moves, prio_dict = get_valid_moves(game_state)   # Use heuristics to so
        priority_list = self.custom_sort_generic(prio_dict)
        best_move = None
        best_score = float("-inf")

        for depth in range(1, depth + 1):
            depth_move_scores = []

            for move in priority_list:
                next_state = simulate_move(game_state, move)
                score = minimax(next_state, depth, float(
                    "-inf"), float("inf"), maximizing=False, ai_player_index=ai_player_index,)
                depth_move_scores.append((move, score))

                if score >= best_score:
                    best_score = score
                    best_move = move
                if best_move:
                    self.propose_move(best_move)

            valid_moves = [i[0]for i in sorted(
                depth_move_scores, key=lambda x: x[1], reverse=True)]
