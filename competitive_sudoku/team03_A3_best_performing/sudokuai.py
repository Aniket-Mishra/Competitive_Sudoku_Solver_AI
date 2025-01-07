import random
from competitive_sudoku.sudoku import (
    GameState,
    Move,
)
import competitive_sudoku.sudokuai
from typing import Tuple
from team03_A3_best_performing.helper_functions import simulate_move, get_valid_moves
from team03_A3_best_performing.minimax import minimax
from team03_A3_best_performing.taboo_helpers import naked_singles


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI Class that computes a move for a given sudoku configuration using Minimax.

    Args:
        SudokuAI (sudoku ai object): parent class - inherited here
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Computes the best move for the AI using Minimax with iterative deepening.

        The function evaluates the game state to determine the optimal move for the AI player. 
        It applies iterative deepening to explore moves to increasing depths, sorting moves by their 
        associated values for more efficient evaluation. The best move is periodically updated based on scores.

        Parameters:
        - game_state (GameState): The current state of the Sudoku game.

        Returns:
        - None: Proposes the best move for the AI to make.
        """

        ai_player_index = game_state.current_player - 1
        depth = (game_state.board.board_height() * game_state.board.board_width()) - (
            len(game_state.occupied_squares1) +
            len(game_state.occupied_squares2)
        )

        valid_moves = get_valid_moves(game_state)
        valid_moves = naked_singles(game_state, valid_moves)
        valid_moves = [
            Move((row, col), value)
            for (row, col), values in valid_moves.items()
            for value in values
        ]

        self.propose_move(random.choice(valid_moves))

        value_groups = {}
        for move in valid_moves:
            _, value = move.square, move.value
            if value not in value_groups:
                value_groups[value] = []
            value_groups[value].append(move)

        sorted_moves = []
        for value in sorted(value_groups.keys()):
            sorted_moves.extend(value_groups[value])

        for depth in range(1, depth + 1):
            best_move = None
            best_score = float("-inf")
            depth_move_scores = []
            for move in sorted_moves:
                next_state = simulate_move(game_state, move)
                score = minimax(next_state, depth, float(
                    "-inf"), float("inf"), maximizing=False, ai_player_index=ai_player_index,)
                depth_move_scores.append((move, score))

                if score >= best_score:
                    best_score = score
                    best_move = move
                if best_move:
                    self.propose_move(best_move)

            sorted_moves = [i[0]for i in sorted(
                depth_move_scores, key=lambda x: x[1], reverse=True)]
