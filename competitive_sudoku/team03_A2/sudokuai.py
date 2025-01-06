import random
import copy
from competitive_sudoku.sudoku import (
    GameState,
    Move,
    TabooMove,
    SudokuBoard,
)
import competitive_sudoku.sudokuai
from typing import Tuple
from team03_A2.helper_functions import simulate_move, get_valid_moves
from team03_A2.minimax import minimax
from team03_A2.taboo_helpers import naked_singles, hidden_singles
import time  # Tried to time or agent, will use again later.


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
        Minimax with iterative deepening depth  # ToDo - Caching if needed
        Args:
            game_state (GameState): current game state
        """
        ai_player_index = game_state.current_player - 1
        depth = (game_state.board.board_height() * game_state.board.board_width()) - (
            len(game_state.occupied_squares1) + len(game_state.occupied_squares2)
        )  # Number of unoccupied squares

        valid_moves = get_valid_moves(game_state)

        valid_moves = naked_singles(game_state, valid_moves)  # Taboo stuff
        # valid_moves = hidden_singles(game_state, valid_moves) # we'll get it to work some day

        valid_moves = [
            Move((row, col), value)
            for (row, col), values in valid_moves.items()
            for value in values
        ]
        self.propose_move(random.choice(valid_moves))

        best_move = None
        best_score = float("-inf")
        for depth in range(1, depth + 1):
            depth_move_scores = []
            for move in valid_moves:
                next_state = simulate_move(game_state, move, ai_player_index)
                score = minimax(
                    next_state,
                    depth,
                    float("-inf"),
                    float("inf"),
                    maximizing=False,
                    ai_player_index=ai_player_index,
                )
                # print(f"move {move} gives score {score} for depth {depth}")
                depth_move_scores.append((move, score))

                if score >= best_score:
                    best_score = score
                    best_move = move
                if best_move:
                    self.propose_move(best_move)

            # Sort so that we have the best option at the beginning
            # Makes our boo better
            valid_moves = [
                i[0]
                for i in sorted(depth_move_scores, key=lambda x: x[1], reverse=True)
            ]
