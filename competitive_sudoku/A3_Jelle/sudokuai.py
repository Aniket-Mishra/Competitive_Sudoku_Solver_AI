import random
import copy
import time

from competitive_sudoku.sudoku import (
    GameState,
    Move,
    TabooMove,
    SudokuBoard,
)
import competitive_sudoku.sudokuai
from typing import Tuple
from A3_Jelle.helper_functions import simulate_move, get_valid_moves, filter_moves
from A3_Jelle.minimax import minimax
from A3_Jelle.taboo_helpers import naked_singles, hidden_singles

from A3_Jelle.sudoku_solver import SudokuSolver
from A3_Jelle.tt import solve_sudoku

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
        import time

        start_time = time.perf_counter()

        _, moves, tim = solve_sudoku(game_state)
        valid_moves = filter_moves(game_state, moves)

  


        ai_player_index = game_state.current_player - 1
        # depth = (game_state.board.board_height() * game_state.board.board_width()) - (
        #     len(game_state.occupied_squares1) +
        #     len(game_state.occupied_squares2)
        # )  
        # # Total number of unoccupied squares

        # valid_moves = get_valid_moves(game_state)

        # # Call Taboo Heuristics
        # valid_moves = naked_singles(game_state, valid_moves)

        # valid_moves = [
        #     Move((row, col), value)
        #     for (row, col), values in valid_moves.items()
        #     for value in values
        # ]
        l_time = time.perf_counter()

        print(tim)
        print(f'last time {l_time - start_time}')
        self.propose_move(random.choice(valid_moves))

        for depth in range(1, 10):
            best_move = None
            best_score = float("-inf")
            depth_move_scores = []
            for move in valid_moves:

                next_state = simulate_move(game_state, move)
                score = minimax(next_state, depth, float(
                    "-inf"), float("inf"), maximizing=False, ai_player_index=ai_player_index, )
                
                print(f"move {move} gives score {score} for depth {depth}")
                

                depth_move_scores.append((move, score))

                if score >= best_score:
                    best_score = score
                    best_move = move
                if best_move:
                    self.propose_move(best_move)
                
            valid_moves = [i[0]for i in sorted(depth_move_scores, key=lambda x: x[1], reverse=True)]


# python .\simulate_game.py --first=A3_Jelle --second=greedy_player --board=boards/empty-3x3.txt
