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
from A3_Jelle.helper_functions import simulate_move, get_valid_moves, get_illegal_moves
from A3_Jelle.minimax import minimax
from A3_Jelle.taboo_helpers import naked_singles, hidden_singles
import time


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
        N = game_state.board.N
        ai_player_index = game_state.current_player - 1
        depth = (game_state.board.board_height() * game_state.board.board_width()) - (
            len(game_state.occupied_squares1) +
            len(game_state.occupied_squares2)
        )  # Total number of unoccupied squares

        if self.load() is None:
            saved_illegal_moves = set()
        else:
            saved_illegal_moves, old_game_state = self.load()
            print(f'outdated saved illegal moves {saved_illegal_moves}')
            print(get_illegal_moves(game_state, old_game_state))
            saved_illegal_moves = saved_illegal_moves | get_illegal_moves(game_state, old_game_state)
            print(f'updated saved illegal moves {saved_illegal_moves}')


        valid_moves = []
        for square in game_state.player_squares():
            for value in range(1, N + 1):
                if (square, value) not in saved_illegal_moves and game_state.board.get(square) == SudokuBoard.empty:
                    if Move(square, value) not in game_state.taboo_moves:
                        valid_moves.append(Move(square, value))


        self.propose_move(random.choice(valid_moves))

        for depth in range(1, depth + 1):
            best_move = None
            best_score = float("-inf")
            depth_move_scores = []
            for move in valid_moves:

                next_state = simulate_move(game_state, move)
                illegal_moves = get_illegal_moves(next_state, game_state) | saved_illegal_moves
                #print(f'move {move} - {illegal_moves}')
                score = minimax(next_state, depth, float("-inf"), float("inf"), maximizing=False, ai_player_index=ai_player_index, illegal_moves=illegal_moves)
                
                #print(f"move {move} gives score {score} for depth {depth}")
                

                depth_move_scores.append((move, score))

                if score >= best_score:
                    best_score = score
                    best_move = move
                if best_move:
                    self.propose_move(best_move)
                    self.save((get_illegal_moves(next_state, game_state) |
                              saved_illegal_moves, simulate_move(game_state, best_move)))
                
            valid_moves = [i[0]for i in sorted(depth_move_scores, key=lambda x: x[1], reverse=True)]


# python .\simulate_game.py --first=A3_Jelle --second=greedy_player --board=boards/empty-3x3.txt
