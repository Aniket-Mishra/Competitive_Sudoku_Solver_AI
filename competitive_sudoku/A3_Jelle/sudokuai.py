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
        ai_player_index = game_state.current_player - 1
        depth = (game_state.board.board_height() * game_state.board.board_width()) - (len(game_state.occupied_squares1) +len(game_state.occupied_squares2))  # Total number of unoccupied squares
        N = game_state.board.N


        if self.load() is None:
            loaded_illegal_moves = get_illegal_moves(game_state, None)
        else:
            loaded_illegal_moves, old_game_state = self.load()
            loaded_illegal_moves = get_illegal_moves(game_state, old_game_state) | loaded_illegal_moves

        valid_moves = []
        for square in game_state.player_squares():
            for value in range(1, N+1):
                if (square, value) not in loaded_illegal_moves and game_state.board.get(square) == SudokuBoard.empty:
                    if Move(square, value) not in game_state.taboo_moves:
                        valid_moves.append(Move(square, value))

        #print([(move.square, move.value) for move in valid_moves])
        value_groups = {}
        for move in valid_moves:
            square, value = move.square, move.value
            if value not in value_groups:
                value_groups[value] = []
            value_groups[value].append(move)

        # Interleave the groups
        sorted_moves = []
        for value in sorted(value_groups.keys()):  # Sort by value
            sorted_moves.extend(value_groups[value])

        self.propose_move(random.choice(sorted_moves))
        for depth in range(1, depth + 1):
            best_move = None
            best_score = float("-inf")
            depth_move_scores = []
            print(f'A3 {depth}')
            for move in sorted_moves:
                next_state = simulate_move(game_state, move)
                new_illegal_moves = get_illegal_moves(next_state, game_state) | loaded_illegal_moves

                score = minimax(next_state, depth, float("-inf"), float("inf"), maximizing=False,
                                ai_player_index=ai_player_index, illegal_moves=new_illegal_moves)
                depth_move_scores.append((move, score))

                if score >= best_score:
                    best_score = score
                    best_move = move
                if best_move:
                    self.propose_move(best_move)
                    #new_state = simulate_move(game_state, best_move)
                    #new_illegal_moves = get_illegal_moves(new_state, game_state) | loaded_illegal_moves
                    self.save((loaded_illegal_moves, game_state))
                
            sorted_moves = [i[0]for i in sorted(depth_move_scores, key=lambda x: x[1], reverse=True)]


# python .\simulate_game.py --first=A3_Jelle --second=greedy_player --board=boards/empty-3x3.txt
