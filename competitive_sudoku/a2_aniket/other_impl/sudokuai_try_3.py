import numpy as np
import copy
from a2_aniket.sudoku import (
    GameState,
    Move,
    TabooMove,
    SudokuBoard,
)


class HeuristicSudokuAI:
    def __init__(self):
        pass

    def get_valid_moves(self, game_state: GameState):
        """
        Generate all valid moves for the current state, avoiding taboo moves.
        """
        N = game_state.board.N
        valid_moves = []

        def is_valid(square, value):
            row, col = square
            return (
                game_state.board.get(square) == SudokuBoard.empty
                and value
                not in [game_state.board.get((row, c)) for c in range(N)]  # Row check
                and value
                not in [
                    game_state.board.get((r, col)) for r in range(N)
                ]  # Column check
                and value
                not in self.get_region_values(game_state.board, square)  # Subgrid check
            )

        for row in range(N):
            for col in range(N):
                for value in range(1, N + 1):
                    square = (row, col)
                    if is_valid(square, value):
                        valid_moves.append(Move(square, value))
        return valid_moves

    def prioritize_moves(self, valid_moves, game_state: GameState):
        """
        Prioritize moves by calculating potential regions completed.
        """
        move_scores = []
        for move in valid_moves:
            score = self.simulate_and_score(move, game_state)
            move_scores.append((move, score))
        move_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by score
        return [move for move, _ in move_scores]

    def simulate_and_score(self, move: Move, game_state: GameState):
        """
        Simulate a move and score it based on region completion.
        """
        new_state = copy.deepcopy(game_state)
        new_state.board.put(move.square, move.value)
        return self.calculate_region_completion(move, new_state)

    def calculate_region_completion(self, move: Move, game_state: GameState):
        """
        Calculate how many regions (rows, columns, subgrids) are completed by the move.
        """
        N = game_state.board.N
        row, col = move.square
        score = 0
        # Check row
        if all(game_state.board.get((row, c)) != SudokuBoard.empty for c in range(N)):
            score += 1

        # Check column
        if all(game_state.board.get((r, col)) != SudokuBoard.empty for r in range(N)):
            score += 1

        # Check subgrid
        region_width = game_state.board.region_width()
        region_height = game_state.board.region_height()
        start_row = (row // region_height) * region_height
        start_col = (col // region_width) * region_width
        if all(
            game_state.board.get((r, c)) != SudokuBoard.empty
            for r in range(start_row, start_row + region_height)
            for c in range(start_col, start_col + region_width)
        ):
            score += 1

        return score

    def compute_best_move(self, game_state: GameState):
        """
        Compute the best move for the current game state using heuristics.
        """
        valid_moves = self.get_valid_moves(game_state)
        prioritized_moves = self.prioritize_moves(valid_moves, game_state)
        return prioritized_moves[0] if prioritized_moves else None
