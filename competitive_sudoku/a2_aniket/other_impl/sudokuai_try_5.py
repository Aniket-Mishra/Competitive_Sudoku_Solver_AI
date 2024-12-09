import numpy as np
import copy
from competitive_sudoku.sudoku import (
    GameState,
    Move,
    TabooMove,
    SudokuBoard,
)
from competitive_sudoku.sudokuai import SudokuAI


class SudokuAI(SudokuAI):
    def __init__(self):
        pass

    def compute_best_move(self, game_state: GameState) -> None:
        """Compute and propose the best move for the given game state."""
        valid_moves = self.calculate_possible_moves(game_state)
        if not valid_moves:
            return  # No valid moves available

        # Fallback to a random valid move if needed
        fallback_move = list(valid_moves.keys())[0] if valid_moves else None

        # Prioritize moves based on heuristic scores
        prioritized_moves = self.prioritize_moves(valid_moves, game_state)

        # Propose the best move (highest priority)
        if prioritized_moves:
            best_move = prioritized_moves[0]
            self.propose_move(Move(best_move[0], best_move[1]))
        elif fallback_move:
            self.propose_move(Move(fallback_move[0], fallback_move[1]))

    def is_valid_move(self, square, value, game_state):
        """Check if placing a value in the square is a valid move."""
        row, col = square
        N = game_state.board.N

        # Taboo and basic constraints
        return (
            game_state.board.get(square) == SudokuBoard.empty
            and not TabooMove(square, value) in game_state.taboo_moves
            and value not in [game_state.board.get((row, c)) for c in range(N)]  # Row
            and value
            not in [game_state.board.get((r, col)) for r in range(N)]  # Column
            and value not in self.get_region_values(game_state.board, square)  # Subgrid
        )

    def get_region_values(self, board: SudokuBoard, square):
        """Retrieve all values in the subgrid containing the given square."""
        region_width = board.region_width()
        region_height = board.region_height()
        start_row = (square[0] // region_height) * region_height
        start_col = (square[1] // region_width) * region_width
        return [
            board.get((r, c))
            for r in range(start_row, start_row + region_height)
            for c in range(start_col, start_col + region_width)
        ]

    def calculate_possible_moves(self, game_state: GameState):
        """Get all possible moves with candidates for each square."""
        N = game_state.board.N
        possible_moves = {}

        for row in range(N):
            for col in range(N):
                square = (row, col)
                if game_state.board.get(square) == SudokuBoard.empty:
                    candidates = self.get_candidates(square, game_state)
                    for value in candidates:
                        if self.is_valid_move(square, value, game_state):
                            possible_moves[(square, value)] = self.evaluate_move(
                                square, value, game_state
                            )
        return possible_moves

    def get_candidates(self, square, game_state: GameState):
        """Get possible candidates for a square."""
        row, col = square
        N = game_state.board.N
        if game_state.board.get(square) != SudokuBoard.empty:
            return []
        candidates = set(range(1, N + 1))
        # Remove numbers already in the same row
        for c in range(N):
            candidates.discard(game_state.board.get((row, c)))
        # Remove numbers already in the same column
        for r in range(N):
            candidates.discard(game_state.board.get((r, col)))
        # Remove numbers already in the same region
        region_values = self.get_region_values(game_state.board, square)
        candidates.difference_update(region_values)
        return list(candidates)

    def prioritize_moves(self, valid_moves, game_state: GameState):
        """Sort moves by their heuristic value in descending order."""
        return sorted(valid_moves.items(), key=lambda x: x[1], reverse=True)

    def evaluate_move(self, square, value, game_state: GameState):
        """Evaluate the heuristic score of a move."""
        row, col = square
        # Score based on region completions and blocking opponent
        completed_regions = self.count_completed_regions(game_state, row, col, value)
        # Simple weight for now, extend for blocking logic
        return completed_regions

    def count_completed_regions(self, game_state: GameState, row, col, value):
        """Count regions (row, column, subgrid) completed by the move."""
        N = game_state.board.N
        score = 0

        # Simulate move
        game_state.board.put((row, col), value)

        # Row completion
        if all(game_state.board.get((row, c)) != SudokuBoard.empty for c in range(N)):
            score += 1

        # Column completion
        if all(game_state.board.get((r, col)) != SudokuBoard.empty for r in range(N)):
            score += 1

        # Subgrid completion
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

        # Undo simulation
        game_state.board.put((row, col), SudokuBoard.empty)
        return score

    def propose_move(self, move: Move) -> None:
        """Propose the chosen move."""
        print(f"Proposing move: {move.square}, Value: {move.value}")
