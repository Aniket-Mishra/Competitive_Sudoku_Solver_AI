import numpy as np
import copy
from competitive_sudoku.sudoku import GameState, Move, TabooMove, SudokuBoard
from competitive_sudoku.sudokuai import SudokuAI


class SudokuAI(SudokuAI):
    def __init__(self):
        pass

    def compute_best_move(self, game_state: GameState) -> None:
        """Compute and propose the best move for the given game state."""
        valid_moves = self.get_valid_moves(game_state)
        if not valid_moves:
            return  # No valid moves available

        # Prioritize moves based on heuristic scores
        prioritized_moves = self.prioritize_moves(valid_moves, game_state)

        # Propose the best move (highest priority)
        if prioritized_moves:
            self.propose_move(prioritized_moves[0])

    def get_valid_moves(self, game_state: GameState):
        """Generate all valid moves for the current state, avoiding taboo moves."""
        N = game_state.board.N
        valid_moves = []

        for row in range(N):
            for col in range(N):
                for value in range(1, N + 1):
                    square = (row, col)
                    if self.is_valid_move(square, value, game_state):
                        valid_moves.append(Move(square, value))
        return valid_moves

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

    def prioritize_moves(self, valid_moves, game_state: GameState):
        """Rank moves based on heuristic evaluation."""
        move_scores = []
        for move in valid_moves:
            score = self.evaluate_move(move, game_state)
            move_scores.append((move, score))

        # Sort moves by descending score
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_scores]

    def evaluate_move(self, move: Move, game_state: GameState):
        """Evaluate the heuristic score of a move."""
        simulated_state = self.simulate_move(game_state, move)
        return self.calculate_heuristic(simulated_state, move)

    def simulate_move(self, game_state: GameState, move: Move):
        """Simulate applying a move to the game state."""
        new_state = copy.deepcopy(game_state)
        new_state.board.put(move.square, move.value)
        new_state.moves.append(move)
        return new_state

    def calculate_heuristic(self, game_state: GameState, move: Move):
        """Heuristic: Points for region completions and blocking opponent."""
        completed_regions = self.count_completed_regions(game_state, move)
        block_opponent = self.check_opponent_disruption(game_state, move)
        return completed_regions + 0.5 * block_opponent  # Weighting

    def count_completed_regions(self, game_state: GameState, move: Move):
        """Count the number of regions (rows, columns, subgrids) completed by the move."""
        row, col = move.square
        N = game_state.board.N
        score = 0

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

        return score

    def check_opponent_disruption(self, game_state: GameState, move: Move):
        """Evaluate if the move disrupts the opponent's potential completions."""
        # Placeholder logic: Could involve analyzing opponent's allowed squares
        return 0  # Extend with opponent analysis

    def propose_move(self, move: Move) -> None:
        """Propose the chosen move."""
        print(f"Proposing move: {move.square}, Value: {move.value}")
