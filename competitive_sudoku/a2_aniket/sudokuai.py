import random
import copy
from competitive_sudoku.sudoku import GameState, Move
import competitive_sudoku.sudokuai
from solver.sudokusolver import AdvancedSudokuSolver  # Using AdvancedSudokuSolver
from typing import List


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Optimized Sudoku AI leveraging AdvancedSudokuSolver for competitive play.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Compute the best move using heuristics and solver methods.
        """
        solver = AdvancedSudokuSolver(game_state.board)

        # Find the most constrained empty square (prioritize cells with fewest candidates)
        empty_square = solver.find_empty()
        if empty_square is not None:
            row, col = empty_square
            candidates = solver.get_candidates(row, col)
            moves = [Move((row, col), candidate) for candidate in candidates]

            # Prioritize the best move based on number of candidates
            best_move = None
            best_score = float("-inf")

            for move in moves:
                next_state = self.simulate_move(game_state, move)
                score = self.evaluate(next_state, game_state.current_player - 1)
                if score > best_score:
                    best_score = score
                    best_move = move

            if best_move:
                self.propose_move(best_move)
        else:
            # Fallback: Use a random move if no empty squares remain
            valid_moves = self.get_valid_moves(game_state, solver)
            if valid_moves:
                self.propose_move(random.choice(valid_moves))

    def get_valid_moves(
        self, game_state: GameState, solver: AdvancedSudokuSolver
    ) -> List[Move]:
        """
        Generate valid moves for the current game state using AdvancedSudokuSolver.
        """
        valid_moves = []
        for row in range(solver.N):
            for col in range(solver.N):
                if game_state.board.get((row, col)) == game_state.board.empty:
                    candidates = solver.get_candidates(row, col)
                    valid_moves.extend(
                        [Move((row, col), value) for value in candidates]
                    )
        return valid_moves

    def simulate_move(self, game_state: GameState, move: Move) -> GameState:
        """
        Simulate a move and return the resulting game state.
        """
        new_state = copy.deepcopy(game_state)
        new_state.board.put(move.square, move.value)
        new_state.moves.append(move)
        new_state.current_player = 3 - game_state.current_player
        return new_state

    def evaluate(self, game_state: GameState, ai_player_index: int) -> float:
        """
        Evaluate the game state using a heuristic that combines score and remaining moves.
        """
        w1 = 0.9  # Weight for score difference
        w2 = 0.1  # Weight for remaining moves

        # Calculate score difference
        score_diff = (
            game_state.scores[ai_player_index] - game_state.scores[1 - ai_player_index]
        )

        # Calculate the difference in allowed squares
        if ai_player_index == 0:
            allowed_diff = len(game_state.allowed_squares1) - len(
                game_state.allowed_squares2
            )
        else:
            allowed_diff = len(game_state.allowed_squares2) - len(
                game_state.allowed_squares1
            )

        # Combine heuristic components
        return w1 * score_diff + w2 * allowed_diff
