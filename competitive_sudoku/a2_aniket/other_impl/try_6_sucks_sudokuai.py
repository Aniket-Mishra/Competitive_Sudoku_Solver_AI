import random
from competitive_sudoku.sudoku import GameState, Move, TabooMove, SudokuBoard
import competitive_sudoku.sudokuai
from typing import List, Tuple
import copy


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Advanced Sudoku AI for competitive play with heuristic optimizations.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Compute the best move using heuristics and minimax.
        """
        valid_moves = self.get_valid_moves(game_state)
        heuristic_moves = self.heuristic_filter(game_state, valid_moves)

        # Use heuristic moves for focused minimax search
        best_move = None
        best_score = float("-inf")
        for move in heuristic_moves:
            next_state = self.simulate_move(game_state, move)
            score = self.evaluate(next_state, game_state.current_player - 1)
            if score > best_score:
                best_score = score
                best_move = move

        if best_move:
            self.propose_move(best_move)

    def get_valid_moves(self, game_state: GameState) -> List[Move]:
        """
        Generate all valid moves for the current game state.
        """
        N = game_state.board.N

        def is_valid_move(square, value):
            return (
                game_state.board.get(square) == SudokuBoard.empty
                and not TabooMove(square, value) in game_state.taboo_moves
                and value not in self.get_row_values(game_state.board, square[0])
                and value not in self.get_col_values(game_state.board, square[1])
                and value not in self.get_region_values(game_state.board, square)
            )

        return [
            Move((i, j), value)
            for i in range(N)
            for j in range(N)
            for value in range(1, N + 1)
            if is_valid_move((i, j), value)
        ]

    def heuristic_filter(
        self, game_state: GameState, valid_moves: List[Move]
    ) -> List[Move]:
        """
        Filter moves using heuristic strategies to reduce branching factor.
        """
        single_option_moves = [
            move
            for move in valid_moves
            if self.is_single_option(game_state, move.square)
        ]
        if single_option_moves:
            return single_option_moves

        # Strategic focus: prioritize moves that maximize score potential
        return sorted(
            valid_moves,
            key=lambda move: self.amount_of_regions_completed(game_state, move),
            reverse=True,
        )

    def is_single_option(self, game_state: GameState, square: Tuple[int, int]) -> bool:
        """
        Check if a square has only one valid value.
        """
        N = game_state.board.N
        valid_values = [
            value
            for value in range(1, N + 1)
            if value not in self.get_row_values(game_state.board, square[0])
            and value not in self.get_col_values(game_state.board, square[1])
            and value not in self.get_region_values(game_state.board, square)
        ]
        return len(valid_values) == 1

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
        w1 = 0.9
        w2 = 0.1

        score_diff = (
            game_state.scores[ai_player_index] - game_state.scores[1 - ai_player_index]
        )

        if ai_player_index == 0:
            allowed_diff = len(game_state.allowed_squares1) - len(
                game_state.allowed_squares2
            )
        else:
            allowed_diff = len(game_state.allowed_squares2) - len(
                game_state.allowed_squares1
            )

        return w1 * score_diff + w2 * allowed_diff

    def amount_of_regions_completed(self, game_state: GameState, move: Move) -> int:
        """
        Calculate the number of regions completed by a move.
        """
        completed = 0
        N = game_state.board.N
        row, col = move.square

        if all(game_state.board.get((row, c)) != SudokuBoard.empty for c in range(N)):
            completed += 1
        if all(game_state.board.get((r, col)) != SudokuBoard.empty for r in range(N)):
            completed += 1

        region_width, region_height = (
            game_state.board.region_width(),
            game_state.board.region_height(),
        )
        start_row, start_col = (row // region_height) * region_height, (
            col // region_width
        ) * region_width
        if all(
            game_state.board.get((r, c)) != SudokuBoard.empty
            for r in range(start_row, start_row + region_height)
            for c in range(start_col, start_col + region_width)
        ):
            completed += 1

        return completed

    def get_row_values(self, board: SudokuBoard, row: int) -> List[int]:
        """
        Get all values in a row.
        """
        return [
            board.get((row, col))
            for col in range(board.N)
            if board.get((row, col)) != SudokuBoard.empty
        ]

    def get_col_values(self, board: SudokuBoard, col: int) -> List[int]:
        """
        Get all values in a column.
        """
        return [
            board.get((row, col))
            for row in range(board.N)
            if board.get((row, col)) != SudokuBoard.empty
        ]

    def get_region_values(
        self, board: SudokuBoard, square: Tuple[int, int]
    ) -> List[int]:
        """
        Get all values in the region of a square.
        """
        region_width, region_height = board.region_width(), board.region_height()
        start_row, start_col = (square[0] // region_height) * region_height, (
            square[1] // region_width
        ) * region_width
        return [
            board.get((r, c))
            for r in range(start_row, start_row + region_height)
            for c in range(start_col, start_col + region_width)
            if board.get((r, c)) != SudokuBoard.empty
        ]
