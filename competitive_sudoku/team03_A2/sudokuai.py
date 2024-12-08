import random
import copy
from competitive_sudoku.sudoku import (
    GameState,
    Move,
    TabooMove,
    SudokuBoard,
    pretty_print_sudoku_board,
    pretty_print_game_state,
)
import competitive_sudoku.sudokuai
from typing import Tuple


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI with Minimax and heuristics for competitive Sudoku.
    """

    def __init__(self):
        super().__init__()
        self.round_counter = 0

    def get_region_values(self, board: SudokuBoard, square: Tuple[int, int]):
        """
        Gets the values in the region corresponding to the given square.
        """
        region_width = board.region_width()
        region_height = board.region_height()
        start_row = (square[0] // region_height) * region_height
        start_col = (square[1] // region_width) * region_width

        region_values = [
            board.get((r, c))
            for r in range(start_row, start_row + region_height)
            for c in range(start_col, start_col + region_width)
        ]
        return region_values

    def amount_of_regions_completed(self, game_state: GameState, move: Move):
        """
        Checks how many regions (rows, columns, blocks) are completed by the move.
        """
        completed = 0
        N = game_state.board.N
        row, col = move.square

        if all(game_state.board.get((row, c)) != SudokuBoard.empty for c in range(N)):
            completed += 1

        if all(game_state.board.get((r, col)) != SudokuBoard.empty for r in range(N)):
            completed += 1

        region_width = game_state.board.region_width()
        region_height = game_state.board.region_height()
        start_row = (row // region_height) * region_height
        start_col = (col // region_width) * region_width
        if all(
            game_state.board.get((r, c)) != SudokuBoard.empty
            for r in range(start_row, start_row + region_height)
            for c in range(start_col, start_col + region_width)
        ):
            completed += 1

        return completed

    def is_terminal(self, game_state: GameState):
        """
        Checks if the game state is terminal (no valid moves left).
        """
        return len(self.get_valid_moves(game_state)) == 0

    def is_valid_move(
        self, square: Tuple[int, int], value: int, game_state: GameState
    ) -> bool:
        """
        Checks if placing a value in a square is a valid move, including diagonal moves.
        """
        N = game_state.board.N
        row, col = square

        allowed_squares = game_state.player_squares()
        if allowed_squares is not None and square not in allowed_squares:
            return False

        if game_state.board.get(square) != SudokuBoard.empty:
            return False

        if TabooMove(square, value) in game_state.taboo_moves:
            return False

        row_values = [game_state.board.get((row, c)) for c in range(N)]
        col_values = [game_state.board.get((r, col)) for r in range(N)]
        region_values = self.get_region_values(game_state.board, square)

        return (
            value not in row_values
            and value not in col_values
            and value not in region_values
        )

    def get_valid_moves(self, game_state: GameState):
        """
        Gets the valid moves for the current Game State, including diagonal moves.
        """
        N = game_state.board.N

        valid_moves = [
            Move((i, j), value)
            for i in range(N)
            for j in range(N)
            for value in range(1, N + 1)
            if self.is_valid_move((i, j), value, game_state)
        ]

        return valid_moves

    def is_opponent_close(self, square: Tuple[int, int], game_state: GameState) -> bool:
        """
        Checks if the opponent has squares adjacent (including diagonals) to the given square.
        """
        row, col = square
        adjacent_squares = [
            (row + dr, col + dc)
            for dr, dc in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
            if 0 <= row + dr < game_state.board.N and 0 <= col + dc < game_state.board.N
        ]
        opponent_squares = set(
            game_state.occupied_squares2
            if game_state.current_player == 1
            else game_state.occupied_squares1
        )
        return any(adj_square in opponent_squares for adj_square in adjacent_squares)

    def opponent_can_complete(self, game_state: GameState, move: Move) -> bool:
        """
        Checks if the opponent can complete a region immediately after the given move.
        Considers diagonally adjacent squares as well.
        """
        simulated_state = self.simulate_move(game_state, move)
        opponent_valid_moves = self.get_valid_moves(simulated_state)

        for opponent_move in opponent_valid_moves:
            next_state = self.simulate_move(simulated_state, opponent_move)
            if self.amount_of_regions_completed(next_state, opponent_move) > 0:
                return True  # Opponent can complete a region
        return False

    def get_valid_moves_with_heuristics(self, game_state: GameState):
        """
        Gets valid moves and prioritizes them based on heuristics, including prioritization for diagonal moves.
        """
        valid_moves = self.get_valid_moves(game_state)
        if not valid_moves:
            return []

        priority_order = {1: 1, 3: 2, 5: 3, 7: 4, 8: 5, 6: 6, 4: 7, 2: 8}
        prioritized_moves = []

        for move in valid_moves:
            square = move.square
            possible_values = [
                value
                for value in range(1, game_state.board.N + 1)
                if self.is_valid_move(square, value, game_state)
            ]
            num_options = len(possible_values)

            row, col = square
            is_diagonal = abs(row - col) <= 1
            if is_diagonal:
                priority = 0
            elif self.round_counter < 7:
                priority = priority_order.get(num_options, float("inf"))
            else:
                if num_options == 2 and not self.is_opponent_close(square, game_state):
                    if not self.opponent_can_complete(game_state, move):
                        priority = 0  # High priority if safe
                    else:
                        priority = float("inf")  # Low priority if unsafe
                else:
                    priority = priority_order.get(num_options, float("inf"))

            prioritized_moves.append((priority, move))

        prioritized_moves.sort(key=lambda x: x[0])
        return [move for _, move in prioritized_moves]

    def evaluate(self, game_state: GameState, ai_player_index: int):
        """
        Evaluates the game state with a heuristic based on the score and potential moves.
        """
        w1 = 0.9
        w2 = 0.1

        if ai_player_index == 0:
            return w1 * (game_state.scores[0] - game_state.scores[1]) + (
                w2
                * (len(game_state.allowed_squares1) - len(game_state.allowed_squares2))
            )

        if ai_player_index == 1:
            return w1 * (game_state.scores[1] - game_state.scores[0]) + (
                w2
                * (len(game_state.allowed_squares2) - len(game_state.allowed_squares1))
            )

    def simulate_move(self, game_state: GameState, move: Move):
        """
        Simulates a move and returns a new game state.
        """
        new_state = copy.deepcopy(game_state)
        new_state.board.put(move.square, move.value)
        new_state.moves.append(move)

        completed_regions = self.amount_of_regions_completed(new_state, move)

        # Update scores
        score_dict = {0: 0, 1: 1, 2: 3, 3: 7}
        new_state.scores[new_state.current_player - 1] += score_dict[completed_regions]

        # Switch players
        new_state.current_player = 3 - new_state.current_player
        return new_state

    def minimax(
        self,
        game_state: GameState,
        depth: int,
        alpha: int,
        beta: int,
        maximizing: bool,
        ai_player_index: int,
    ):
        """
        Minimax implementation with depth-limited search.
        """
        if depth == 0 or self.is_terminal(game_state):
            return self.evaluate(game_state, ai_player_index)

        valid_moves = self.get_valid_moves(game_state)

        if maximizing:
            max_eval = float("-inf")
            for move in valid_moves:
                next_state = self.simulate_move(game_state, move)
                eval = self.minimax(
                    next_state,
                    depth - 1,
                    alpha,
                    beta,
                    maximizing=False,
                    ai_player_index=ai_player_index,
                )
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in valid_moves:
                next_state = self.simulate_move(game_state, move)
                eval = self.minimax(
                    next_state,
                    depth - 1,
                    alpha,
                    beta,
                    maximizing=True,
                    ai_player_index=ai_player_index,
                )
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Determines the best move using heuristics, including strategic diagonal moves.
        """
        self.round_counter += 1
        valid_moves = self.get_valid_moves_with_heuristics(game_state)

        if not valid_moves:
            valid_moves = self.get_valid_moves(
                game_state
            )  # Fallback to any valid moves
            if not valid_moves:
                return

        for move in valid_moves:
            if self.round_counter < 6 or not self.opponent_can_complete(
                game_state, move
            ):
                self.propose_move(move)
                return

        best_move = None
        best_score = float("-inf")

        for move in valid_moves:
            next_state = self.simulate_move(game_state, move)
            score = self.minimax(
                next_state,
                depth=3,
                alpha=float("-inf"),
                beta=float("inf"),
                maximizing=False,
                ai_player_index=game_state.current_player - 1,
            )

            if score > best_score:
                best_score = score
                best_move = move

        if best_move:
            self.propose_move(best_move)
