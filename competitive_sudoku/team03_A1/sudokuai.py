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
    Sudoku AI that computes a move for a given sudoku configuration using Minimax.
    """

    def __init__(self):
        super().__init__()

    def amount_of_regions_completed(self, game_state: GameState, move: Move):
        """
        Checks how many regions (rows, columns, blocks) are completed by the move.
        """
        completed = 0
        N = game_state.board.N
        row, col = move.square

        # Check row
        if all(game_state.board.get((row, c)) != SudokuBoard.empty for c in range(N)):
            completed += 1

        # Check column
        if all(game_state.board.get((r, col)) != SudokuBoard.empty for r in range(N)):
            completed += 1

        # Check block
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

    def simulate_move(self, game_state: GameState, move: Move):
        """
        Simulates a move and returns a new game state.
        """
        score_dict = {0: 0, 1: 1, 2: 3, 3: 7}

        new_state = copy.deepcopy(game_state)
        new_state.board.put(move.square, move.value)
        new_state.moves.append(move)

        completed_regions = self.amount_of_regions_completed(new_state, move)
        
        new_state.scores[new_state.current_player - 1] += score_dict[completed_regions]

        new_state.current_player = (3 - new_state.current_player)
        return new_state

    def get_valid_moves(self, game_state: GameState):
        """
        Gets the valid moves for the current Game State
        """
        N = game_state.board.N

        def is_valid_move(square, value):
            return (
                game_state.board.get(square) == SudokuBoard.empty
                and not TabooMove(square, value) in game_state.taboo_moves
                and (
                    square in game_state.player_squares()
                    if game_state.player_squares() is not None
                    else True
                )
                and value
                not in [game_state.board.get((square[0], col)) for col in range(N)]
                and value
                not in [game_state.board.get((row, square[1])) for row in range(N)]
                and value not in self.get_region_values(game_state.board, square)
            )

        valid_moves = [
            Move((i, j), value)
            for i in range(N)
            for j in range(N)
            for value in range(1, N + 1)
            if is_valid_move((i, j), value)
        ]
        return valid_moves

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


    def is_terminal(self, game_state: GameState):
        """
        Checks if the game state is terminal (no valid moves left).
        """
        return len(self.get_valid_moves(game_state)) == 0


    def evaluate(self, game_state: GameState, ai_player_index: int):
        """
        Evaluates the game state with a heuristic based on the score and potential moves.
        """
        w1 = 0.9
        w2 = 0.1

        if ai_player_index == 0:
            return (w1 * (game_state.scores[0] - game_state.scores[1]) + (w2 * (len(game_state.allowed_squares1) - len(game_state.allowed_squares2))))
        
        if ai_player_index == 1:
            return (w1 * (game_state.scores[1] - game_state.scores[0]) + (w2 * (len(game_state.allowed_squares2) - len(game_state.allowed_squares1))))


    def minimax(self, game_state: GameState, depth: int, alpha: int, beta: int, maximizing: bool, ai_player_index: int):
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
                eval = self.minimax(next_state, depth - 1, alpha, beta, maximizing=False, ai_player_index=ai_player_index)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in valid_moves:
                next_state = self.simulate_move(game_state, move)
                eval = self.minimax(next_state, depth - 1, alpha, beta, maximizing=True, ai_player_index=ai_player_index)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Minimax with iterative deepening depth  # ToDo - Caching if needed
        """
        ai_player_index = game_state.current_player-1
        depth = (game_state.board.board_height() * game_state.board.board_width()) - (len(game_state.occupied_squares1) + len(game_state.occupied_squares2)) # Total number of unoccupied squares
       
        valid_moves = self.get_valid_moves(game_state)

        best_move = None
        best_score = float("-inf")

        for depth in range(1, depth + 1):  
            depth_move_scores = []

            for move in valid_moves:
                next_state = self.simulate_move(game_state, move)
                score = self.minimax(next_state, depth, float("-inf"), float("inf"), maximizing=False, ai_player_index=ai_player_index) 
                depth_move_scores.append((move, score))

                if score >= best_score:
                    best_score = score
                    best_move = move
                if best_move:
                    self.propose_move(best_move)
            
            valid_moves = [i[0] for i in sorted(depth_move_scores, key=lambda x: x[1], reverse=True)] 
