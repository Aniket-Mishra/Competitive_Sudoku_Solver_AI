from competitive_sudoku.sudoku import GameState, Move, SudokuBoard
from A3_Jelle.helper_functions import simulate_move, get_valid_moves, get_illegal_moves
from A3_Jelle.taboo_helpers import naked_singles
from A3_Jelle.evaluation_functions import (
    score_center_moves,
    score_difference,
    score_not_reachable_by_opponent,
)


def minimax(
    game_state: GameState,
    depth: int,
    alpha: int,
    beta: int,
    maximizing: bool,
    ai_player_index: int,
    illegal_moves
):
    """
    Minimax implementation with depth-limited search.
    """
    N = game_state.board.N
    valid_moves = []
    for square in game_state.player_squares():
        for i in range(1, N+1):
            if (square, i) not in illegal_moves and game_state.board.get(square) == SudokuBoard.empty and Move(square, i) not in game_state.taboo_moves:
                valid_moves.append(Move(square, i))

    if depth == 0 or is_terminal(game_state):
        return evaluate(game_state, ai_player_index)

    if maximizing:
        max_eval = float("-inf")
        for move in valid_moves:

            next_state = simulate_move(game_state, move)
            new_illegal_moves = get_illegal_moves(game_state, move)

            eval = minimax(
                next_state,
                depth - 1,
                alpha,
                beta,
                maximizing=False,
                ai_player_index=ai_player_index,
                illegal_moves=new_illegal_moves
            )
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for move in valid_moves:

            next_state = simulate_move(game_state, move)
            new_illegal_moves = get_illegal_moves(game_state, move)

            eval = minimax(
                next_state,
                depth - 1,
                alpha,
                beta,
                maximizing=True,
                ai_player_index=ai_player_index,
                illegal_moves=new_illegal_moves
            )
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def is_terminal(game_state: GameState):
    """
    Checks if the game state is terminal (no valid moves left).
    """
    valid_moves_dict = get_valid_moves(game_state)
    valid_moves = [move for moves in valid_moves_dict.values()
                   for move in moves]
    return len(valid_moves) == 0


def evaluate(
    game_state: GameState,
    ai_player_index: int,
):
    """
    Evaluates the game state with a heuristic based on the score, potential moves,
    and the priority of the move being considered.
    """
    w1 = 0.5
    w2 = 0.5
    w3 = 1
    center_scores = score_center_moves(game_state, ai_player_index)
    point_scores = score_difference(game_state, ai_player_index)
    opponent_reachable_scores = -score_not_reachable_by_opponent(
        game_state, ai_player_index
    )

    return w1 * center_scores + w2 * point_scores + w3 * opponent_reachable_scores
