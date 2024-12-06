from competitive_sudoku.sudoku import GameState
from A2_Heuristics.helper_functions import simulate_move, get_valid_moves


def minimax(game_state: GameState, depth: int, alpha: int, beta: int, maximizing: bool, ai_player_index: int,):
    """
    Minimax implementation with depth-limited search.
    """
    if depth == 0 or is_terminal(game_state):
        return evaluate(game_state, ai_player_index)

    valid_moves, _ = get_valid_moves(game_state)

    if maximizing:
        max_eval = float("-inf")
        for move in valid_moves:

            next_state = simulate_move(game_state, move)
            eval = minimax(next_state, depth - 1, alpha, beta,
                           maximizing=False, ai_player_index=ai_player_index,)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for move in valid_moves:
            next_state = simulate_move(game_state, move)
            eval = minimax(next_state, depth - 1, alpha, beta,
                           maximizing=True, ai_player_index=ai_player_index,)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def is_terminal(game_state: GameState):
    """
    Checks if the game state is terminal (no valid moves left).
    """
    valid_moves, _ = get_valid_moves(game_state)
    return len(valid_moves) == 0


def evaluate(game_state: GameState, ai_player_index: int):
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
