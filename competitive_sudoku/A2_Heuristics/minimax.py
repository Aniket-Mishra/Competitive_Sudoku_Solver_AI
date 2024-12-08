from competitive_sudoku.sudoku import GameState, Move
from A2_Heuristics.helper_functions import simulate_move, get_valid_moves
from A2_Heuristics.taboo_helpers import naked_singles

def minimax(game_state: GameState, depth: int, alpha: int, beta: int, maximizing: bool, ai_player_index: int,):
    """
    Minimax implementation with depth-limited search.
    """
    valid_moves_dict = get_valid_moves(game_state)
    valid_moves = naked_singles(game_state, valid_moves_dict)

    valid_moves = [Move((row, col), value) for (row, col),
                   values in valid_moves_dict.items() for value in values]

    if depth == 0 or is_terminal(game_state):
        return evaluate(game_state, ai_player_index)

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
    valid_moves_dict = get_valid_moves(game_state)
    valid_moves = [move for moves in valid_moves_dict.values()
                   for move in moves]
    return len(valid_moves) == 0


def evaluate(game_state: GameState,
             ai_player_index: int,
             ):
    """
    Evaluates the game state with a heuristic based on the score, potential moves,
    and the priority of the move being considered.
    """
    w1 = 0.9  # Weight for score difference
    w2 = 0.1  # Weight for allowed squares difference

    score_diff = (game_state.scores[ai_player_index] -
                  game_state.scores[1 - ai_player_index])

    if ai_player_index == 0:
        allowed_squares_diff = len(game_state.allowed_squares1) - len(
            game_state.allowed_squares2)
    else:
        allowed_squares_diff = len(game_state.allowed_squares2) - len(
            game_state.allowed_squares1)

    return w1 * score_diff + w2 * allowed_squares_diff
