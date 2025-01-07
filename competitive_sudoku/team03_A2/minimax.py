from competitive_sudoku.sudoku import GameState, Move
from team03_A2.helper_functions import simulate_move, get_valid_moves
from team03_A2.evaluation_functions import (
    score_center_moves,
    score_difference,
    score_not_reachable_by_opponent,
)


def minimax(game_state: GameState, depth: int, alpha: int, beta: int, maximizing: bool, ai_player_index: int,) -> float:
    """
    Minimax implementation with depth-limited search.

    Args:
        game_state (GameState): Current Game state
        depth (int): depth of the minimax recursion
        alpha (int): maximiser
        beta (int): minimiser
        maximizing (bool): Maximising player or not
        ai_player_index (int): Index denoting our agent

    Returns:
        float: evaluation of the minimax
    """
    valid_moves_dict = get_valid_moves(game_state)

    valid_moves = [
        Move((row, col), value)
        for (row, col), values in valid_moves_dict.items()
        for value in values
    ]

    if depth == 0 or is_terminal(game_state):
        return evaluate(game_state, ai_player_index)

    if maximizing:
        max_eval = float("-inf")
        for move in valid_moves:

            next_state = simulate_move(game_state, move, ai_player_index)
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
            next_state = simulate_move(game_state, move, ai_player_index)
            eval = minimax(next_state, depth - 1, alpha, beta,
                           maximizing=True, ai_player_index=ai_player_index,)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def is_terminal(game_state: GameState) -> bool:
    """
    Checks if the game state is terminal (no valid moves left).

    Args:
        game_state (GameState): Current Game state

    Returns:
        bool: Is last terminal state or not
    """
    valid_moves_dict = get_valid_moves(game_state)
    valid_moves = [move for moves in valid_moves_dict.values() for move in moves]
    return len(valid_moves) == 0


def evaluate(game_state: GameState, ai_player_index: int,) -> float:
    """
    Evaluates the game state with a heuristic based on the score, center scores
    and reachability of the moves.

    Args:
        game_state (GameState): Current game state
        ai_player_index (int): our agent index

    Returns:
        float: Weighted score based on the center scores, point scores and reachability scores
    """
    w1 = 0.5
    w2 = 0.5
    w3 = 1
    center_scores = score_center_moves(game_state, ai_player_index)
    point_scores = score_difference(game_state, ai_player_index)
    opponent_reachable_scores = -score_not_reachable_by_opponent(game_state, ai_player_index)
    return w1 * center_scores + w2 * point_scores + w3 * opponent_reachable_scores
