from competitive_sudoku.sudoku import GameState


def score_difference(game_state: GameState, ai_player_index: int) -> float:
    """
    Calculate game state

    Args:
        game_state (GameState): Current game state
        ai_player_index (int): index of our agent

    Returns:
        float: score diff of our agent vs opponent
    """
    return game_state.scores[ai_player_index] - game_state.scores[1 - ai_player_index]


def score_center_moves(game_state: GameState, ai_player_index) -> float:
    """
    Calculates the score for a node, rewarding moves closer to the center.
    Always evaluates from the perspective of the AI player.

    Args:
        game_state (GameState): Current game state
        ai_player_index (int): index of our agent

    Returns:
        float: score of the move
    """
    row_weight = 1
    col_weight = 0.5

    center = (game_state.board.N - 1) / 2

    ai_squares = (
        game_state.occupied_squares1
        if ai_player_index == 0
        else game_state.occupied_squares2
    )

    row_prox = 0
    col_prox = 0
    for row, col in ai_squares:
        row_prox += abs(row - center)
        col_prox += abs(col - center)

    num_squares = len(ai_squares)
    if num_squares > 0:
        row_prox /= num_squares
        col_prox /= num_squares

    return -(row_weight * col_weight + col_weight * col_prox)


def score_not_reachable_by_opponent(game_state: GameState, ai_player_index: int) -> float:
    """
    Computes a score that rewards squares the AI can access and also the opponent can access..
    
    The function evaluates how many of the AI's allowed squares are reachable by the opponent.
    The score penalizes squares that the opponent can not access and rewards those that are accessible to both players.
    
    Parameters:
    - game_state (GameState): The current state of the Sudoku game.
    - ai_player_index (int): index of our agent.
    
    Returns:
    - float: A normalized score, where a higher value indicates that more squares are reachable by both players.
    """
    original_player = game_state.current_player

    if ai_player_index == 0:
        game_state.current_player = 1
        ai_allowed_squares = game_state.player_squares()
        game_state.current_player = 2
        opponent_allowed_squares = game_state.player_squares()
    else:
        game_state.current_player = 2
        ai_allowed_squares = game_state.player_squares()
        game_state.current_player = 1
        opponent_allowed_squares = game_state.player_squares()

    game_state.current_player = original_player

    score = 0
    max_score = 0
    for square in ai_allowed_squares:
        max_score += 1
        if square in opponent_allowed_squares:
            score += 1
        else:
            score -= 6

    if max_score > 0:
        normalized_score = score / max_score
    else:
        normalized_score = 0
    return normalized_score
