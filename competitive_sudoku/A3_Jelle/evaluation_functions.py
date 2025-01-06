from competitive_sudoku.sudoku import GameState


def score_difference(game_state, ai_player_index):
    return game_state.scores[ai_player_index] - game_state.scores[1 - ai_player_index]


def score_center_moves(game_state: GameState, ai_player_index):
    """
    Calculates the score for a node, rewarding moves closer to the center.
    Always evaluates from the perspective of the AI player.
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

    return -(row_weight * col_weight + col_weight * col_prox)



def score_not_reachable_by_opponent(
    game_state: GameState, ai_player_index: int
) -> float:
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
            score -= 7

    if max_score > 0:
        normalized_score = score / max_score
    else:
        normalized_score = 0
    return normalized_score
