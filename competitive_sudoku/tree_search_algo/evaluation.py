from team21_A2.helper_functions import (
    check_legal_column,
    check_legal_region,
    check_legal_row,
)


def evaluate_move(game_board, move, our_move, score):
    """
    Returns what the score would be after we play a move
    @param game_board: The current SudokuBoard of the Competitive Sudoku game
    @param move: The current GameState of the Competitive Sudoku game
    @param our_move: Whether it is currently our turn, or our opponents' turn.
    @param score: The score of our agent minus the score of the opposing agent before the move is played
    """

    count = 0
    if check_legal_row(
        game_board, move.square[0], 0
    ):  # An agent scores points by completing a row on the board
        count += 1
    if check_legal_column(
        game_board, move.square[1], 0
    ):  # An agent scores points by completing a column on the board
        count += 1
    if check_legal_region(
        game_board, move.square[0], move.square[1], 0
    ):  # An agent scores points by completing a region on the board
        count += 1

    if count == 0:  # If an agent completes 0 rows/columns/regions, they get 0 points
        player_score = 0
    elif count == 1:  # If an agent completes 1 rows/columns/regions, they get 1 points
        player_score = 1
    elif count == 2:  # If an agent completes 2 rows/columns/regions, they get 3 points
        player_score = 3
    else:  # If an agent completes 3 rows/columns/regions, they get 7 points
        player_score = 7

    if (
        our_move
    ):  # If it is our agents' turn, points scored should be added to our score parameter
        return score + player_score
    else:  # If it is the opposing agents' turn, points scored should be subtracted from our score parameter
        return score - player_score
