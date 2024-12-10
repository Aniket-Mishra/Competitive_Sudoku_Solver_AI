from competitive_sudoku.sudoku import SudokuBoard, TabooMove, GameState, Move
from typing import Dict, List, Tuple


def naked_singles(game_state: GameState, valid_moves: Dict) -> Dict:
    """
    Calculate naked singles
    If a single square is left in a region/row/col
    We know what value it iwll take
    So we remove said value from remaining dependent regions
    And add it to taboo moves.

    Args:
        game_state (GameState): Current game state
        valid_moves (Dict): Dict of calid moves

    Returns:
        Dict: Dict of valid moves post naked singles
    """
    N = game_state.board.N

    for naked_single in valid_moves:
        if len(valid_moves[naked_single]) == 1:
            single_row = naked_single[0]
            single_column = naked_single[1]
            single_value = valid_moves[naked_single][0]

            for i in range(N):
                if (
                    i != single_row
                    and (i, single_column) in valid_moves
                    and single_value in valid_moves[(i, single_column)]
                ):
                    valid_moves[(i, single_column)].remove(single_value)
                    game_state.taboo_moves.append(
                        TabooMove((i, single_column), single_value)
                    )

            for i in range(N):
                if (
                    i != single_column
                    and (single_row, i) in valid_moves
                    and single_value in valid_moves[(single_row, i)]
                ):
                    valid_moves[(single_row, i)].remove(single_value)
                    game_state.taboo_moves.append(
                        TabooMove((single_row, i), single_value)
                    )
    return valid_moves


def hidden_singles(game_state: GameState, valid_moves: Dict) -> Dict:
    """
        Validates existing `valid_moves` by dynamically checking for hidden singles
        In rows, columns, and regions.
        Remove any moves that conflict with
        Hidden singles.

    Args:
        game_state (GameState): Current gamestate
        valid_moves (Dict): A dictionary of already computed valid moves for the player's reach.

    Returns:
        Dict: Valid moves dict
    """
    N = game_state.board.N
    board = game_state.board
    region_width = board.region_width()
    region_height = board.region_height()

    for move in valid_moves:
        hidden_row = move[0]
        hidden_col = move[1]
        hidden_val = valid_moves[move]

        col_values = []
        for row in range(N):
            col_values.append(game_state.board.get((row, hidden_col)))
            print(game_state.board.get((row, hidden_col)))
        print("-----")

    return valid_moves
