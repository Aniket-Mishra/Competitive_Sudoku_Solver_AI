from competitive_sudoku.sudoku import SudokuBoard, TabooMove, GameState, Move


def naked_singles(game_state: GameState, valid_moves):
    N = game_state.board.N

    for hidden_single in valid_moves:
        if len(valid_moves[hidden_single]) == 1:
            single_row = hidden_single[0]
            single_column = hidden_single[1]
            single_value = valid_moves[hidden_single][0]

            for i in range(N):
                if i != single_row and (i, single_column) in valid_moves and single_value in valid_moves[(i, single_column)]:
                    valid_moves[(i, single_column)].remove(single_value)
                    game_state.taboo_moves.append(
                        TabooMove((i, single_column), single_value))

            for i in range(N):
                if i != single_column and (single_row, i) in valid_moves and single_value in valid_moves[(single_row, i)]:
                    valid_moves[(single_row, i)].remove(single_value)
                    game_state.taboo_moves.append(
                        TabooMove((single_row, i), single_value))
    return valid_moves


def hidden_singles(game_state: GameState, valid_moves):
    """
    Find values which has to be placed in a certain cell, we check this as follows:
    check possible values in the same row, same column and in the same box. 
    If one of these groups constraint the cell such that it can only take 1 value, it is a hidden single
    """
