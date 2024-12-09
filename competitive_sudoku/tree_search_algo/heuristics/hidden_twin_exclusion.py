from competitive_sudoku.sudoku import SudokuBoard


def calculate_region_index(board: SudokuBoard, m, n):
    row_region_index = (m // board.m) * board.m
    column_region_index = (n // board.n) * board.n

    return row_region_index, column_region_index


def hidden_twin_exclusion(board: SudokuBoard, moves: list):
    """
    Returns a filtered list of moves using the hidden twin exclusion.
    @param board: A sudoku board. It contains the current position of a game.
    @param moves: A list of moves to evaluate.
    """
    filtered_moves = []
    new_taboo_result = None
    potential_twins = {}
    twins = {}

    # Store the moves in a dictionary where the key is the position of the move
    # and the value is a list of possible values
    for m in moves:
        position = (m.square[0], m.square[1])
        if position in potential_twins:
            potential_twins[position].append(m.value)
        else:
            potential_twins[position] = [m.value]

    # Check if the potential twins are valid by comparing for each position the possible values
    for k, v in potential_twins.items():
        for k_, v_ in potential_twins.items():
            # Check if the two positions are not the same and that they are in the same region
            if k != k_ and calculate_region_index(
                board, k[0], k[1]
            ) == calculate_region_index(board, k_[0], k_[1]):
                intersection = set(v) & set(v_)
                if len(intersection) == 2:
                    twins[k] = list(intersection)
                    twins[k_] = list(intersection)
            else:
                continue

    # Filter out moves using the twins
    for move in moves:
        key = (move.square[0], move.square[1])
        if key in twins:
            if move.value in twins[key]:
                filtered_moves.append(move)
            elif new_taboo_result is not None:
                # We have found a move that will get rejected by the Oracle and will be placed on the taboo list
                # This is very valuable, so we store it (But we don't need more than 1)
                new_taboo_result = move
        else:
            filtered_moves.append(move)

    return filtered_moves, new_taboo_result
