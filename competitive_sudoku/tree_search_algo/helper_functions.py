from competitive_sudoku.sudoku import Move, SudokuBoard, TabooMove
import copy


def check_legal_row(board: SudokuBoard, m, value):
    """
    Returns whether a value is legal for a row.
    @param board: A sudoku board. It contains the current position of a game.
    @param m: A row value in the range [0, ..., N).
    @param value: A value for a square.
    """
    row = []

    for n in range(board.N):
        row.append(board.get((m, n)))

    return value not in row


def check_legal_column(board: SudokuBoard, n, value):
    """
    Returns whether a value is legal for a column.
    @param board: A sudoku board. It contains the current position of a game.
    @param n: A column value in the range [0, ..., N).
    @param value: A value for a square.
    """
    column = []

    for m in range(board.N):
        column.append(board.get((m, n)))

    return value not in column


def check_legal_region(board: SudokuBoard, m, n, value):
    """
    Returns whether a value is legal for a region.
    @param board: A sudoku board. It contains the current position of a game.
    @param m: A row value in the range [0, ..., N).
    @param n: A column value in the range [0, ..., N).
    @param value: The number that we want to put on the board
    """
    row_region_index = (m // board.m) * board.m
    column_region_index = (n // board.n) * board.n
    region = []

    for m_i in range(row_region_index, row_region_index + board.m):
        for n_i in range(column_region_index, column_region_index + board.n):
            region.append(board.get((m_i, n_i)))

    return value not in region


def is_empty_cell(board: SudokuBoard, m, n):
    """
    Checks if current coordinate is empty (has value 0).
    @param board: A sudoku board. It contains the current position of a game.
    @param m: A row value in the range [0, ..., N)
    @param n: A column value in the range [0, ..., N)
    """
    return board.get((m, n)) == 0


def is_legal_move(game_board: SudokuBoard, taboo_moves, m, n, value):
    """
    Checks if the current move would disqualify you from the game
    @param game_board: The current Sudoku board of type SudokuBoard
    @param taboo_moves: The list of current taboo moves
    @param m: A row value in the range [0, ..., N)
    @param n: A column value in the range [0, ..., N)
    @param value: A value in the range [1, ..., N]
    """
    move = TabooMove((m, n), value)
    return (
        move not in taboo_moves
        and check_legal_row(game_board, m, value)
        and check_legal_column(game_board, n, value)
        and check_legal_region(game_board, m, n, value)
    )


def get_legal_moves(game_board: SudokuBoard, taboo_moves):
    """
    A method to calculate all legal moves from a GameState return a list of legal moves.
    @param game_board: The current Sudoku board of type SudokuBoard
    @param taboo_moves: The list of current taboo moves
    """
    legal_moves = []

    # Go over all possible index combination of the board
    for i in range(game_board.N):
        for j in range(game_board.N):
            # if the cell is not empty we can skip evaluating values
            if not is_empty_cell(game_board, i, j):
                continue
            else:
                # Try all possible values [1, ..., N] for a cell
                for k in range(1, game_board.N + 1):
                    # If a move is legal append to list of legal moves
                    if is_legal_move(game_board, taboo_moves, i, j, k):
                        cur_move = Move((i, j), k)
                        legal_moves.append(cur_move)

    return legal_moves


def calculate_new_game_board(game_board: SudokuBoard, move: Move):
    """
    Calculates the changes a move would make to the Sudoku board
    @param game_board: The current Sudoku board of type SudokuBoard
    @param move: The current move of type Move)
    """
    new_game_board = copy.deepcopy(game_board)
    new_game_board.put(move.square, move.value)
    return new_game_board


def is_board_full(board: SudokuBoard):
    """
    Checks if the board is filled up
    (so we know we can't make any moves anymore, this eliminates a possible error because of no possible moves)
    @param board: The current sudoku board of type SudokuBoard
    """
    if 0 in board.squares:
        return False
    else:
        return True
