from competitive_sudoku.sudoku import SudokuBoard, GameState, Move, TabooMove
import copy
from typing import Tuple

def get_region_values(board: SudokuBoard, square: Tuple[int, int]):
    """
    Gets the values in the region corresponding to the given square.
    """
    region_width = board.region_width()
    region_height = board.region_height()
    start_row = (square[0] // region_height) * region_height
    start_col = (square[1] // region_width) * region_width

    region_values = [
        board.get((r, c))
        for r in range(start_row, start_row + region_height)
        for c in range(start_col, start_col + region_width)
    ]
    return region_values


def get_valid_moves(game_state):
    N = game_state.board.N
    board = game_state.board
    player_squares = set(game_state.player_squares())
    region_height = board.region_height()
    region_width = board.region_width()

    row_values = [set(board.get((i, col)) for col in range(
        N) if board.get((i, col)) != SudokuBoard.empty) for i in range(N)]
    col_values = [set(board.get((row, j)) for row in range(
        N) if board.get((row, j)) != SudokuBoard.empty) for j in range(N)]
    region_values = {}
    for i in range(0, N, region_height):
        for j in range(0, N, region_width):
            region = (i // region_height, j // region_width)
            region_values[region] = set(
                board.get((r, c))
                for r in range(i, i + region_height)
                for c in range(j, j + region_width)
                if board.get((r, c)) != SudokuBoard.empty
            )

    def possible(i, j, value):
        region = (i // region_height, j // region_width)

        return (
            board.get((i, j)) == SudokuBoard.empty
            and TabooMove((i, j), value) not in game_state.taboo_moves
            and value not in row_values[i]
            and value not in col_values[j]
            and value not in region_values[region]
        )

    valid_moves = {}
    for (i,j) in player_squares:
        for value in range(1, N+1):
            if possible(i,j,value):
                if (i,j) not in valid_moves.keys():
                    valid_moves[(i,j)] = [value]
                else:
                    valid_moves[(i, j)].append(value)
    return valid_moves


def amount_of_regions_completed(game_state: GameState, move: Move):
    """
    Checks how many regions (rows, columns, blocks) are completed by the move.
    """
    completed = 0
    N = game_state.board.N
    row, col = move.square

    # Check row
    if all(game_state.board.get((row, c)) != SudokuBoard.empty for c in range(N)):
        completed += 1

    # Check column
    if all(game_state.board.get((r, col)) != SudokuBoard.empty for r in range(N)):
        completed += 1

    # Check block
    region_width = game_state.board.region_width()
    region_height = game_state.board.region_height()
    start_row = (row // region_height) * region_height
    start_col = (col // region_width) * region_width
    if all(
        game_state.board.get((r, c)) != SudokuBoard.empty
        for r in range(start_row, start_row + region_height)
        for c in range(start_col, start_col + region_width)
    ):
        completed += 1

    return completed


def simulate_move(game_state: GameState, move: Move, ai_player_index):
    """
    Simulates a move and updates allowed squares and occupied squares correctly.
    """
    score_dict = {0: 0, 1: 1, 2: 3, 3: 7}
    new_state = copy.deepcopy(game_state)
    new_state.board.put(move.square, move.value)
    new_state.moves.append(move)

    if new_state.current_player == 1:
        new_state.occupied_squares1.append(move.square)
    else:
        new_state.occupied_squares2.append(move.square)


    completed_regions = amount_of_regions_completed(new_state, move)
    new_state.scores[new_state.current_player -
                     1] += score_dict[completed_regions]

    # Switch the current player
    new_state.current_player = 3 - new_state.current_player

    return new_state
