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

def get_illegal_moves(game_state, last_move=None):
    board = game_state.board
    N = board.N

    illegal_moves = []
    if last_move is not None:
        row, col = last_move.square
        value = last_move.value


        for i in range(1, N+1):
            illegal_moves.append(((row,col), i))

        # Add all squares in the same row
        for c in range(N):
            if (row, c) != (row, col):  # Skip the square where the move was just played
                illegal_moves.append(((row, c), value))

        # Add all squares in the same column
        for r in range(N):
            if (r, col) != (row, col):  # Skip the square where the move was just played
                illegal_moves.append(((r, col), value))
    
        # Add all squares in the same region
        region_start_row = (row // board.region_height()) * board.region_height()
        region_start_col = (col // board.region_width()) * board.region_width()
        for r in range(region_start_row, region_start_row + board.region_height()):
            for c in range(region_start_col, region_start_col + board.region_width()):
                if (r, c) != (row, col):  # Skip the square where the move was just played
                    illegal_moves.append(((r, c), value))

    opponent_occupied_squares = (
        game_state.occupied_squares2 if game_state.current_player == 1 else game_state.occupied_squares1
    )
    for square in opponent_occupied_squares:
        r, c = square
        v = board.get(square)
        if v != SudokuBoard.empty:  # Only add if there's a value
            # Add all row, column, and region constraints for opponent's value
            for i in range(N):
                illegal_moves.append(((r, i), v))  # Same row
                illegal_moves.append(((i, c), v))  # Same column

            region_start_row = (r // board.region_height()
                                ) * board.region_height()
            region_start_col = (c // board.region_width()
                                ) * board.region_width()
            for rr in range(region_start_row, region_start_row + board.region_height()):
                for cc in range(region_start_col, region_start_col + board.region_width()):
                    illegal_moves.append(((rr, cc), v))  # Same region
                    
    return set(illegal_moves)

    

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
    for (i, j) in player_squares:
        for value in range(1, N+1):
            if possible(i, j, value):
                if (i, j) not in valid_moves.keys():
                    valid_moves[(i, j)] = [value]
                else:
                    valid_moves[(i, j)].append(value)
    return valid_moves


def naked_singles(game_state: GameState, valid_moves):
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


def simulate_move(game_state: GameState, move: Move):
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
