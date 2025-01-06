# import time
# from competitive_sudoku.sudoku import Move, SudokuBoard, GameState
# import copy

# def solve_sudoku(game_state: GameState):
#     start_time = time.perf_counter()
#     board = copy.deepcopy(game_state.board)
#     N, m, n = board.N, board.m, board.n

#     # Sets for row/col/box constraints
#     row_sets = [set() for _ in range(N)]
#     col_sets = [set() for _ in range(N)]
#     box_sets = [set() for _ in range((N // n) * (N // m))]

#     def get_box_index(r, c):
#         return (r // m) * (N // n) + (c // n)

#     # Validate board & fill sets
#     for r in range(N):
#         for c in range(N):
#             val = board.get((r, c))
#             if val != 0:
#                 idx = get_box_index(r, c)
#                 if (val in row_sets[r] or
#                     val in col_sets[c] or
#                         val in box_sets[idx]):
#                     return False, {}
#                 row_sets[r].add(val)
#                 col_sets[c].add(val)
#                 box_sets[idx].add(val)

#     # We'll store the final solution in a dictionary:
#     #   { (row, col): value }
#     solution_dict = {}

#     def backtrack():
#         for r in range(N):
#             for c in range(N):
#                 if board.get((r, c)) == 0:
#                     box_idx = get_box_index(r, c)
#                     for candidate in range(1, N + 1):
#                         if (candidate not in row_sets[r] and
#                             candidate not in col_sets[c] and
#                                 candidate not in box_sets[box_idx]):

#                             # Place candidate
#                             board.put((r, c), candidate)
#                             row_sets[r].add(candidate)
#                             col_sets[c].add(candidate)
#                             box_sets[box_idx].add(candidate)

#                             # Add/update the solution dictionary
#                             solution_dict[(r, c)] = candidate

#                             # Recurse
#                             if backtrack():
#                                 return True

#                             # Undo
#                             board.put((r, c), 0)
#                             row_sets[r].remove(candidate)
#                             col_sets[c].remove(candidate)
#                             box_sets[box_idx].remove(candidate)
#                             # Remove it from dictionary
#                             if (r, c) in solution_dict:
#                                 del solution_dict[(r, c)]
#                     return False
#         return True

#     solved = backtrack()
#     if not solved:
#         return False, {}

#     # Now solution_dict has (row, col) -> value for each filled cell
#     total_time = time.perf_counter() - start_time
#     return True, solution_dict, total_time


import math
import time


import time
import copy
from competitive_sudoku.sudoku import GameState, SudokuBoard


def solve_sudoku(game_state: GameState):
    start_time = time.perf_counter()
    board = copy.deepcopy(game_state.board)
    N, m, n = board.N, board.m, board.n

    def get_box_index(r, c):
        return (r // m) * (N // n) + (c // n)

    # Track used digits in row, col, and box
    row_sets = [set() for _ in range(N)]
    col_sets = [set() for _ in range(N)]
    # for N = m*n, we have (N//n)*(N//m) boxes, which is also N.
    box_sets = [set() for _ in range(N)]

    # 1) Initialize row_sets, col_sets, and box_sets from the current board
    for r in range(N):
        for c in range(N):
            val = board.get((r, c))
            if val != 0:
                idx = get_box_index(r, c)
                if (val in row_sets[r] or
                    val in col_sets[c] or
                        val in box_sets[idx]):
                    # Invalid puzzle
                    return False, {}, 0.0
                row_sets[r].add(val)
                col_sets[c].add(val)
                box_sets[idx].add(val)

    # A dictionary to store (row, col) -> placed_value for the final solution
    solution_dict = {}

    # 2) A quick function to find legal candidates for a cell
    def legal_candidates(r, c):
        box_idx = get_box_index(r, c)
        used = row_sets[r] | col_sets[c] | box_sets[box_idx]
        return [x for x in range(1, N + 1) if x not in used]

    # 3) Function to pick the next cell using MRV: the cell with the fewest legal candidates
    def pick_cell_mrv():
        best_rc = None
        best_count = N + 1
        for rr in range(N):
            for cc in range(N):
                if board.get((rr, cc)) == SudokuBoard.empty:
                    # Count how many candidates
                    cands = legal_candidates(rr, cc)
                    count = len(cands)
                    if count < best_count:
                        best_count = count
                        best_rc = (rr, cc)
                        if count == 1:
                            # can't do better than a single candidate
                            return best_rc
        return best_rc

    # 4) Backtracking with forward checking + MRV
    def backtrack():
        cell = pick_cell_mrv()
        if cell is None:
            # No empty cell found => solved
            return True
        (r, c) = cell

        candidates = legal_candidates(r, c)
        if not candidates:
            return False

        for candidate in candidates:
            # Place candidate
            board.put((r, c), candidate)
            row_sets[r].add(candidate)
            col_sets[c].add(candidate)
            box_sets[get_box_index(r, c)].add(candidate)

            # Record it in solution_dict
            solution_dict[(r, c)] = candidate

            # Recurse
            if backtrack():
                return True

            # Undo
            board.put((r, c), SudokuBoard.empty)
            row_sets[r].remove(candidate)
            col_sets[c].remove(candidate)
            box_sets[get_box_index(r, c)].remove(candidate)
            if (r, c) in solution_dict:
                del solution_dict[(r, c)]

        return False

    # 5) Solve and measure time
    solved = backtrack()
    total_time = time.perf_counter() - start_time
    if not solved:
        return False, {}, total_time

    return True, solution_dict, total_time
