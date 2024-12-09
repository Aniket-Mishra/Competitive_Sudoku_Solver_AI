class GenericSudokuSolver:
    def __init__(self, gamestate):
        """
        Initialize the solver with a given gamestate.
        :param gamestate: A gamestate object that includes the board and its dimensions.
        """
        self.board = gamestate.board
        self.N = gamestate.N  # Size of the board (NxN)
        self.block_height = gamestate.block_height  # Height of each block
        self.block_width = gamestate.block_width  # Width of each block

    def is_valid(self, row, col, num):
        """Check if placing `num` in `board[row][col]` is valid."""
        for i in range(self.N):
            if self.board[row][i] == num or self.board[i][col] == num:
                return False

        start_row, start_col = (row // self.block_height) * \
            self.block_height, (col // self.block_width) * self.block_width
        for i in range(self.block_height):
            for j in range(self.block_width):
                if self.board[start_row + i][start_col + j] == num:
                    return False

        return True

    def get_possible_values(self, row, col):
        """Return a set of possible values for the cell (row, col)."""
        if self.board[row][col] != 0:
            return set()

        possible_values = set(range(1, self.N + 1))
        for i in range(self.N):
            possible_values.discard(self.board[row][i])
            possible_values.discard(self.board[i][col])

        start_row, start_col = (row // self.block_height) * \
            self.block_height, (col // self.block_width) * self.block_width
        for i in range(self.block_height):
            for j in range(self.block_width):
                possible_values.discard(
                    self.board[start_row + i][start_col + j])

        return possible_values

    def solve(self, candidates=None):
        """Backtracking solver for the generic Sudoku board."""
        if candidates is None:
            candidates = [[self.get_possible_values(
                r, c) for c in range(self.N)] for r in range(self.N)]

        # Find the cell with the fewest candidates
        min_candidates = self.N + 1
        best_cell = None
        for r in range(self.N):
            for c in range(self.N):
                if self.board[r][c] == 0 and len(candidates[r][c]) < min_candidates:
                    min_candidates = len(candidates[r][c])
                    best_cell = (r, c)

        if not best_cell:
            return True  # Solved

        row, col = best_cell
        for num in candidates[row][col]:
            if self.is_valid(row, col, num):
                self.board[row][col] = num

                # Recompute candidates after placing a number
                new_candidates = [[set(c) for c in row] for row in candidates]
                for r in range(self.N):
                    new_candidates[r][col].discard(num)
                for c in range(self.N):
                    new_candidates[row][c].discard(num)
                start_row, start_col = (
                    row // self.block_height) * self.block_height, (col // self.block_width) * self.block_width
                for i in range(self.block_height):
                    for j in range(self.block_width):
                        new_candidates[start_row +
                                       i][start_col + j].discard(num)

                if self.solve(new_candidates):
                    return True

                # Undo move
                self.board[row][col] = 0

        return False

    def find_values_keeping_solvable(self):
        """
        Return a dictionary mapping each empty cell to the set of values 
        that keep the Sudoku solvable for a generic board.
        """
        valid_values = {}
        for r in range(self.N):
            for c in range(self.N):
                if self.board[r][c] == 0:
                    possible_values = self.get_possible_values(r, c)
                    valid_for_cell = set()
                    for value in possible_values:
                        test_board = [row[:] for row in self.board]
                        test_board[r][c] = value
                        solver = GenericSudokuSolver(
                            gamestate=self._create_gamestate(test_board))
                        if solver.solve():
                            valid_for_cell.add(value)
                    valid_values[(r, c)] = valid_for_cell
        return valid_values

    def _create_gamestate(self, board):
        """
        Helper function to create a gamestate object from a board.
        This function assumes gamestate structure is well-defined.
        """
        class GameState:
            def __init__(self, board, block_height, block_width):
                self.board = board
                self.N = len(board)
                self.block_height = block_height
                self.block_width = block_width

        return GameState(board, self.block_height, self.block_width)

# Example usage


class GameState:
    def __init__(self, board, block_height, block_width):
        self.board = board
        self.N = len(board)
        self.block_height = block_height
        self.block_width = block_width


example_grid = [
    [0, 0, 4, 0],
    [0, 4, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 2, 0],
]

gamestate = GameState(example_grid, block_height=2, block_width=2)
solver = GenericSudokuSolver(gamestate)
result = solver.find_values_keeping_solvable()

print("Board with values keeping it solvable:")
for cell, values in result.items():
    print(f"Cell {cell}: {values}")
