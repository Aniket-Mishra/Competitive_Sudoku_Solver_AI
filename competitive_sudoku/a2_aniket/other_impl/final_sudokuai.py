#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import copy
import io
import random
from typing import List, Tuple, Union, Any, Optional, Iterator

# A square consists of a row and column index. Both are zero-based.
Square = Tuple[int, int]


class SudokuSettings(object):
    print_ascii_states: bool = False  # Print game states in ascii format


class Move(object):
    """A Move is a tuple (square, value) that represents the action board.put(square, value) for a given
    sudoku configuration board."""

    def __init__(self, square: Square, value: int):
        """
        Constructs a move.
        @param square: A square with coordinates in the range [0, ..., N)
        @param value: A value in the range [1, ..., N]
        """
        self.square = square
        self.value = value

    def __str__(self):
        row, col = self.square
        return f"({row},{col}) -> {self.value}"

    def __eq__(self, other):
        return (self.square, self.value) == (other.square, other.value)


class TabooMove(Move):
    """A TabooMove is a Move that was flagged as illegal by the sudoku oracle. In other words, the execution of such a
    move would cause the sudoku to become unsolvable.
    """

    """
    Constructs a taboo move.
    @param square: A square with coordinates in the range [0, ..., N)
    @param value: A value in the range [1, ..., N]
    """

    def __init__(self, square: Square, value: int):
        super().__init__(square, value)


class SudokuBoard(object):
    """
    A simple board class for Sudoku. It supports arbitrary rectangular regions.
    """

    empty = 0  # Empty squares contain the value SudokuBoard.empty

    def __init__(self, m: int = 3, n: int = 3):
        """
        Constructs an empty Sudoku with regions of size m x n.
        @param m: The number of rows in a region.
        @param n: The number of columns in a region.
        """
        N = m * n
        self.m = m
        self.n = n
        self.N = N  # N = m * n, numbers are in the range [1, ..., N]
        self.squares = [SudokuBoard.empty] * (N * N)  # The N*N squares of the board

    def square2index(self, square: Square) -> int:
        """
        Converts row/column coordinates to the corresponding index in the board array.
        @param square: A square with coordinates in the range [0, ..., N)
        @return: The corresponding index k in the board array
        """
        i, j = square
        N = self.N
        return N * i + j

    def index2square(self, k: int) -> Square:
        """
        Converts an index in the board array to the corresponding row/column coordinates.
        @param k: A value in the range [0, ..., N * N)
        @return: The corresponding row/column coordinates
        """
        N = self.N
        i = k // N
        j = k % N
        return i, j

    def put(self, square: Square, value: int) -> None:
        """
        Puts a value on a square.
        @param square: A square with coordinates in the range [0, ..., N)
        @param value: A value in the range [1, ..., N]
        """
        k = self.square2index(square)
        self.squares[k] = value

    def get(self, square: Square) -> int:
        """
        Gets the value of the given square.
        @param square: A square with coordinates in the range [0, ..., N)
        @return: The value of the square.
        """
        k = self.square2index(square)
        return self.squares[k]

    def region_width(self):
        """
        Gets the number of columns in a region.
        @return: The number of columns in a region.
        """
        return self.n

    def region_height(self):
        """
        Gets the number of rows in a region.
        @return: The number of rows in a region.
        """
        return self.m

    def board_width(self):
        """
        Gets the number of columns of the board.
        @return: The number of columns of the board.
        """
        return self.N

    def board_height(self):
        """
        Gets the number of rows of the board.
        @return: The number of rows of the board.
        """
        return self.N

    def __str__(self) -> str:
        """
        Prints the board in a simple textual format. The first line contains the values m and n. Then the contents of
        the rows are printed as space separated lists, where a dot '.' is used to represent an empty square.
        @return: The generated string.
        """
        return print_sudoku_board(self)


# written by Gennaro Gala
def pretty_print_sudoku_board(board: SudokuBoard, gamestate=None) -> str:
    import io

    m = board.m
    n = board.n
    N = board.N
    out = io.StringIO()

    def print_square(square: Square):
        value = board.get(square)
        s = " -" if value == 0 else f"{value:2}"

        if gamestate == None:
            return s + " "
        if square in gamestate.occupied_squares1:
            return s + "+"
        elif square in gamestate.occupied_squares2:
            return s + "-"
        else:
            return s + " "

    for i in range(N):

        # open the grid
        if i == 0:
            out.write("  ")
            for j in range(N):
                out.write(f"    {j}  ")
            out.write("\n")
            for j in range(N):
                if j % n != 0:
                    out.write("╤══════")
                elif j != 0:
                    out.write("╦══════")
                else:
                    out.write("   ╔══════")
            out.write("╗\n")

        # separate regions horizontally
        if i % m == 0 and i != 0:
            for j in range(N):
                if j % n != 0:
                    out.write("╪══════")
                elif j != 0:
                    out.write("╬══════")
                else:
                    out.write("   ╠══════")
            out.write("║\n")

        # plot values
        out.write(f"{i:2} ")
        for j in range(N):
            square = i, j
            symbol = print_square(square)
            if j % n != 0:
                out.write(f"│ {symbol}  ")
            else:
                out.write(f"║ {symbol}  ")
            if len(symbol) < 2:
                out.write(" ")
        out.write("║\n")

        # close the grid
        if i == N - 1:
            for j in range(N):
                if j % n != 0:
                    out.write("╧══════")
                elif j != 0:
                    out.write("╩══════")
                else:
                    out.write("   ╚══════")
            out.write("╝\n")

    return out.getvalue()


def print_sudoku_board(board: SudokuBoard) -> str:
    """
    Prints the board in a simple textual format. The first line contains the values m and n. Then the contents of
    the rows are printed as space separated lists, where a dot '.' is used to represent an empty square.
    @return: The generated string.
    """
    m = board.m
    n = board.n
    N = board.N
    out = io.StringIO()

    def print_square(square: Square):
        value = board.get(square)
        s = "   ." if value == 0 else f"{value:>4}"
        out.write(s)

    out.write(f"{m} {n}\n")
    for i in range(N):
        for j in range(N):
            square = i, j
            print_square(square)
        out.write("\n")
    return out.getvalue()


def parse_sudoku_board(text: str) -> SudokuBoard:
    """
    Loads a sudoku board from a string, in the same format as used by the SudokuBoard.__str__ function.
    @param text: A string representation of a sudoku board.
    @return: The generated Sudoku board.
    """
    words = text.split()
    if len(words) < 2:
        raise RuntimeError("The string does not contain a sudoku board")
    m = int(words[0])
    n = int(words[1])
    N = m * n
    if len(words) != N * N + 2:
        raise RuntimeError("The number of squares in the sudoku is incorrect.")
    result = SudokuBoard(m, n)
    N = result.N
    for k in range(N * N):
        s = words[k + 2]
        if s != ".":
            value = int(s)
            result.squares[k] = value
    return result


class GameState(object):
    def __init__(
        self,
        initial_board: SudokuBoard = None,
        board: SudokuBoard = None,
        taboo_moves: List[TabooMove] = None,
        moves: List[Union[Move, TabooMove]] = None,
        scores: List[int] = None,
        current_player: int = 1,
        allowed_squares1: Optional[List[Square]] = None,
        allowed_squares2: Optional[List[Square]] = None,
        occupied_squares1: Optional[List[Square]] = None,
        occupied_squares2: Optional[List[Square]] = None,
    ):
        """
        @param initial_board: A sudoku board. It contains the start position of a game.
        @param board: A sudoku board. It contains the current position of a game.
        @param taboo_moves: A list of taboo moves. Moves in this list cannot be played.
        @param moves: The history of a sudoku game, starting in initial_board. The history includes taboo moves.
        @param scores: The cumulative rewards of the first and the second player.
        @param current_player: The current player (1 or 2).
        @param allowed_squares1: The squares where player1 is always allowed to play (None if all squares are allowed).
        @param allowed_squares2: The squares where player2 is always allowed to play (None if all squares are allowed).
        @param occupied_squares1: The squares occupied by player1.
        @param occupied_squares2: The squares occupied by player2.
        """
        if taboo_moves is None:
            taboo_moves = []
        if moves is None:
            moves = []
        if scores is None:
            scores = [0, 0]
        if initial_board is None and board is None:
            initial_board = SudokuBoard(2, 2)
            board = SudokuBoard(2, 2)
        elif board is None:
            board = copy.deepcopy(initial_board)
            for move in moves:
                board.put(move.square, move.value)
        elif initial_board is None:
            initial_board = copy.deepcopy(board)
            for move in moves:
                initial_board.put(move.square, SudokuBoard.empty)
        self.initial_board = initial_board
        self.board = board
        self.taboo_moves = taboo_moves
        self.moves = moves
        self.scores = scores
        self.current_player = current_player
        self.allowed_squares1 = allowed_squares1
        self.allowed_squares2 = allowed_squares2
        self.occupied_squares1 = occupied_squares1
        self.occupied_squares2 = occupied_squares2

    def simulate_move(self, move):
        new_state = copy.deepcopy(self)
        new_state.board.put(move.square, move.value)

        completed_regions = self.amount_of_regions_completed(new_state, move)
        if completed_regions > 0:
            new_state.scores[new_state.current_player - 1] += self.get_region_score(
                completed_regions
            )
        return new_state

    def is_classic_game(self):
        """
        Returns True if the game is classic, i.e. all squares are allowed.
        """
        return self.allowed_squares1 is None and self.allowed_squares2 is None

    def occupied_squares(self):
        """
        Returns the occupied squares of the current player.
        """
        return (
            self.occupied_squares1
            if self.current_player == 1
            else self.occupied_squares2
        )

    def player_squares(self) -> Optional[List[Square]]:
        """
        Returns the squares where the current player can play, or None if all squares are allowed.
        """
        allowed_squares = (
            self.allowed_squares1 if self.current_player == 1 else self.allowed_squares2
        )
        occupied_squares = (
            self.occupied_squares1
            if self.current_player == 1
            else self.occupied_squares2
        )
        N = self.board.N

        if allowed_squares is None:
            return None

        def is_empty(square: Square) -> bool:
            return self.board.get(square) == SudokuBoard.empty

        def neighbors(square: Square) -> Iterator[Square]:
            row, col = square
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    if 0 <= r < N and 0 <= c < N:
                        yield r, c

        # add the empty allowed squares
        result = [s for s in allowed_squares if is_empty(s)]

        # add the empty neighbors to result
        for s1 in occupied_squares:
            for s2 in neighbors(s1):
                if is_empty(s2):
                    result.append(s2)

        # remove duplicates
        return sorted(list(set(result)))

    def __str__(self):
        return print_game_state(self)


def parse_properties(text: str) -> dict[str, str]:
    """
    Parses a string containing key-value pairs.
    Lines should have the format `key = value`. A value can be a multiline string. In that
    case subsequent lines should start with a space. Lines starting with '#' are ignored.
    @param text: A string.
    @return: A dictionary of key-value pairs.
    """
    result = {}
    key = None
    value = []

    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("#") or not line.strip():
            continue
        elif line.startswith(" "):
            value.append(line.lstrip())
        else:
            if key:
                result[key] = "\n".join(value).strip()
            words = line.split("=", 1)
            if len(words) not in [1, 2]:
                raise ValueError(f"Unexpected line '{line}'")
            key = words[0].strip()
            value = [words[1].strip()] if len(words) > 1 else []

    if key:
        result[key] = "\n".join(value).strip()

    return result


def print_game_state(game_state: GameState) -> str:
    """
    Saves a game state as a string containing key-value pairs.
    @param game_state: A game state.
    """
    out = io.StringIO()

    is_classic_game = game_state.is_classic_game()

    board = game_state.board
    m = board.m
    n = board.n
    N = board.N

    def print_square(square: Square):
        value = board.get(square)
        if is_classic_game:
            s = "   ." if value == 0 else f"{value:>4}"
        else:
            occupied = "+" if square in game_state.occupied_squares1 else "-"
            s = "     ." if value == 0 else f" {value:>4}{occupied}"
        out.write(s)

    out.write(f"rows = {m}\n")
    out.write(f"columns = {n}\n")
    out.write(f"board =\n")
    for i in range(N):
        for j in range(N):
            square = i, j
            print_square(square)
        out.write("\n")
    taboo_moves = [f"{move}" for move in game_state.taboo_moves]
    out.write(f'taboo-moves = [{", ".join(taboo_moves)}]\n')
    moves = [f"{move}" for move in game_state.moves]
    out.write(f'moves = [{", ".join(moves)}]\n')
    out.write(f"scores = {game_state.scores}\n")
    out.write(f"current-player = {game_state.current_player}\n")
    if not game_state.is_classic_game():
        allowed_squares1 = [
            f"({square[0]},{square[1]})" for square in game_state.allowed_squares1
        ]
        out.write(f'allowed-squares1 = {", ".join(allowed_squares1)}\n')
        allowed_squares2 = [
            f"({square[0]},{square[1]})" for square in game_state.allowed_squares2
        ]
        out.write(f'allowed-squares2 = {", ".join(allowed_squares2)}\n')
        occupied_squares1 = [
            f"({square[0]},{square[1]})" for square in game_state.occupied_squares1
        ]
        out.write(f'occupied-squares1 = {", ".join(occupied_squares1)}\n')
        occupied_squares2 = [
            f"({square[0]},{square[1]})" for square in game_state.occupied_squares2
        ]
        out.write(f'occupied-squares2 = {", ".join(occupied_squares2)}\n')
    return out.getvalue()


def pretty_print_game_state(game_state: GameState) -> str:
    out = io.StringIO()
    out.write(pretty_print_sudoku_board(game_state.board, game_state))
    out.write(f"Score: {game_state.scores[0]} - {game_state.scores[1]}\n")
    out.write(f"Current player: player{game_state.current_player}\n")
    if not game_state.is_classic_game():
        out.write(
            f'Player1 allowed squares: {"None (all squares are allowed)" if game_state.allowed_squares1 is None else game_state.allowed_squares1}\n'
        )
        out.write(
            f'Player2 allowed squares: {"None (all squares are allowed)" if game_state.allowed_squares2 is None else game_state.allowed_squares2}\n'
        )
        out.write(
            f"Player1 occupied squares: {list(sorted(game_state.occupied_squares1))}\n"
        )
        out.write(
            f"Player2 occupied squares: {list(sorted(game_state.occupied_squares2))}\n"
        )
    return out.getvalue()


def generate_random_tuples(N):
    """
    Generates 2N random and distinct tuples of (i, j) where 0 <= i, j < N.

    Args:
        N: A positive integer.

    Returns:
        A list of 2N distinct tuples of (i, j) where 0 <= i, j < N.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer")

    unique_tuples = set()

    # Fill the set with random tuples until we have 2N elements
    while len(unique_tuples) < 2 * N:
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        unique_tuples.add((i, j))

    # Convert the set of tuples to a list
    return list(unique_tuples)


def allowed_squares(
    board: SudokuBoard, playmode: str
) -> Tuple[List[Square], List[Square]]:
    """
    Generates allowed squares for player1 and player2.
    @param board: A SudokuBoard object.
    @param playmode: The playing playmode (classic, rows, random)
    """
    N = board.N
    if playmode == "classic":
        return [], []
    elif playmode == "rows":
        return [(0, j) for j in range(N)], [(N - 1, j) for j in range(N)]
    elif playmode == "border":
        top = [(0, j) for j in range(N)]
        bottom = [(N - 1, j) for j in range(N)]
        right = [(i, 0) for i in range(1, N - 1)]
        left = [(i, N - 1) for i in range(1, N - 1)]
        border = top + bottom + right + left
        return border, border
    elif playmode == "random":
        squares = generate_random_tuples(N)
        return squares[:N], squares[N:]


def parse_game_state(text: str, playmode: str) -> GameState:
    """
    Loads a game state from a string containing key-value pairs.
    @param text: A string representation of a game state.
    """
    properties = parse_properties(text)
    is_classic_game = playmode == "classic"

    def remove_special_characters(text):
        for char in "[](),->":
            text = text.replace(char, " ")
        return text

    def parse_board(
        key: str, m: int, n: int
    ) -> Tuple[Optional[SudokuBoard], Optional[List[Square]], Optional[List[Square]]]:
        text = properties.get(key)
        if text is None:
            return None, None, None
        if is_classic_game:
            return parse_sudoku_board(f"{m} {n}\n{text}"), None, None
        occupied_squares1 = []
        occupied_squares2 = []
        N = m * n
        words = text.strip().split()
        if len(words) != N * N:
            raise ValueError("The number of squares in the sudoku board is incorrect.")
        board = SudokuBoard(m, n)

        for k, word in enumerate(words):
            if word != ".":
                value, occupied = word[:-1], word[-1]
                value = int(value)
                board.squares[k] = value
                if occupied == "+":
                    occupied_squares1.append(board.index2square(k))
                else:
                    occupied_squares2.append(board.index2square(k))

        return board, occupied_squares1, occupied_squares2

    def parse_moves(key: str, move_class) -> Union[List[Move], List[TabooMove]]:
        text = properties.get(key)
        if text is None:
            return []
        result = []
        items = remove_special_characters(text).strip().split()
        items = [int(item) for item in items]
        assert (
            len(items) % 3 == 0
        ), "The number of elements in the a move list must be divisible by 3."
        for index in range(0, len(items), 3):
            i, j, value = items[index], items[index + 1], items[index + 2]
            result.append(move_class((i, j), value))
        return result

    def parse_scores(key: str) -> Optional[List[int]]:
        text = properties.get(key)
        if text is None:
            return None
        items = remove_special_characters(text).strip().split()
        items = [int(item) for item in items]
        assert len(items) == 2, "The number of elements in the scores list must be 2."
        return items

    def parse_squares(key: str) -> Optional[List[Square]]:
        text = properties.get(key)
        if text is None:
            return None
        result = []
        items = remove_special_characters(text).strip().split()
        assert (
            len(items) % 2 == 0
        ), "The number of elements in the a square list must be divisible by 2."
        items = [int(item) for item in items]
        for index in range(0, len(items), 2):
            i, j = items[index], items[index + 1]
            result.append((i, j))
        return result

    m = int(properties["rows"])
    n = int(properties["columns"])

    moves = parse_moves("moves", Move)
    taboo_moves = parse_moves("taboo-moves", TabooMove)
    scores = parse_scores("scores")
    current_player = int(properties.get("current-player", "1"))

    if is_classic_game:
        initial_board = None
        board = parse_sudoku_board(f"{m} {n}\n" + properties["board"])
        occupied_squares1 = None
        occupied_squares2 = None
        allowed_squares1 = None
        allowed_squares2 = None
    else:
        initial_board, _, _ = parse_board("initial-board", m, n)
        board, occupied_squares1, occupied_squares2 = parse_board("board", m, n)
        allowed_squares1 = parse_squares("allowed-squares1")
        allowed_squares2 = parse_squares("allowed-squares2")
        if allowed_squares1 is None or allowed_squares2 is None:
            allowed_squares1, allowed_squares2 = allowed_squares(board, playmode)

    return GameState(
        initial_board=initial_board,
        board=board,
        taboo_moves=taboo_moves,
        moves=moves,
        scores=scores,
        current_player=current_player,
        allowed_squares1=allowed_squares1,
        allowed_squares2=allowed_squares2,
        occupied_squares1=occupied_squares1,
        occupied_squares2=occupied_squares2,
    )


import random
import copy
from competitive_sudoku.sudoku import (
    GameState,
    Move,
    TabooMove,
    SudokuBoard,
    pretty_print_sudoku_board,
    pretty_print_game_state,
)
import competitive_sudoku.sudokuai
from typing import Tuple


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration using Minimax.
    """

    def __init__(self):
        super().__init__()

    def amount_of_regions_completed(self, game_state: GameState, move: Move):
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

    def simulate_move(self, game_state: GameState, move: Move):
        """
        Simulates a move and returns a new game state.
        """
        score_dict = {0: 0, 1: 1, 2: 3, 3: 7}

        new_state = copy.deepcopy(game_state)
        new_state.board.put(move.square, move.value)
        new_state.moves.append(move)

        completed_regions = self.amount_of_regions_completed(new_state, move)

        new_state.scores[new_state.current_player - 1] += score_dict[completed_regions]

        new_state.current_player = 3 - new_state.current_player
        return new_state

    def get_valid_moves(self, game_state: GameState):
        """
        Gets the valid moves for the current Game State
        """
        N = game_state.board.N

        def is_valid_move(square, value):
            return (
                game_state.board.get(square) == SudokuBoard.empty
                and not TabooMove(square, value) in game_state.taboo_moves
                and (
                    square in game_state.player_squares()
                    if game_state.player_squares() is not None
                    else True
                )
                and value
                not in [game_state.board.get((square[0], col)) for col in range(N)]
                and value
                not in [game_state.board.get((row, square[1])) for row in range(N)]
                and value not in self.get_region_values(game_state.board, square)
            )

        valid_moves = [
            Move((i, j), value)
            for i in range(N)
            for j in range(N)
            for value in range(1, N + 1)
            if is_valid_move((i, j), value)
        ]
        return valid_moves

    def get_region_values(self, board: SudokuBoard, square: Tuple[int, int]):
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

    def is_terminal(self, game_state: GameState):
        """
        Checks if the game state is terminal (no valid moves left).
        """
        return len(self.get_valid_moves(game_state)) == 0

    def evaluate(self, game_state: GameState, ai_player_index: int):
        """
        Evaluates the game state with a heuristic based on the score and potential moves.
        """
        w1 = 0.9
        w2 = 0.1

        if ai_player_index == 0:
            return w1 * (game_state.scores[0] - game_state.scores[1]) + (
                w2
                * (len(game_state.allowed_squares1) - len(game_state.allowed_squares2))
            )

        if ai_player_index == 1:
            return w1 * (game_state.scores[1] - game_state.scores[0]) + (
                w2
                * (len(game_state.allowed_squares2) - len(game_state.allowed_squares1))
            )

    def minimax(
        self,
        game_state: GameState,
        depth: int,
        alpha: int,
        beta: int,
        maximizing: bool,
        ai_player_index: int,
    ):
        """
        Minimax implementation with depth-limited search.
        """
        if depth == 0 or self.is_terminal(game_state):
            return self.evaluate(game_state, ai_player_index)

        valid_moves = self.get_valid_moves(game_state)

        if maximizing:
            max_eval = float("-inf")
            for move in valid_moves:

                next_state = self.simulate_move(game_state, move)
                eval = self.minimax(
                    next_state,
                    depth - 1,
                    alpha,
                    beta,
                    maximizing=False,
                    ai_player_index=ai_player_index,
                )
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in valid_moves:
                next_state = self.simulate_move(game_state, move)
                eval = self.minimax(
                    next_state,
                    depth - 1,
                    alpha,
                    beta,
                    maximizing=True,
                    ai_player_index=ai_player_index,
                )
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Minimax with iterative deepening depth  # ToDo - Caching if needed
        """
        ai_player_index = game_state.current_player - 1
        depth = (game_state.board.board_height() * game_state.board.board_width()) - (
            len(game_state.occupied_squares1) + len(game_state.occupied_squares2)
        )  # Total number of unoccupied squares

        valid_moves = self.get_valid_moves(game_state)

        best_move = None
        best_score = float("-inf")

        for depth in range(1, depth + 1):
            depth_move_scores = []

            for move in valid_moves:
                next_state = self.simulate_move(game_state, move)
                score = self.minimax(
                    next_state,
                    depth,
                    float("-inf"),
                    float("inf"),
                    maximizing=False,
                    ai_player_index=ai_player_index,
                )
                depth_move_scores.append((move, score))

                if score >= best_score:
                    best_score = score
                    best_move = move
                if best_move:
                    self.propose_move(best_move)

            valid_moves = [
                i[0]
                for i in sorted(depth_move_scores, key=lambda x: x[1], reverse=True)
            ]


class AdvancedSudokuAI(SudokuAI):
    """Advanced Sudoku AI with enhanced heuristics for competitive play."""

    def __init__(self):
        super().__init__()

    def evaluate(self, game_state: GameState, ai_player_index: int):
        """Enhanced evaluation function with additional heuristics."""
        # Weights for the evaluation components
        w1, w2, w3 = 0.7, 0.2, 0.1

        # Score difference as primary heuristic
        score_diff = (
            game_state.scores[ai_player_index] - game_state.scores[1 - ai_player_index]
        )

        # Remaining valid moves for opponent vs AI (blocking advantage)
        opp_valid_moves = len(self.get_valid_moves_for_player(game_state, 3 - game_state.current_player))
        ai_valid_moves = len(self.get_valid_moves_for_player(game_state, game_state.current_player))
        move_diff = ai_valid_moves - opp_valid_moves

        # Centrality heuristic: prefer moves closer to the center of the board
        N = game_state.board.N
        center = (N // 2, N // 2)
        centrality_score = -sum(
            abs(move.square[0] - center[0]) + abs(move.square[1] - center[1])
            for move in self.get_valid_moves(game_state)
        )

        return w1 * score_diff + w2 * move_diff + w3 * centrality_score

    def get_valid_moves_for_player(self, game_state: GameState, player: int):
        """Get valid moves for a specific player."""
        current_player = game_state.current_player
        game_state.current_player = player
        moves = self.get_valid_moves(game_state)
        game_state.current_player = current_player  # Restore original player
        return moves

    def get_valid_moves(self, game_state: GameState):
        """Enhanced valid moves to prioritize better moves."""
        N = game_state.board.N

        def is_valid_move(square, value):
            return (
                game_state.board.get(square) == SudokuBoard.empty
                and not TabooMove(square, value) in game_state.taboo_moves
                and (
                    square in game_state.player_squares()
                    if game_state.player_squares() is not None
                    else True
                )
                and value
                not in [game_state.board.get((square[0], col)) for col in range(N)]
                and value
                not in [game_state.board.get((row, square[1])) for row in range(N)]
                and value not in self.get_region_values(game_state.board, square)
            )

        valid_moves = [
            Move((i, j), value)
            for i in range(N)
            for j in range(N)
            for value in range(1, N + 1)
            if is_valid_move((i, j), value)
        ]

        # Rank moves by their impact on scoring
        ranked_moves = sorted(
            valid_moves,
            key=lambda move: self.amount_of_regions_completed(game_state, move),
            reverse=True,
        )
        return ranked_moves

    def compute_best_move(self, game_state: GameState) -> None:
        """Enhanced best move computation with dynamic depth."""
        ai_player_index = game_state.current_player - 1
        remaining_moves = (
            game_state.board.board_height() * game_state.board.board_width()
            - len(game_state.occupied_squares1)
            - len(game_state.occupied_squares2)
        )

        depth = min(5, remaining_moves // 10)  # Dynamically adjust depth

        valid_moves = self.get_valid_moves(game_state)
        best_move = None
        best_score = float("-inf")

        for move in valid_moves:
            next_state = self.simulate_move(game_state, move)
            score = self.minimax(
                next_state,
                depth,
                float("-inf"),
                float("inf"),
                maximizing=False,
                ai_player_index=ai_player_index,
            )
            if score > best_score:
                best_score = score
                best_move = move

        if best_move:
            self.propose_move(best_move)
