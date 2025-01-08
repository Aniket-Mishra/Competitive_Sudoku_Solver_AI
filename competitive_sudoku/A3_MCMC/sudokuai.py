import random
from competitive_sudoku.sudoku import (
    GameState,
    Move,
)
import competitive_sudoku.sudokuai
from A3_MCMC.helper_functions import get_valid_moves
from A3_MCMC.MCMC import mcmc_search


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration using Minimax.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        ai_index = game_state.current_player - 1

        moves_dict = get_valid_moves(game_state)
        all_moves = []
        for (r, c), vals in moves_dict.items():
            for v in vals:
                all_moves.append(Move((r, c), v))
        if not all_moves:
            return

        self.propose_move(random.choice(all_moves))

        best_move = mcmc_search(
            root_state=game_state,
            ai_player_index=ai_index,
            iterations=1000,  # can increase if jelle gives candy
            temperature=2.0,
            time_limit=1.0,
        )

        if best_move is not None:
            self.propose_move(best_move)
        else:
            self.propose_move(random.choice(all_moves))


# python .\simulate_game.py --first=A3_MCTS --second=greedy_player --board=boards/empty-3x3.txt
