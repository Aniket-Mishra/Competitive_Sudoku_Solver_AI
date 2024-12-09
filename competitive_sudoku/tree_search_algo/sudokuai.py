#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

from competitive_sudoku.sudoku import GameState
import competitive_sudoku.sudokuai
from team21_A2.tree_search import Tree, get_heuristic_moves
from team21_A2.evaluation import evaluate_move


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Sends the move that our agent should play whenever we find a good move
        @param game_state: The current GameState of the Competitive Sudoku game
        """

        # Get all legal moves, split by whether we know that the Oracle will call them taboo (but not disqualify us)
        legal_moves, legal_taboo_move = get_heuristic_moves(game_state.board, game_state.taboo_moves)

        # Check what player we are by the length of the GameStates' list of moves,
        # and calculate how well we are doing based on that and the current score of both players
        what_player_are_we = len(game_state.moves) % 2
        if what_player_are_we == 1:
            initial_scores = game_state.scores[1]-game_state.scores[0]
        else:
            initial_scores = game_state.scores[0]-game_state.scores[1]

        # oh god just submit any move we know of so we won't get disqualified for not having a move
        best_move = legal_moves[0]
        self.propose_move(best_move)

        # Now at least check which move gives us the most points (we are now at the level of the greedy_player agent)
        best_score = -999
        for cur_move in legal_moves[0:]:
            score = evaluate_move(game_state.board, cur_move, True, initial_scores)
            if score > best_score:
                best_move = cur_move
        self.propose_move(best_move)

        # Now we create a Tree object (which will be used for minimax search trees),
        # and we make it depth 1 (also considering what possibilities our move gives to the opposing player)
        search_tree = Tree(legal_moves, game_state.taboo_moves, legal_taboo_move, game_state.board, initial_scores)

        # We search up to 9 moves ahead, because more is simply impossible in practice (would take WAY too much time)
        depth = 1
        while depth < 9:
            # We keep extending the search depth and finding the best move using our minimax algorithm until depth 9
            best_move = search_tree.deepen_search()
            print("Best move at tree depth " + str(depth) + " found, attempting to increase depth")

            # After we find the best move for the current depth, we give it to the program so it is saved
            # & we increment the depth variable to keep track of our current depth
            self.propose_move(best_move)
            depth += 1


