import random
import copy
from competitive_sudoku.sudoku import (
    GameState,
    Move,
    TabooMove,
    SudokuBoard,
)
import competitive_sudoku.sudokuai
from typing import Tuple
from A3_MCTS.helper_functions import get_valid_moves, simulate_move, naked_singles
import math


class MCTSNode:
    def __init__(self, game_state: GameState, move: Move | None):
        self.game_state = game_state
        self.move = move
        self.player_1_wins = 0
        self.player_2_wins = 0
        self.children = []
        self.parent = None
        self.UCT_Score = float('inf')
        self.visit_count = 0

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)


class MonteCarloTree:
    def __init__(self, game_state: GameState, ):
        self.root = MCTSNode(game_state, None)

    def visit(self):
        current_node = self.root

        while current_node.children:
            best_score = float('-inf')
            best_node = None

            for child in random.sample(current_node.children, len(current_node.children)):
                if best_score < child.UCT_Score:
                    best_score = child.UCT_Score
                    best_node = child
            current_node = best_node

        return current_node

    def expand(self, selected_node: MCTSNode):
        moves = get_valid_moves(selected_node.game_state)
        moves = naked_singles(selected_node.game_state, moves)
        valid_moves = [
            Move((row, col), value)
            for (row, col), values in moves.items()
            for value in values
        ]
        

        if not selected_node.children:
            for move in valid_moves:
                child_state = simulate_move(selected_node.game_state, move)
                child = MCTSNode(child_state, move)
                selected_node.add_child(child)


        return random.choice(selected_node.children) if moves else selected_node

    def simulate_playout(self, selected_node: MCTSNode):
        current_game_state = copy.deepcopy(selected_node.game_state)
        player1_out_of_moves = False
        player2_out_of_moves = False

        while True:
            if player1_out_of_moves and player2_out_of_moves:
                score_difference = current_game_state.scores[0] - \
                    current_game_state.scores[1]
                return 1 if score_difference > 0 else 2 if score_difference < 0 else 0

            if (current_game_state.current_player == 1 and player1_out_of_moves) or (current_game_state.current_player == 2 and player2_out_of_moves):
                current_game_state.current_player = 3 - current_game_state.current_player

            moves = get_valid_moves(current_game_state)
            moves = naked_singles(selected_node.game_state, moves)
            valid_moves = [
                Move((row, col), value)
                for (row, col), values in moves.items()
                for value in values
            ]

            if len(valid_moves) == 0:
                if current_game_state.current_player == 1:
                    player1_out_of_moves = True
                else:
                    player2_out_of_moves = True
                continue

            current_game_state = simulate_move(
                current_game_state, random.choice(valid_moves))

    def backpropagate(self, simulated_node: MCTSNode, winner: int, player: int):
        current_node = simulated_node

        while current_node is not None:
            current_node.visit_count += 1
            if winner == 1:
                current_node.player_1_wins += 1
            elif winner == 2:
                current_node.player_2_wins += 1
            elif winner == 0:
                current_node.player_1_wins += 0.5
                current_node.player_2_wins += 0.5
            else:
                pass

            current_node.UCT_Score = calculate_score(current_node, player)
            current_node = current_node.parent


def calculate_score(node: MCTSNode, player: int):
    exploration_weight = 1.5

    if node.parent is None:
        return 0

    modifier = -1 if node.game_state.current_player == 0 else 1
    nwins = modifier * node.player_1_wins if player == 0 else modifier * node.player_2_wins

    nsim = node.visit_count

    if nsim == 0:
        return float('inf')

    ln_n = math.log(node.parent.visit_count)
    return (nwins/nsim) + exploration_weight * math.sqrt(ln_n / nsim)
