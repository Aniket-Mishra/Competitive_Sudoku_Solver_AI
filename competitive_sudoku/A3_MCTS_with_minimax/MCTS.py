import math
import random
import copy
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard

from A3_MCTS_with_minimax.helper_functions import (
    get_valid_moves,
    simulate_move,
)
from A3_MCTS_with_minimax.minimax import (
    minimax,
)

from A3_MCTS_with_minimax.taboo_helpers import naked_singles, hidden_singles


def score_mobility(game_state: GameState, ai_player_index: int) -> float:
    """Compute how many moves the AI can make minus how many the opponent can make."""
    original_player = game_state.current_player
    game_state.current_player = ai_player_index + 1
    ai_moves = get_valid_moves(game_state)
    ai_moves = naked_singles(game_state, ai_moves)
    ai_count = sum(len(vals) for vals in ai_moves.values())

    # Opponent
    opp_index = 1 - ai_player_index
    game_state.current_player = opp_index + 1
    opp_moves = get_valid_moves(game_state)
    opp_moves = naked_singles(game_state, opp_moves)
    opp_count = sum(len(vals) for vals in opp_moves.values())

    game_state.current_player = original_player

    return ai_count - opp_count


def score_near_completion(
    game_state: GameState, ai_player_index: int
) -> float:
    """Reward rows/columns/blocks that are nearly complete and belong to the AI."""
    board = game_state.board
    N = board.N

    original_player = game_state.current_player
    game_state.current_player = ai_player_index + 1
    ai_squares = (
        game_state.occupied_squares1
        if ai_player_index == 0
        else game_state.occupied_squares2
    )
    game_state.current_player = original_player

    row_bonus = 0.0
    for row in range(N):
        row_count = sum(1 for col in range(N) if (row, col) in ai_squares)
        if row_count >= (N - 1):
            row_bonus += 2.0

    return row_bonus


def score_difference(game_state: GameState, ai_player_index: int) -> float:
    return (
        game_state.scores[ai_player_index]
        - game_state.scores[1 - ai_player_index]
    )


def evaluate_domain_heuristics(
    game_state: GameState, ai_player_index: int
) -> float:
    w_mobility = 0.3
    w_near_comp = 0.4
    w_diff = 0.3

    mob = score_mobility(game_state, ai_player_index)
    near_comp = score_near_completion(game_state, ai_player_index)
    diff = score_difference(game_state, ai_player_index)

    return w_mobility * mob + w_near_comp * near_comp + w_diff * diff


def HYBRID_DEPTH_THRESHOLD() -> int:
    return 2


def call_minimax_for_evaluation(
    game_state: GameState, ai_player_index: int
) -> float:
    valid_moves_dict = get_valid_moves(game_state)
    valid_moves_dict = naked_singles(game_state, valid_moves_dict)
    valid_moves = []
    for (row, col), vals in valid_moves_dict.items():
        for v in vals:
            valid_moves.append(Move((row, col), v))

    if not valid_moves:
        return float("-inf")

    search_depth = 2
    best_score = float("-inf")
    for move in valid_moves:
        next_state = simulate_move(game_state, move)
        score = minimax(
            next_state,
            search_depth,
            alpha=float("-inf"),
            beta=float("inf"),
            maximizing=False,
            ai_player_index=ai_player_index,
        )
        if score > best_score:
            best_score = score
    return best_score


def rollout_simulation(
    game_state: GameState, ai_player_index: int, depth: int = 0, max_depth=10
) -> float:
    if depth > HYBRID_DEPTH_THRESHOLD():
        mm_val = call_minimax_for_evaluation(game_state, ai_player_index)
        heuristic_val = evaluate_domain_heuristics(game_state, ai_player_index)
        return mm_val + 0.1 * heuristic_val

    if depth >= max_depth:
        return evaluate_domain_heuristics(game_state, ai_player_index)

    moves_dict = get_valid_moves(game_state)
    moves_dict = naked_singles(game_state, moves_dict)
    valid_moves = [
        Move((row, col), v)
        for (row, col), vals in moves_dict.items()
        for v in vals
    ]
    if not valid_moves:
        return evaluate_domain_heuristics(game_state, ai_player_index)

    random_move = random.choice(valid_moves)
    next_state = simulate_move(game_state, random_move)
    return rollout_simulation(
        next_state, ai_player_index, depth + 1, max_depth
    )


class MCTSNode:
    def __init__(self, game_state: GameState, move: Move | None):
        self.game_state = game_state
        self.move = move
        self.parent = None
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.UCT_Score = float("inf")

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)


def ucb_score(node: "MCTSNode", col=1.4) -> float:
    if (
        node.visit_count == 0
        or node.parent is None
        or node.parent.visit_count == 0
    ):
        return float("inf")

    exploit = node.value_sum / node.visit_count
    n_parent = node.parent.visit_count
    explore = col * math.sqrt(math.log(n_parent) / node.visit_count)
    return exploit + explore

def uct_score(node: "MCTSNode", exploration_constant=1.4) -> float:
    """
    Computes the UCT (Upper Confidence Bound for Trees) score for a node.
    
    Args:
        node (MCTSNode): The current node.
        exploration_constant (float): The exploration weight (default is 1.4).
    
    Returns:
        float: The UCT score.
    """
    if (
        node.visit_count == 0
        or node.parent is None
        or node.parent.visit_count == 0
    ):
        return float("inf")  # Encourage exploration for unvisited nodes.

    # Exploitation term
    exploitation_value = node.value_sum / node.visit_count

    # Exploration term
    parent_visits = node.parent.visit_count
    exploration_value = exploration_constant * math.sqrt(math.log(parent_visits) / node.visit_count)

    # Combine exploitation and exploration
    return exploitation_value + exploration_value



def state_key(game_state: GameState) -> tuple:
    """
    Create a hashable key for transposition table lookup.
    (board squares, current_player, scores).
    """
    return (
        tuple(game_state.board.squares),
        game_state.current_player,
        tuple(game_state.scores),
    )


class MonteCarloTree:
    def __init__(self, game_state: GameState, ai_player_index: int):
        self.ai_player_index = ai_player_index
        self.transposition_table = {}
        self.root = self._get_or_create_node(game_state, move=None)

    def _get_or_create_node(self, game_state: GameState, move: Move | None):
        key = state_key(game_state)
        if key in self.transposition_table:
            node = self.transposition_table[key]
            return node
        else:
            node = MCTSNode(game_state, move)
            self.transposition_table[key] = node
            return node

    def visit(self) -> MCTSNode:
        current_node = self.root
        while current_node.children:
            current_node = max(
                current_node.children, key=lambda col: col.UCT_Score
            )
        return current_node

    def expand(self, node: MCTSNode, max_new_children=5) -> MCTSNode:
        moves_dict = get_valid_moves(node.game_state)
        moves_dict = naked_singles(node.game_state, moves_dict)
        valid_moves = [
            Move((row, col), value)
            for (row, col), vals in moves_dict.items()
            for value in vals
        ]

        expanded_moves = {
            (child.move.square, child.move.value)
            for child in node.children
            if child.move
        }
        unexpanded = [
            move
            for move in valid_moves
            if (move.square, move.value) not in expanded_moves
        ]
        if not unexpanded:
            return node

        random.shuffle(unexpanded)
        new_moves = unexpanded[:max_new_children]
        for move in new_moves:
            nxt_state = simulate_move(node.game_state, move)
            child_node = self._get_or_create_node(nxt_state, move)
            if child_node.parent is None:
                child_node.parent = node
                node.children.append(child_node)

        if node.children:
            return random.choice(node.children)
        else:
            return node

    def simulate_playout(self, node: MCTSNode) -> float:
        return rollout_simulation(
            node.game_state, self.ai_player_index, depth=0, max_depth=10
        )

    def backpropagate(self, node: MCTSNode, result: float):
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += result
            current.UCT_Score = uct_score(current)
            current = current.parent

    def do_iteration(self):
        leaf = self.visit()
        expanded = self.expand(leaf, max_new_children=3)
        result = self.simulate_playout(expanded)
        self.backpropagate(expanded, result)

    def search(self, iterations=500, time_limit=2.0):
        start_t = time.time()
        for _ in range(iterations):
            if (time.time() - start_t) > time_limit:
                break
            self.do_iteration()

    def best_move(self) -> Move | None:
        if not self.root.children:
            return None
        best_child = max(self.root.children, key=lambda x: x.visit_count)
        return best_child.move
