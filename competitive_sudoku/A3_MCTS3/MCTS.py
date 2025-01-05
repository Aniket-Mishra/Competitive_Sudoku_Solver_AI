import math
import random
import copy
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard

# Adjust imports to your environment
from A3_MCTS.helper_functions import (
    get_valid_moves,
    simulate_move,
    naked_singles,
)
from team03_A2.minimax import (
    minimax,
)  # or wherever your old minimax is located

###############################################################################
#                          HELPER FUNCTIONS
###############################################################################
from typing import Dict, List, Any, Optional, Tuple


def get_valid_moves(game_state: GameState) -> Dict[Tuple[int, int], List[int]]:
    """
    Returns a dictionary of {(row, col): [valid_values, ...], ...}.
    This is a simplified placeholder. You should replace with your real logic.
    """
    valid_moves = {}
    N = game_state.board.N
    for row in range(N):
        for col in range(N):
            idx = row * N + col
            if game_state.board.squares[idx] == 0:  # empty
                # Example: allow any digit from 1..N (not checking Sudoku constraints here!)
                # Replace with real Sudoku validation
                valid_moves[(row, col)] = list(range(1, N + 1))
    return valid_moves


def simulate_move(game_state, move: Move):
    """
    Applies 'move' to 'game_state' and returns a NEW, *deep-copied* GameState.
    We do not use game_state.clone(), because sudoku.py doesn't provide it.
    """
    next_state = copy.deepcopy(game_state)

    row, col = move.square
    val = move.value

    # Place the value on the board
    next_state.board.put((row, col), val)

    # Update occupant squares (lists, not sets!)
    if next_state.current_player == 1:
        if (row, col) not in next_state.occupied_squares1:
            next_state.occupied_squares1.append((row, col))
    else:
        if (row, col) not in next_state.occupied_squares2:
            next_state.occupied_squares2.append((row, col))

    # (Optional) Update scores if your game logic says so.
    next_state.scores[next_state.current_player - 1] += 1

    # Switch player
    next_state.current_player = 2 if next_state.current_player == 1 else 1

    return next_state


def naked_singles(
    game_state: GameState, moves_dict: Dict[Tuple[int, int], List[int]]
) -> Dict[Tuple[int, int], List[int]]:
    """
    Example placeholder for applying Sudoku constraints to reduce possible moves.
    Replace with your actual 'naked_singles' logic or more advanced constraints.
    """
    # For demonstration, we do nothing and just return moves_dict as is.
    return moves_dict


###############################################################################
#                          MINIMAX (Simplified)
###############################################################################


def is_terminal_state(game_state: GameState) -> bool:
    """
    Checks if there are no valid moves or if the board is full.
    """
    moves = get_valid_moves(game_state)
    if not moves:
        return True
    # or check if board is fully occupied:
    if all(v != 0 for v in game_state.board.squares):
        return True
    return False


def evaluate_state(game_state: GameState, ai_player_index: int) -> float:
    """
    Simple evaluation for demonstration. Adjust as needed.
    """
    # e.g., difference in scores
    return (
        game_state.scores[ai_player_index]
        - game_state.scores[1 - ai_player_index]
    )


###############################################################################
#                        DOMAIN-SPECIFIC HEURISTICS
###############################################################################


def score_mobility(game_state: GameState, ai_player_index: int) -> float:
    """Number of moves AI can make minus number of moves the opponent can make."""
    original_player = game_state.current_player

    # AI
    game_state.current_player = ai_player_index + 1
    ai_moves = get_valid_moves(game_state)
    ai_count = sum(len(vals) for vals in ai_moves.values())

    # Opponent
    opp_index = 1 - ai_player_index
    game_state.current_player = opp_index + 1
    opp_moves = get_valid_moves(game_state)
    opp_count = sum(len(vals) for vals in opp_moves.values())

    # restore
    game_state.current_player = original_player

    return ai_count - opp_count


def score_near_completion(
    game_state: GameState, ai_player_index: int
) -> float:
    """
    Reward rows (example) that are nearly complete for the AI.
    This is a placeholder; adjust for columns/blocks if needed.
    """
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
    """Simple difference in scores."""
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


###############################################################################
#                       MCTS + MINIMAX HYBRID
###############################################################################


def HYBRID_DEPTH_THRESHOLD() -> int:
    return 5  # If you want to experiment, reduce or increase this.


def call_minimax_for_evaluation(
    game_state: GameState, ai_player_index: int
) -> float:
    """
    Use Minimax at a shallow depth to evaluate the position.
    """
    valid_moves_dict = get_valid_moves(game_state)
    valid_moves = []
    for (r, c), vals in valid_moves_dict.items():
        for v in vals:
            valid_moves.append(Move((r, c), v))

    if not valid_moves:
        # No valid moves => negative infinity or some terminal scoring
        return float("-inf")

    # You can set the search_depth to 2 or 3, etc.
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


def rollout_simulation(game_state, max_depth=5) -> float:
    """
    A simple random playout. If you want a more advanced evaluation,
    you can incorporate a domain-specific heuristic or even a shallow Minimax call.
    """
    depth = 0
    current_state = copy.deepcopy(game_state)

    while depth < max_depth:
        # Get all valid moves
        moves_dict = get_valid_moves(current_state)  # Provided by your code
        valid_moves = [
            Move((r, c), v)
            for (r, c), values in moves_dict.items()
            for v in values
        ]
        if not valid_moves:
            break  # No more moves => rollout ends
        mv = random.choice(valid_moves)
        current_state = simulate_move(current_state, mv)
        depth += 1

    # Simple evaluation: difference in scores
    ai_player_index = game_state.current_player - 1
    return (
        current_state.scores[ai_player_index]
        - current_state.scores[1 - ai_player_index]
    )


###############################################################################
#                    MCTS NODE & TRANSPOSITION TABLE
###############################################################################


class MCTSNode:
    def __init__(self, game_state, move: Optional[Move]):
        self.game_state = game_state
        self.move = move
        self.parent: Optional["MCTSNode"] = None
        self.children: List["MCTSNode"] = []
        self.visit_count = 0
        self.value_sum = 0.0
        # We'll compute UCT on the fly rather than store it persistently.


def ucb_score(child: "MCTSNode", c: float = 1.4) -> float:
    """Calculate the UCB1 / UCT score for a child node."""
    if child.visit_count == 0:
        return float("inf")  # Force exploration of unvisited child

    parent_visits = child.parent.visit_count if child.parent else 1
    exploit = child.value_sum / child.visit_count
    explore = c * math.sqrt(math.log(parent_visits) / child.visit_count)
    return exploit + explore


def state_key(game_state) -> Tuple:
    """
    Create a unique key for the transposition table (if desired).
    For instance, (tuple of squares, current_player, tuple of scores).
    """
    # This only works if game_state.board.squares is a list of ints that can be converted to a tuple.
    return (
        tuple(game_state.board.squares),
        game_state.current_player,
        tuple(game_state.scores),
    )


###############################################################################
#                           MCTS TREE
###############################################################################


class MonteCarloTree:
    def __init__(self, game_state, ai_player_index: int):
        self.ai_player_index = ai_player_index
        self.root = MCTSNode(game_state, move=None)
        self.root.parent = None
        self.transposition_table = {}  # optional

    def _get_or_create_node(
        self, game_state, move: Optional[Move]
    ) -> MCTSNode:
        # Optional usage of a transposition table
        key = state_key(game_state)
        if key in self.transposition_table:
            return self.transposition_table[key]
        else:
            node = MCTSNode(game_state, move)
            self.transposition_table[key] = node
            return node

    def visit(self) -> MCTSNode:
        """
        Selection: pick a leaf node by following the child with the highest UCB
        until no children or a node that hasn't been expanded yet.
        """
        current = self.root
        while current.children:
            best_child = max(current.children, key=lambda c: ucb_score(c))
            current = best_child
        return current

    def expand(self, node: MCTSNode, max_children=5) -> MCTSNode:
        """
        Expansion: Add children for unexpanded moves.
        If there's no unexpanded move, we return the node itself (leaf).
        """
        # Gather all valid moves
        moves_dict = get_valid_moves(node.game_state)  # Provided by your code
        valid_moves = [
            Move((r, c), v)
            for (r, c), vals in moves_dict.items()
            for v in vals
        ]
        # Already expanded
        expanded_moves = {
            (child.move.square, child.move.value)
            for child in node.children
            if child.move
        }

        # Filter unexpanded
        unexpanded = [
            mv
            for mv in valid_moves
            if (mv.square, mv.value) not in expanded_moves
        ]
        if not unexpanded:
            return node  # can't expand further

        random.shuffle(unexpanded)
        to_expand = unexpanded[:max_children]
        for mv in to_expand:
            nxt_state = simulate_move(node.game_state, mv)
            child_node = self._get_or_create_node(nxt_state, mv)
            # Link child if not linked yet
            if child_node.parent is None:
                child_node.parent = node
                node.children.append(child_node)

        # Return one newly created child as the node to simulate from
        return random.choice(node.children) if node.children else node

    def simulate(self, node: MCTSNode) -> float:
        """
        Simulation (rollout). If you want a 'hybrid' with minimax or heuristics,
        replace or extend 'rollout_simulation' accordingly.
        """
        return rollout_simulation(node.game_state, max_depth=5)

    def backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation: propagate the result (reward) up to the root.
        """
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += reward
            current = current.parent

    def do_iteration(self):
        # Selection
        leaf = self.visit()

        # If leaf is unvisited, we skip expansion to let the rollout happen from here
        if leaf.visit_count > 0:
            # Expand
            leaf = self.expand(leaf, max_children=5)

        # Simulation
        reward = self.simulate(leaf)

        # Backpropagation
        self.backpropagate(leaf, reward)

    def best_move(self) -> Optional[Move]:
        """
        Return the move of the child with the highest visit_count from the root.
        If no children, return None.
        """
        if not self.root.children:
            return None
        best_child = max(self.root.children, key=lambda c: c.visit_count)
        return best_child.move

    def search(self, iterations=1000, time_limit=2.0):
        """
        Conduct the MCTS search. You can limit by iteration count or time.
        """
        start_time = time.time()
        for i in range(iterations):
            if time.time() - start_time > time_limit:
                break
            self.do_iteration()
