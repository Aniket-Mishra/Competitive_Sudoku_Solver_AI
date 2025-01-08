import math
import random
import time
from competitive_sudoku.sudoku import GameState, Move

from A3_MCTS_with_minimax.helper_functions import (
    get_valid_moves,
    simulate_move,
)
from A3_MCTS_with_minimax.minimax import (
    minimax,
)

from A3_MCTS_with_minimax.taboo_helpers import naked_singles


def score_mobility(game_state: GameState, ai_player_index: int) -> float:
    """
    Computes the mobility score for the current game state.

    This score is based on the number of valid moves the AI can make compared to its opponent. 
    A higher score indicates a greater number of valid moves for the AI relative to the opponent.

    Parameters:
    - game_state (GameState): The current state of the Sudoku game.
    - ai_player_index (int): The index of our agent.

    Returns:
    - float: The difference in mobility between the AI and its opponent.
    """

    original_player = game_state.current_player
    game_state.current_player = ai_player_index + 1
    ai_moves = get_valid_moves(game_state)
    ai_moves = naked_singles(game_state, ai_moves)
    ai_count = sum(len(vals) for vals in ai_moves.values())

    opp_index = 1 - ai_player_index
    game_state.current_player = opp_index + 1
    opp_moves = get_valid_moves(game_state)
    opp_moves = naked_singles(game_state, opp_moves)
    opp_count = sum(len(vals) for vals in opp_moves.values())

    game_state.current_player = original_player

    return ai_count - opp_count


def score_difference(game_state: GameState, ai_player_index: int) -> float:
    """
    Calculate game state

    Args:
        game_state (GameState): Current game state
        ai_player_index (int): The index of our agent

    Returns:
        float: score diff of our agent vs opponent
    """
    return game_state.scores[ai_player_index] - game_state.scores[1 - ai_player_index]


def evaluate_domain_heuristics(game_state: GameState, ai_player_index: int) -> float:
    """
    Combines multiple heuristics to evaluate the current game state.

    This function aggregates scores for mobility and score difference 
    to provide a comprehensive evaluation of the game state.

    Parameters:
    - game_state (GameState): The current state of the Sudoku game.
    - ai_player_index (int): The index of our agent

    Returns:
    - float: A weighted sum of heuristic scores representing the game state's favorability for the AI.
    """
    w_mobility = 0.5
    w_diff = 0.5

    mob = score_mobility(game_state, ai_player_index)
    diff = score_difference(game_state, ai_player_index)

    return w_mobility * mob + w_diff * diff


def call_minimax_for_evaluation(game_state: GameState, valid_moves, ai_player_index: int) -> float:
    """
    Calls Minimax to evaluate the game state at deeper depths.

    This function performs a Minimax search to assess the outcome of the current game state 
    when MCTS reaches its depth threshold.

    Parameters:
    - game_state (GameState): The current state of the Sudoku game.
    - ai_player_index (int): The index of our agent

    Returns:
    - float: The evaluation score from the Minimax algorithm.
    """
    if not valid_moves:
        return float("-inf")

    search_depth = 2
    best_score = float("-inf")
    for move in valid_moves:
        next_state = simulate_move(game_state, move)
        score = minimax(next_state, search_depth, alpha=float(
            "-inf"), beta=float("inf"), maximizing=False, ai_player_index=ai_player_index,)
        if score > best_score:
            best_score = score
    return best_score


def rollout_simulation(game_state: GameState, ai_player_index: int, depth: int = 0, max_depth=10) -> float:
    """
    Simulates a random playout from the current game state.

    This function performs a random simulation or calls Minimax when the depth exceeds the 
    hybrid threshold, returning an evaluation score.

    Parameters:
    - game_state (GameState): The current state of the Sudoku game.
    - ai_player_index (int): The index of the AI player (0 for player 1, 1 for player 2).
    - depth (int): The current depth of the simulation (default is 0).
    - max_depth (int): The maximum depth for the simulation (default is 10).

    Returns:
    - float: The evaluation score of the simulated playout.
    """
    moves_dict = get_valid_moves(game_state)
    moves_dict = naked_singles(game_state, moves_dict)
    valid_moves = [
        Move((row, col), val)
        for (row, col), vals in moves_dict.items()
        for val in vals
    ]

    depth_threshold = 2
    if depth > depth_threshold:
        mm_val = call_minimax_for_evaluation(game_state, valid_moves, ai_player_index)
        heuristic_val = evaluate_domain_heuristics(game_state, ai_player_index)
        return mm_val + 0.1 * heuristic_val

    if depth >= max_depth:
        return evaluate_domain_heuristics(game_state, ai_player_index)


    if not valid_moves:
        return evaluate_domain_heuristics(game_state, ai_player_index)

    random_move = random.choice(valid_moves)
    next_state = simulate_move(game_state, random_move)
    return rollout_simulation(next_state, ai_player_index, depth + 1, max_depth)


class MCTSNode:
    """
    Initializes a Monte Carlo Tree Search (MCTS) node.

    Parameters:
    - game_state (GameState): The game state represented by this node.
    - move (Move | None): The move that led to this node.
    """

    def __init__(self, game_state: GameState, move: Move | None):
        self.game_state = game_state
        self.move = move
        self.parent = None
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.UCT_Score = float("inf")

    def add_child(self, child_node):
        """
        Adds a child node to the current node.

        Parameters:
        - child_node (MCTSNode): The child node to add.
        """
        child_node.parent = self
        self.children.append(child_node)


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

    uct = (node.value_sum / node.visit_count) + (exploration_constant *
                                                 math.sqrt(math.log(node.parent.visit_count) / node.visit_count))

    return uct


def state_key(game_state: GameState) -> tuple:
    """
    Generates a hashable key for the game state.

    This key uniquely identifies a game state for use in transposition tables.

    Parameters:
    - game_state (GameState): The current state of the Sudoku game.

    Returns:
    - tuple: A unique key representing the game state.
    """

    return (
        tuple(game_state.board.squares),
        game_state.current_player,
        tuple(game_state.scores),
    )


class MonteCarloTree:
    """
    Initializes the Monte Carlo Tree.

    This class manages the MCTS process, including visiting nodes, expanding the tree, 
    and backpropagating results.

    Parameters:
    - game_state (GameState): The root game state.
    - ai_player_index (int): The index of our agent.
    """

    def __init__(self, game_state: GameState, ai_player_index: int):
        self.ai_player_index = ai_player_index
        self.transposition_table = {}
        self.root = self._get_or_create_node(game_state, move=None)

    def _get_or_create_node(self, game_state: GameState, move: Move | None):
        """
        Retrieves or creates a node for a given game state.

        Checks if a node exists for the game state in the transposition table. If not, 
        creates a new node and stores it.

        Parameters:
        - game_state (GameState): The game state to retrieve or create a node for.
        - move (Move | None): The move that led to this game state.

        Returns:
        - MCTSNode: The node associated with the game state.
        """

        key = state_key(game_state)
        if key in self.transposition_table:
            node = self.transposition_table[key]
            return node
        else:
            node = MCTSNode(game_state, move)
            self.transposition_table[key] = node
            return node

    def visit(self) -> MCTSNode:
        """
        Selects the best node to visit based on UCT scores.

        This function performs the selection phase of MCTS, traversing the tree to 
        find the most promising node.

        Returns:
        - MCTSNode: The node selected for further exploration.
        """
        current_node = self.root
        while current_node.children:
            current_node = max(
                current_node.children, key=lambda col: col.UCT_Score
            )
        return current_node

    def expand(self, node: MCTSNode, max_new_children=5) -> MCTSNode:
        """
        Expands the tree by generating child nodes for the current node.

        This function creates new nodes for unexpanded moves and adds them as children 
        to the current node.

        Parameters:
        - node (MCTSNode): The node to expand.
        - max_new_children (int): The maximum number of children to add (default is 5).

        Returns:
        - MCTSNode: A newly expanded child node or the original node if no expansion is possible.
        """
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
        """
        Simulates a random playout from the current node.

        This function uses the rollout simulation to evaluate the game state by playing 
        random moves until a terminal state or maximum depth is reached.

        Parameters:
        - node (MCTSNode): The node to simulate a playout from.

        Returns:
        - float: The evaluation score of the playout.
        """
        return rollout_simulation(
            node.game_state, self.ai_player_index, depth=0, max_depth=10
        )

    def backpropagate(self, node: MCTSNode, result: float):
        """
        Backpropagates the result of a playout through the tree.

        This function updates the visit count and value sum of nodes along the path 
        from the simulated node back to the root.

        Parameters:
        - node (MCTSNode): The node from which to start backpropagation.
        - result (float): The evaluation score to propagate.
        """
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += result
            current.UCT_Score = uct_score(current)
            current = current.parent

    def do_iteration(self):
        """
        Performs a single iteration of MCTS.

        This includes the selection, expansion, simulation, and backpropagation phases.
        """
        leaf = self.visit()
        expanded = self.expand(leaf, max_new_children=3)
        result = self.simulate_playout(expanded)
        self.backpropagate(expanded, result)

    def search(self, iterations=500, time_limit=2.0):
        """
        Executes multiple MCTS iterations within a time or iteration limit.

        This function performs the MCTS search to identify the best move.

        Parameters:
        - iterations (int): The maximum number of iterations (default is 500).
        - time_limit (float): The time limit for the search in seconds (default is 2.0).

        Returns:
        - None
        """
        start_t = time.time()
        for _ in range(iterations):
            if (time.time() - start_t) > time_limit:
                break
            self.do_iteration()

    def best_move(self) -> Move | None:
        """
        Identifies the best move based on visit counts of child nodes.

        This function determines the move leading to the most explored node, indicating 
        the most promising move.

        Returns:
        - Move | None: The best move or None if no moves are available.
        """
        if not self.root.children:
            return None
        best_child = max(self.root.children, key=lambda x: x.visit_count)
        return best_child.move
