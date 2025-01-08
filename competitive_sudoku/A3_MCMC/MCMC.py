import math
import random
import copy
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard

from A3_MCMC.helper_functions import get_valid_moves, simulate_move
from A3_MCMC.taboo_helpers import naked_singles


def score_trap_opponent(game_state: GameState, ai_player_index: int) -> float:
    """
    Evaluates the advantage of limiting the opponent's valid moves.

    Args:
        game_state (GameState): The current game state.
        ai_player_index (int): The index of the AI player (0 or 1).

    Returns:
        float: A high score (e.g., 100.0) if the opponent has no valid moves,
               or a lower score inversely proportional to the number of valid moves.
    """
    original_player = game_state.current_player
    opponent_index = 1 - ai_player_index
    game_state.current_player = opponent_index + 1
    opp_moves = get_valid_moves(game_state)
    opp_moves = naked_singles(game_state, opp_moves)

    game_state.current_player = original_player

    total_moves = sum(len(vals) for vals in opp_moves.values())
    if total_moves == 0:
        return 100.0
    else:
        return 10.0 / (total_moves + 0.01)


def score_region_completion(
    game_state: GameState, ai_player_index: int
) -> float:
    """
    Computes the score difference between the AI player and the opponent.

    Args:
        game_state (GameState): The current game state.
        ai_player_index (int): The index of the AI player (0 or 1).

    Returns:
        float: The difference between the AI's score and the opponent's score.
    """
    return (
        game_state.scores[ai_player_index]
        - game_state.scores[1 - ai_player_index]
    )


def score_state(game_state: GameState, ai_player_index: int) -> float:
    """
    Aggregates various scoring metrics to evaluate the current game state.

    Args:
        game_state (GameState): The current game state.
        ai_player_index (int): The index of the AI player (0 or 1).

    Returns:
        float: A weighted sum of the trap opponent score, region completion score,
               and self-mobility score.
    """
    trap_opponent = score_trap_opponent(game_state, ai_player_index)
    region_score = score_region_completion(game_state, ai_player_index)
    my_mobility = score_self_mobility(game_state, ai_player_index)

    w_trap = 0.5
    w_region = 0.2
    w_mobility = 0.3
    return (
        w_trap * trap_opponent
        + w_region * region_score
        + w_mobility * my_mobility
    )


def rollout_evaluation(
    state: GameState, ai_player_index: int, max_depth: int = 10
) -> float:
    """
    Simulates a sequence of moves from the current state to estimate its value.

    Args:
        state (GameState): The initial game state for the rollout.
        ai_player_index (int): The index of the AI player (0 or 1).
        max_depth (int): The maximum number of moves to simulate (default: 10).

    Returns:
        float: The evaluated score of the game state after the rollout.
    """
    game_state = copy.deepcopy(state)
    for _ in range(max_depth):
        moves_dict = get_valid_moves(game_state)
        moves_dict = naked_singles(game_state, moves_dict)

        if not moves_dict:
            break

        all_moves = []
        for (row, col), vals in moves_dict.items():
            for val in vals:
                mv = Move((row, col), val)
                nxt = simulate_move(game_state, mv)
                sc = score_state(nxt, ai_player_index)
                all_moves.append((mv, sc))

        if not all_moves:
            break

        all_moves.sort(key=lambda x: x[1], reverse=True)
        # Picking top 12.5% cuz y not
        top_moves = all_moves[: max(1, len(all_moves) // 10)]

        chosen_mv = random.choice(top_moves)[0]
        game_state = simulate_move(game_state, chosen_mv)

    return score_state(game_state, ai_player_index)


def weighted_random_move(
    game_state: GameState, ai_player_index: int, temperature: float = 1.0
):
    """
    Selects a move using a softmax probability distribution based on rollout evaluations.

    Args:
        game_state (GameState): The current game state.
        ai_player_index (int): The index of the AI player (0 or 1).
        temperature (float): The exploration temperature (default: 1.0).

    Returns:
        Move: The selected move, or None if no valid moves exist.
    """
    moves_dict = get_valid_moves(game_state)
    moves_dict = naked_singles(game_state, moves_dict)

    moves_list = []
    for (row, col), vals in moves_dict.items():
        for val in vals:
            moves_list.append(Move((row, col), val))

    if not moves_list:
        return None

    scores = []
    for m in moves_list:
        next_st = simulate_move(game_state, m)
        val = rollout_evaluation(next_st, ai_player_index, max_depth=5)
        scores.append(val)

    exps = [math.exp(s / temperature) for s in scores]
    total = sum(exps)
    if total <= 1e-9:
        return random.choice(moves_list)

    rnd = random.random() * total
    cumulative = 0.0
    for move, w in zip(moves_list, exps):
        cumulative += w
        if rnd < cumulative:
            return move

    return moves_list[-1]


def score_self_mobility(game_state: GameState, ai_player_index: int) -> float:
    """
    Evaluates the mobility of the AI player by counting its valid moves.

    Args:
        game_state (GameState): The current game state.
        ai_player_index (int): The index of the AI player (0 or 1).

    Returns:
        float: A score proportional to the logarithm of the number of valid moves.
    """
    original_player = game_state.current_player
    game_state.current_player = ai_player_index + 1  # 1-based
    my_moves = get_valid_moves(game_state)
    game_state.current_player = original_player

    total_moves = sum(len(vals) for vals in my_moves.values())
    return 10.0 * math.log1p(total_moves)  # or some function


def mcmc_search(
    root_state: GameState,
    ai_player_index: int,
    iterations: int = 500,
    temperature: float = 2.0,
    time_limit: float = 1.0,
    rollout_depth: int = 10,
) -> tuple:
    """
    Performs a Markov Chain Monte Carlo (MCMC) search with a simulated annealing strategy.

    Args:
        root_state (GameState): The initial game state.
        ai_player_index (int): The index of the AI player (0 or 1).
        iterations (int): The maximum number of MCMC iterations (default: 500).
        temperature (float): The initial temperature for simulated annealing (default: 2.0).
        time_limit (float): The maximum allowed time for the search in seconds (default: 1.0).
        rollout_depth (int): The maximum depth for rollouts during evaluations (default: 10).

    Returns:
        tuple: The best move found during the search, or None if no moves are found.
    """
    start_time = time.time()

    def state_id(gs: GameState):
        return (tuple(gs.board.squares), gs.current_player, tuple(gs.scores))

    current_state = copy.deepcopy(root_state)
    current_score = score_state(current_state, ai_player_index)

    best_state = current_state
    best_score = current_score

    first_move_map = {}
    root_key = state_id(root_state)
    first_move_map[root_key] = None

    def ensure_map_entry(gs: GameState):
        sid = state_id(gs)
        if sid not in first_move_map:
            first_move_map[sid] = None

    ensure_map_entry(current_state)

    for step in range(iterations):
        if time.time() - start_time > time_limit:
            break
        # More greedy as we play more moves
        temperature *= 0.99
        if temperature < 0.2:
            temperature = 0.2

        moves_dict = get_valid_moves(current_state)
        moves_dict = naked_singles(current_state, moves_dict)
        moves_list = [
            Move((row, col), val)
            for (row, col), vals in moves_dict.items()
            for val in vals
        ]

        if not moves_list:
            break

        move_chosen = weighted_random_move(
            current_state, ai_player_index, temperature
        )

        if move_chosen is None:
            if moves_list:
                move_chosen = random.choice(moves_list)
            else:
                break

        if move_chosen is None:
            print("[DEBUG] No move chosen, ending MCMC iteration.")
            break

        next_state = simulate_move(current_state, move_chosen)

        next_score = score_state(next_state, ai_player_index)

        old_score = score_state(current_state, ai_player_index)
        delta = next_score - old_score
        if delta > 0:
            accept_prob = 1.0
        else:
            accept_prob = math.exp(delta / temperature)

        if random.random() < accept_prob:
            old_key = state_id(current_state)
            current_state = next_state
            new_key = state_id(current_state)
            ensure_map_entry(current_state)

            if first_move_map[new_key] is None:
                if old_key == root_key:
                    first_move_map[new_key] = move_chosen
                else:
                    first_move_map[new_key] = first_move_map[old_key]

            if next_score > best_score:
                best_score = next_score
                best_state = copy.deepcopy(current_state)
        else:
            pass

    best_key = state_id(best_state)
    best_move = first_move_map.get(best_key, None)

    return best_move
