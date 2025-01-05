import math
import random
import copy
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard

from A3_MCMC.helper_functions import get_valid_moves, simulate_move


def score_trap_opponent(game_state: GameState, ai_player_index: int) -> float:
    """
    Returns a high positive value if the opponent has very few or zero valid moves.
    Returns a smaller value if the opponent has many moves.
    """
    original_player = game_state.current_player
    opponent_index = 1 - ai_player_index
    game_state.current_player = opponent_index + 1
    opp_moves = get_valid_moves(game_state)
    game_state.current_player = original_player

    total_moves = sum(len(vals) for vals in opp_moves.values())
    if total_moves == 0:
        return 100.0
    else:
        return 10.0 / (total_moves + 0.01)


def score_region_completion(
    game_state: GameState, ai_player_index: int
) -> float:
    return (
        game_state.scores[ai_player_index]
        - game_state.scores[1 - ai_player_index]
    )


# def score_state(game_state: GameState, ai_player_index: int) -> float:
#     trap_score = score_trap_opponent(game_state, ai_player_index)
#     region_score = score_region_completion(game_state, ai_player_index)
#     w_trap = 0.8
#     w_region = 0.2
#     return w_trap * trap_score + w_region * region_score


def score_state(game_state: GameState, ai_player_index: int) -> float:
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


# def rollout_evaluation(
#     state: GameState, ai_player_index: int, max_depth: int = 10
# ) -> float:
#     """
#     Optionally do random playout for 'max_depth' moves,
#     then evaluate the resulting board with 'score_state'.
#     """
#     current = copy.deepcopy(state)
#     for _ in range(max_depth):
#         moves_dict = get_valid_moves(current)
#         moves_list = [
#             Move((r, c), v)
#             for (r, c), vals in moves_dict.items()
#             for v in vals
#         ]
#         if not moves_list:
#             break
#         mv = random.choice(moves_list)
#         current = simulate_move(current, mv)
#     return score_state(current, ai_player_index)


def rollout_evaluation(
    state: GameState, ai_player_index: int, max_depth: int = 10
) -> float:
    current = copy.deepcopy(state)
    for _ in range(max_depth):
        moves_dict = get_valid_moves(current)
        if not moves_dict:
            break

        all_moves = []
        for (r, c), vals in moves_dict.items():
            for v in vals:
                mv = Move((r, c), v)
                nxt = simulate_move(current, mv)
                sc = score_state(nxt, ai_player_index)
                all_moves.append((mv, sc))

        if not all_moves:
            break

        all_moves.sort(key=lambda x: x[1], reverse=True)
        # Picking top 12.5% cuz y not
        top_moves = all_moves[: max(1, len(all_moves) // 10)]

        chosen_mv = random.choice(top_moves)[0]
        current = simulate_move(current, chosen_mv)

    return score_state(current, ai_player_index)


def weighted_random_move(
    game_state: GameState, ai_player_index: int, temperature: float = 1.0
):
    """
    Picks a move from get_valid_moves(game_state) with probabilities
    ~ exp( rollout_score / temperature ) (softmax).
    """
    moves_dict = get_valid_moves(game_state)
    moves_list = []
    for (r, c), vals in moves_dict.items():
        for v in vals:
            moves_list.append(Move((r, c), v))

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
):
    """
    Markov Chain Monte Carlo random walk with a simulated annealing flavor.
    Returns the best (first) move from 'root_state' found during the walk.
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

        # Gather valid moves
        moves_dict = get_valid_moves(current_state)
        moves_list = [
            Move((r, c), v)
            for (r, c), vals in moves_dict.items()
            for v in vals
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
