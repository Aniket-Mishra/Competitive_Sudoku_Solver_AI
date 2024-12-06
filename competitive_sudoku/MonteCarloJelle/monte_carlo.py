import math
import random
import copy
from typing import List
from team21_A3_monte_carlo.heuristics.hidden_twin_exclusion import hidden_twin_exclusion
from team21_A3_monte_carlo.heuristics.only_square2 import only_square
from team21_A3_monte_carlo.evaluation import evaluate_move
from competitive_sudoku.sudoku import SudokuBoard, Move


C_PARAMETER = 2


class Node:
    def __init__(self, previous=None, board=None, our_move=True, move=None, moves=[], nr_of_visits=0, total_score=0):
        self.previous = previous
        self.board = board
        self.our_move = our_move
        self.move = move
        self.moves = moves
        self.nr_of_visits = nr_of_visits
        self.total_score = total_score
        self.children = []

    def add_child(self, node):
        """
        Method for adding a child to the current node.
        """
        self.children.append(node)

    def is_leaf(self):
        """
        Method for checking whether the current node is a leaf.
        """
        return len(self.children) == 0

    def calculate_uct(self, node):
        """
        Method for calculating the Upper Confidence Bound for Trees (UCT).
        @param node: The child node to calculate the UCT for
        """
        if node.nr_of_visits == 0:
            return math.inf

        return (node.total_score / node.nr_of_visits) + C_PARAMETER * math.sqrt((math.log(self.nr_of_visits) / node.nr_of_visits))

    def propagate(self, delta):
        if self.our_move:
            self.total_score += delta
        else:
            self.total_score -= delta
        self.nr_of_visits += 1

        if self.previous != None:
            self.previous.propagate(delta)

    def select_leaf(self):
        """
        Method for selecting leaf based on the maximizing the UCT values.
        """
        if self.is_leaf():
            return self

        selected_child = self.children[0]
        for child in self.children:
            if self.calculate_uct(child) > self.calculate_uct(selected_child):
                selected_child = child

        return selected_child.select_leaf()


def simulate(node: Node):
    """
    Method for simulation a full game from the given node using random moves and return if we result.
    """
    board = copy.deepcopy(node.board)
    legal_moves = copy.deepcopy(node.moves)
    our_move = node.our_move
    first_player = node.our_move
    score = 0

    while(len(legal_moves) > 0):
        move_index = random.randrange(len(legal_moves))
        move = legal_moves[move_index]
        board.put(move.i, move.j, move.value)
        score += evaluate_move(board, move, our_move, 0)
        legal_moves.remove(move)
        our_move = not our_move

    if first_player and score > 0:
        return 1
    return -1


def max_child_factor(node: Node):
    """
    Method for calculation the factor for maximizing the child to select.
    """
    if node.nr_of_visits == 0:
        return -math.inf
    return node.total_score / node.nr_of_visits


def find_best_move(node: Node):
    """
    Method for finding the best move using the max child factor.
    """
    if node.is_leaf():
        return node.move

    best_child = node.children[0]
    for child in node.children:
        if max_child_factor(child) > max_child_factor(best_child):
            best_child = child

    return best_child.move


def monte_carlo(board: SudokuBoard, our_move: bool, moves: List[Move], propose_move):
    """
    Method for apply the monte carlo search iteratively until it 
    """
    root = Node(None, board, our_move, None, moves)

    for m in moves:
        new_board = copy.deepcopy(root.board)
        new_board.put(m.i, m.j, m.value)
        new_moves = copy.deepcopy(moves)
        new_moves.remove(m)

        root.add_child(Node(root, new_board, not our_move, m, new_moves))

    # Keep computing until we run out
    while True:
        # Select leaf
        v = root.select_leaf()

        # Leaf expansion
        skip_simulation = False
        selected_child = None
        delta = 0

        if len(v.moves) == 0:
            skip_simulation = True
            selected_child = v
        elif v.nr_of_visits == 0:
            selected_child = v
        else:
            for move in moves:
                new_board = v.board.put(move.i, move.j, move.value)
                v.add_child(Node(v, new_board, not v.our_move))

            selected_child_index = random.randrange(len(v.children))
            selected_child = v.children[selected_child_index]

        # Simulation
        if not skip_simulation:
            simulate(selected_child)

        # Backpropagation
        selected_child.propagate(delta)

        # Find best move
        best_move = find_best_move(root)

        # Propose move for agent
        propose_move(best_move)