from team21_A2.helper_functions import calculate_new_game_board, is_board_full, get_legal_moves
from team21_A2.evaluation import evaluate_move
from team21_A2.heuristics.hidden_twin_exclusion import hidden_twin_exclusion

from competitive_sudoku.sudoku import SudokuBoard, Move


class Node:
    """
    A class used to represent a move and its effects as a node in a search tree
    All Sudoku-related attributes are AFTER the Nodes' move

    Attributes
    ----------
    score : int
        The score of our agent minus the score of the enemy agent after our move
    move : Move
        the move that is being done
    game_board : SudokuBoard
        The SudokuBoard after the move is applied
    father : Node
        the parent of the current Node in the search tree
    children : list
        the list of children of the current Node in the search tree
    our_move : boolean
        Stores whether it is currently our move (True) or the enemies' move (False)
    full_board : boolean
        Stores whether the board is currently full (so that no new moves can be made)

    Methods
    -------
    add_child(child)
        Adds a Node to the list of children
    is_leaf()
        Returns whether the Node has children or not (is a leaf in the search tree)
    evaluate()
        Evaluates what score the current board would have (after the move) and saves it in the Node
    """

    def __init__(self, score: int = None, move: Move = None, game_board: SudokuBoard = None, father=None, our_move=None):
        self.score = score
        self.move = move
        self.game_board = game_board
        self.father = father
        self.children = []
        self.our_move = our_move
        self.full_board = is_board_full(self.game_board)

    def add_child(self, child):
        """
        Adds a Node to the list of children
        @param child: The Node that should be added to the list of children
        """
        self.children.append(child)

    def is_leaf(self):
        """
        Returns whether the Node has children or not (is a leaf in the search tree)
        """
        return self.children == []

    def evaluate(self):
        """
        Evaluates what score the current board would have (after the move) and saves it in the Node
        """

        self.score = evaluate_move(self.game_board, self.move, self.our_move, self.score)


class Tree:
    """
    A class used to represent the search tree, consisting of Nodes, that will be searched through using minimax

    Attributes
    ----------
    root : list
        A list of Nodes containing the moves our agent can take, all roots of their own search trees.
    taboo_moves : list
        A list of moves that have been declared Taboo by the Oracle, and will disqualify us if we try them
    init_score : int
        The score at the start of our turn

    Methods
    -------
    add_children_to_root(Node)
        Increases the depth of the search tree rooted at the given Node by 1
    find_best_move()
        Returns the best move of the search trees in a Tree object using the minimax algorithm with alpha-beta pruning
    deepen_search()
        Increases the depth of the search tree by 1 for all Nodes in the root, and returns the best move for that depth
    """
    def __init__(self, root_moves, taboo_moves, future_taboo_move, cur_game_board, init_score):
        self.root = []
        self.taboo_moves = taboo_moves
        self.init_score = init_score

        # After initializing the parameters, we evaluate the possible moves on our turn,
        # and put each in the root list to serve as roots for their own search trees
        for cur_move in root_moves:
            new_game_board = calculate_new_game_board(cur_game_board, cur_move)
            cur_node = Node(score=init_score, move=cur_move, game_board=new_game_board, our_move=True)
            cur_node.evaluate()

            self.root.append(cur_node)

        # We separately add the future_taboo_move, if any, into the list of root moves as well
        if future_taboo_move is not None:
            future_taboo_node = Node(score=init_score, move=future_taboo_move, game_board=cur_game_board, our_move=True)
            self.root.append(future_taboo_node)

    def add_children_to_root(self, parent_node: Node):
        """
        Increase the depth of the search tree rooted at the given Node by 1
        @param parent_node: The Node acting as the root of the search tree that should be deepened
        """

        # Find leaf nodes:

        if len(parent_node.children) > 0:
            # If it has children, it is NOT a leaf node, so check its' children for leaf nodes
            # This way, we recursively check every node in the search tree
            for child in parent_node.children:
                self.add_children_to_root(child)

        else:
            # This is a leaf node, so we need to give it children

            if parent_node.full_board:
                # If the board is full on this leaf, we can't add any legal moves as children, so we skip it.
                return
            else:
                # Add all possible moves of the current node to the list of children of this node
                # First we retrieve the list of moves, separating moves that will be declared as taboo by the Oracle
                non_taboo_moves, future_taboo_move = get_heuristic_moves(parent_node.game_board, self.taboo_moves)
                for cur_move in non_taboo_moves:
                    # We add all non-future taboo moves by calculating what the effects of that move would look like,
                    # and then putting them into a node and adding it to the list of children of the node that is being deepened.
                    cur_game_board = calculate_new_game_board(parent_node.game_board, cur_move)

                    cur_node = Node(score=parent_node.score, move=cur_move, game_board=cur_game_board,
                                    father=parent_node, our_move=not parent_node.our_move)
                    cur_node.evaluate()

                    parent_node.add_child(cur_node)

                if future_taboo_move is not None:
                    # If we found a move that will be marked as taboo once we play it, we also add that node to the search tree.
                    # We only have to put in one of these moves, because they are all identical: They just skip our turn.
                    # We can also skip some functions that are unnecessary because our turn is skipped
                    cur_game_board = parent_node.game_board
                    cur_node = Node(score=parent_node.score, move=future_taboo_move, game_board=cur_game_board,
                                    father=parent_node, our_move=not parent_node.our_move)

                    parent_node.add_child(cur_node)

    def find_best_move(self):
        """
        Returns the best move of the search trees in a Tree object using the minimax algorithm with alpha-beta pruning
        """
        best_node = None
        best_score = -987654321  # Just a funny low number
        for root_node in self.root:
            # For every search tree, start doing minimax with alpha-beta pruning on its' root
            root_node.score = min_beta(root_node, -999, 999)

            # Whenever the current search tree gives us a better move than all the search trees we already tried, save it as the best move
            if root_node.score > best_score:
                best_score = root_node.score
                best_node = root_node
        print("New score with best move: " + str(best_score))  # Just some extra printing so you can easily keep track
        if best_score == -987654321:  # No move found: We have the last move of the game
            return
        else:  # We found a best move, quick send it back before the Competitive Sudoku timer expires oh god oh fu-
            return best_node.move

    def deepen_search(self):
        """
        Increases the depth of the search tree by 1 for all Nodes in the root, and returns the best move for that depth
        """
        for node in self.root:
            self.add_children_to_root(node)
        return self.find_best_move()


def min_beta(node: Node, alpha, beta):
    """
    Returns the score if the Node is a leaf node, or checks what move would hurt us the most
    @param node: the Node object that is being evaluated
    @param alpha: the the minimum score that the maximizing player is assured of
    @param beta: the maximum score that the minimizing player is assured
    """
    if node.is_leaf():
        # If the current node is a leaf node, we simply return its' score
        return node.score
    else:
        # We start looking at the nodes' children to find the score that would be the worst for us,
        # which is the move our opponent should use, and return that score
        score = 999999999
        for node in node.children:
            score = min(score, max_alpha(node, alpha, beta))
            if score <= alpha:
                # The Node (move) we are checking would never get played with this score, so we can stop checking it
                return score
            beta = min(beta, score)
        return score


def max_alpha(node: Node, alpha, beta):
    """
    Returns the score if the Node is a leaf node, or checks what move would benefit us the most
    @param node: the Node object that is being evaluated
    @param alpha: the the minimum score that the maximizing player is assured of
    @param beta: the maximum score that the minimizing player is assured
    """
    if node.is_leaf():
        # If the current node is a leaf node, we simply return its' score
        return node.score
    else:
        # We start looking at the nodes' children to find the score that would be the best for us,
        # which is the move we should use, and return that score
        score = -999999999
        for node in node.children:
            score = max(score, min_beta(node, alpha, beta))
            if score >= beta:
                # The Node (move) we are checking would never get played with this score, so we can stop checking it
                return score
            alpha = max(alpha, score)
        return score


def get_heuristic_moves(game_board: SudokuBoard, taboo_moves):
    """
    Runs the heuristics on the set of all legal moves, separating moves that can solve the Sudoku board from moves that cannot
    Returns a (hopefully) smaller list of legal moves and a single move that the Oracle would identify as taboo
    @param game_board: The current Sudoku board
    @param taboo_moves: The list of moves declared taboo by the Oracle
    """

    filtered_moves = get_legal_moves(game_board, taboo_moves)
    future_taboo_moves = None

    # We return a filtered list of legal moves, and a taboo move if one was found by the heuristic,
    # and then update our variables with the result
    # We can simply repeat the same process for every heuristic, using the filtered list as input for the new heuristic
    heuristic_filtered, heuristic_taboo = hidden_twin_exclusion(game_board, filtered_moves)
    filtered_moves = heuristic_filtered
    if future_taboo_moves is None and heuristic_taboo is not None:
        future_taboo_moves = heuristic_taboo

    # Once we've used all the heuristics, we return the filtered list of legal moves and a taboo move (if any).
    return filtered_moves, future_taboo_moves
