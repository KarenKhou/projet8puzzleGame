#perfect
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# 8-Puzzle Class
class EightPuzzle:
    def __init__(self, initial_state):
        self.state = np.array(initial_state)
        self.blank_pos = np.argwhere(self.state == 0)[0]

    def move(self, direction):
        """Move the blank tile in a given direction if possible."""
        row, col = self.blank_pos
        if direction == 'up' and row > 0:
            self._swap((row, col), (row - 1, col))
        elif direction == 'down' and row < 2:
            self._swap((row, col), (row + 1, col))
        elif direction == 'left' and col > 0:
            self._swap((row, col), (row, col - 1))
        elif direction == 'right' and col < 2:
            self._swap((row, col), (row, col + 1))

    def _swap(self, pos1, pos2):
        """Swap two positions in the puzzle."""
        self.state[tuple(pos1)], self.state[tuple(pos2)] = self.state[tuple(pos2)], self.state[tuple(pos1)]
        self.blank_pos = pos2  # Update the blank position

    def is_solved(self):
        """Check if the puzzle is solved."""
        return np.array_equal(self.state, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]]))

    def copy(self): #zyede
        """Return a copy of the current puzzle."""
        return EightPuzzle(self.state.copy())

    def legal_moves(self):
        """Returns a list of legal moves from the current state."""
        row, col = self.blank_pos
        moves = []
        if row > 0:
            moves.append('up')
        if row < 2:
            moves.append('down')
        if col > 0:
            moves.append('left')
        if col < 2:
            moves.append('right')
        return moves

    def result(self, move):
        """Returns the resulting state from applying a move."""
        new_puzzle = self.copy()
        new_puzzle.move(move)
        return new_puzzle

    def __eq__(self, other):
        return np.array_equal(self.state, other.state)

    def __hash__(self):
        return hash(self.state.tobytes())

    def __str__(self):
        """String representation of the puzzle state."""
        return '\n'.join([' '.join(map(str, row)) for row in self.state]) + '\n'
    
    def manhattan_distance(self): 
        goal_positions = {1: (0, 0), 2: (0, 1), 3: (0, 2),
                          4: (1, 0), 5: (1, 1), 6: (1, 2),
                          7: (2, 0), 8: (2, 1), 0: (2, 2)}
        distance = 0
        for i in range(3):
            for j in range(3):
                value = self.state[i][j]
                goal_i, goal_j = goal_positions[value]
                distance += abs(i - goal_i) + abs(j - goal_j)
        return distance
    
    def misplaced_tiles(self):
        goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        return np.sum(self.state != goal_state) - 1  # Don't count the blank tile (0)
    
    def _row_conflict(self):
        conflict = 0
        goal_rows = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2}
        for row in range(3):
            current_row_tiles = [self.state[row][col] for col in range(3) if self.state[row][col] != 0]
            for i in range(len(current_row_tiles)):
                for j in range(i + 1, len(current_row_tiles)):
                    if goal_rows[current_row_tiles[i]] == row and goal_rows[current_row_tiles[j]] == row and \
                            current_row_tiles[i] > current_row_tiles[j]:
                        conflict += 2
        return conflict

    def _col_conflict(self):
        conflict = 0
        goal_cols = {1: 0, 4: 0, 7: 0, 2: 1, 5: 1, 8: 1, 3: 2, 6: 2}
        for col in range(3):
            current_col_tiles = [self.state[row][col] for row in range(3) if self.state[row][col] != 0]
            for i in range(len(current_col_tiles)):
                for j in range(i + 1, len(current_col_tiles)):
                    if goal_cols[current_col_tiles[i]] == col and goal_cols[current_col_tiles[j]] == col and \
                            current_col_tiles[i] > current_col_tiles[j]:
                        conflict += 2
        return conflict

    
    def linear_conflict(self):
        conflicts = 0
        conflicts += self._row_conflict()
        conflicts += self._col_conflict()
        return conflicts
    
    def linear_conflict_with_manhattan(self):
        return self.manhattan_distance() + self.linear_conflict()



# BFS Solver
def bfs_solve(puzzle):
    """Solve the 8-puzzle using BFS and return the sequence of moves."""
    frontier = deque([(puzzle.copy(), [])])  # Queue of (puzzle, moves)
    explored = set()
    explored_nodes = 0

    while frontier:
        current_puzzle, moves = frontier.popleft()

        if current_puzzle.is_solved():
            return moves, explored_nodes  # Return the list of moves to solve the puzzle and explored nodes

        if current_puzzle not in explored:
            explored.add(current_puzzle)
            explored_nodes += 1

            for move in current_puzzle.legal_moves():
                new_puzzle = current_puzzle.result(move)
                frontier.append((new_puzzle, moves + [move]))

    return [], explored_nodes  # No solution found

# A* Search Algorithm
def a_star_solve(puzzle):
    """Solve the 8-puzzle using A* with Manhattan distance."""
    frontier = [(puzzle.manhattan_distance(), 0, puzzle, [])]  # (priority, cost, puzzle, moves)
    explored = set()
    explored_nodes = 0

    while frontier:
        _, cost, current_puzzle, moves = frontier.pop(0)

        if current_puzzle.is_solved():
            return moves, explored_nodes  # Return the list of moves and explored nodes

        if current_puzzle not in explored:
            explored.add(current_puzzle)
            explored_nodes += 1

            for move in current_puzzle.legal_moves():
                new_puzzle = current_puzzle.result(move)
                new_cost = cost + 1
                priority = new_cost + new_puzzle.manhattan_distance()
                frontier.append((priority, new_cost, new_puzzle, moves + [move]))

            frontier.sort(key=lambda x: x[0])  # Sort by priority (Manhattan + cost)

    return [], explored_nodes  # No solution found



# A* Search Algorithm using Misplaced Tiles heuristic
def a_star_misplaced_tiles(puzzle):
    frontier = [(puzzle.misplaced_tiles(), 0, puzzle, [])]  # (priority, cost, puzzle, moves)
    explored = set()
    explored_nodes = 0

    while frontier:
        # Sort frontier by priority (priority, cost, puzzle, moves)
        frontier.sort(key=lambda x: x[0])
        priority, cost, current_puzzle, moves = frontier.pop(0)

        if current_puzzle.is_solved():
            return moves, explored_nodes

        if current_puzzle not in explored:
            explored.add(current_puzzle)
            explored_nodes += 1

            for move in current_puzzle.legal_moves():
                new_puzzle = current_puzzle.result(move)
                new_cost = cost + 1
                new_priority = new_cost + new_puzzle.misplaced_tiles()
                frontier.append((new_priority, new_cost, new_puzzle, moves + [move]))

    return [], explored_nodes

def a_star_linear_conflict(puzzle):
    frontier = [(puzzle.linear_conflict_with_manhattan(), 0, puzzle, [])]
    explored = set()
    explored_nodes = 0

    while frontier:
        _, cost, current_puzzle, moves = frontier.pop(0)

        if current_puzzle.is_solved():
            return moves, explored_nodes

        if current_puzzle not in explored:
            explored.add(current_puzzle)
            explored_nodes += 1

            for move in current_puzzle.legal_moves():
                new_puzzle = current_puzzle.result(move)
                new_cost = cost + 1
                priority = new_cost + new_puzzle.linear_conflict_with_manhattan()
                frontier.append((priority, new_cost, new_puzzle, moves + [move]))

            frontier.sort(key=lambda x: x[0])

    return [], explored_nodes

def a_star_uniform_cost(puzzle):
    """Solve the 8-puzzle using A* with a heuristic of 0, equivalent to Uniform Cost Search."""
    frontier = [(0, 0, puzzle, [])]  # (priority, cost, puzzle, moves)
    explored = set()
    explored_nodes = 0

    while frontier:
        # Sort frontier by priority (priority, cost, puzzle, moves)
        frontier.sort(key=lambda x: x[0])
        priority, cost, current_puzzle, moves = frontier.pop(0)

        if current_puzzle.is_solved():
            return moves, explored_nodes

        if current_puzzle not in explored:
            explored.add(current_puzzle)
            explored_nodes += 1

            for move in current_puzzle.legal_moves():
                new_puzzle = current_puzzle.result(move)
                new_cost = cost + 1
                new_priority = new_cost  # No heuristic, so only the cost counts
                frontier.append((new_priority, new_cost, new_puzzle, moves + [move]))

    return [], explored_nodes


def print_solution_path(puzzle_initial_state, solution_moves):
    """Print the puzzle states along the solution path."""
    current_puzzle = EightPuzzle(puzzle_initial_state.copy())  # Start from the initial state
    print("Initial state:")
    print(current_puzzle)

    for move in solution_moves:
        current_puzzle.move(move)
        print(f"After move {move}:")
        print(current_puzzle)


# Puzzle Display and Button-based Manual Progression
def update_display(puzzle, ax):
    """Update the puzzle display."""
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    # Reverse row index to show first row at the top
    for i in range(3):
        for j in range(3):
            value = puzzle.state[2 - i][j]  # Reverse row index
            label = '' if value == 0 else str(value)
            ax.text(j + 0.5, i + 0.5, label, ha='center', va='center', fontsize=45, fontweight='bold',
                    bbox=dict(facecolor='lightgray' if value == 0 else 'white', 
                              edgecolor='black', boxstyle='round,pad=0.6', linewidth=2))

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)  # Adjust to fit the button
    plt.draw()

def on_click(event, puzzle, ax, solution_moves, step_counter):
    """Handle button click to go to the next move."""
    if step_counter[0] < len(solution_moves):
        move = solution_moves[step_counter[0]]
        puzzle.move(move)
        update_display(puzzle, ax)
        step_counter[0] += 1

def manual_animation_with_button(puzzle_initial_state, solution_moves):
    fig, ax = plt.subplots(figsize=(5, 5))

    # Create a button
    ax_next = plt.axes([0.8, 0.02, 0.15, 0.07])  # Position for button
    next_btn = Button(ax_next, 'Next')

    # Initialize the puzzle to the original initial state (resetting it)
    puzzle = EightPuzzle(puzzle_initial_state.copy())  # Reset puzzle to initial state

    # Initialize step counter for manual progression
    step_counter = [0]

    # Button callback
    next_btn.on_clicked(lambda event: on_click(event, puzzle, ax, solution_moves, step_counter))

    # Initial display of the puzzle
    update_display(puzzle, ax)
    plt.show()


def is_solvable(state):
    """Check if the 8-puzzle state is solvable."""
    flat_state = state.flatten()  # Flatten the array into a 1D array
    inversions = 0
    
    # Count inversions
    for i in range(len(flat_state)):
        for j in range(i + 1, len(flat_state)):
            if flat_state[i] > flat_state[j] and flat_state[i] != 0 and flat_state[j] != 0:
                inversions += 1

    # A puzzle is solvable if the number of inversions is even
    return inversions % 2 == 0
