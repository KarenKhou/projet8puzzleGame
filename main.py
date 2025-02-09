from eightpuzzlecomplete import *
import sys
import numpy as np

def main():
    
    initial_state = None  # Initialize the initial state variable

    while True:
        if initial_state is None:  # First-time input for initial state
            initial_state_input = input("Enter the initial state (e.g., '1 2 3 4 5 6 7 8 0'): ")
        else:  # Existing state; provide an option to change it
            print("\nCurrent initial state:")
            print(initial_state)
            initial_state_input = input("Enter a new initial state or type 'keep' to keep the current state: ")

        # Convert the input string into a NumPy array
        if initial_state_input.lower() == 'keep':
            # Keep the current state
            pass
        else:
            try:
                initial_state_list = list(map(int, initial_state_input.split()))

                # Check if the input has exactly 9 numbers
                if len(initial_state_list) != 9:
                    raise ValueError("Invalid input: must provide exactly 9 numbers.")

                # Check if all numbers are between 0 and 8 and if there are duplicates
                if len(set(initial_state_list)) != 9 or any(num < 0 or num > 8 for num in initial_state_list):
                    raise ValueError("Invalid input: numbers must be unique and between 0 and 8.")

                initial_state = np.array(initial_state_list).reshape(3, 3)
                
            except ValueError as e:
                print(f"Error: {e}. Please provide a valid initial state.")
                continue  # Re-prompt for valid input

        # Check if the initial state is solvable
        if not is_solvable(initial_state):
            print("The initial state is not solvable.")
            continue  # Re-prompt for a valid state

        goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])  # Define the goal state

        # Check if the initial state is already the goal state
        if np.array_equal(initial_state, goal_state):
            print("The initial state is the goal state! Please enter a new initial state.")
            initial_state = None  # Reset initial state to prompt for new input
            continue  # Restart the input loop

        # Always reset the puzzle to the initial state before each run
        puzzle = EightPuzzle(initial_state.copy())

        while True:  # Infinite loop to keep asking for choices
            # Ask the user to choose the algorithm or exit
            print("\nChoose an option:")
            print("1. BFS")
            print("2. A* (Manhattan Distance Heuristic)")
            print("3. A* (Linear Conflict Heuristic)")
            print("4. A* (Misplaced Tiles Heuristic)")
            print("5. A* (Uniform Cost Search - Heuristic = 0)")
            print("6. Change the initial state")
            print("7. Exit")  # Exit option

            # Input validation
            choice = input("Please select an option (1-7): ").strip()

            # Check if input is a valid choice (must be between 1 and 7)
            if choice not in ['1', '2', '3', '4', '5', '6', '7']:
                print("The number is unavailable. Please choose a valid option between 1 and 7.")
                continue  # Re-prompt for valid input

            if choice == '6':
                break  # Go back to the initial state input loop
            elif choice == '7':
                print("Exiting the program.")
                sys.exit()  # Exit the program

            # Initialize variables for solution_moves and explored_nodes
            solution_moves = []
            explored_nodes = 0

            # Handle the choice
            if choice == '1':
                print("You chose BFS.")
                solution_moves, explored_nodes = bfs_solve(puzzle)
            elif choice == '2':
                print("You chose A* with Manhattan Distance Heuristic.")
                solution_moves, explored_nodes = a_star_solve(puzzle)
            elif choice == '3':
                print("You chose A* with Linear Conflict Heuristic.")
                solution_moves, explored_nodes = a_star_linear_conflict(puzzle)
            elif choice == '4':
                print("You chose A* with Misplaced Tiles Heuristic.")
                solution_moves, explored_nodes = a_star_misplaced_tiles(puzzle)
            elif choice == '5':
                print("You chose A* with Uniform Cost Search (Heuristic = 0).")
                solution_moves, explored_nodes = a_star_uniform_cost(puzzle)

            # If a solution is found
            if solution_moves:
                print(f"Number of explored nodes: {explored_nodes}")
                print(f"Solution moves: {solution_moves}")
                print_solution_path(initial_state, solution_moves)

                # Display the solution process step by step
                manual_animation_with_button(initial_state, solution_moves)
            else:
                print("No solution found.")

if __name__ == "__main__":
    main()
