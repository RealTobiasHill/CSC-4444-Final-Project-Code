# CSC-4444-Final-Project-Code
This is the code for my final project for csc 4444

SUDOKU SOLVER

A Python implementation of a Sudoku Puzzle Solver using backtracking combined with powerful heuristics:
- MRV (Minimum Remaining Values)
- Degree Heuristic
- LCV (Least Constraining Value)

The solver can solve Sudoku puzzles efficiently and provides tools for:
- Visualizing Sudoku Boards
- Plotting performance statistics
- Visualizing search trees
- Explaining heuristics like MRV

Features:
- Backtracking-based solver enhanced with MRV, Degree, and LCV heuristics for faster solving.
- Performance tracking: number of assignments, backtracks, and time taken.
- Sudoku board visualization using Matplotlib.
- Performance graphs: time vs difficulty, backtracks vs difficulty.
- Illustrations: search tree diagrams and heuristic behaviors.

Requirements:
Make sure you have the following Python packages installed:
pip install numpy matplotlib
- Python 3.7 + recommended
Packages:
- numpy
- matplotlib

How to Run
1. Clone or Download the Script
- Save the code as sudoku_solver.py.
2. Run from the command line
- python sudoku_solver.py
This will:
- Load a sample Sudoku puzzle
- Solve it
- Print the original and solved board the teminal
- Display solve time, assignments made, and backtracks
You will also see various plots (Sudoku board visualization, performance graphs) pop up as windows using Matplotlib.

Other ways to run:
1. Download the file and open it on vscode and click the run button
2. Download the file and open it on Spyder and click the run button

Example Usage:

# Inside sudoku_solver.py

if __name__ == "__main__":
    main()
    
- This will automatically run a pre-defined Sudoku board.
- You can replace the example_board inside main() with any other Sudoku board.

Example Sudoku Puzzle That is Used:

5 3 _ | _ 7 _ | _ _ _
6 _ _ | 1 9 5 | _ _ _
_ 9 8 | _ _ _ | _ 6 _
------+-------+------
8 _ _ | _ 6 _ | _ _ 3
4 _ _ | 8 _ 3 | _ _ 1
7 _ _ | _ 2 _ | _ _ 6
------+-------+------
_ 6 _ | _ _ _ | 2 8 _
_ _ _ | 4 1 9 | _ _ 5
_ _ _ | _ 8 _ | _ 7 9

Functions that were used:
SudokuSolver.solve() - Solves the puzzle using backtracking + heuristics
SudokuSolver.print_board() - Prints the current Sudoku board
plot_sudoku(board, title) -	Plots a Sudoku board nicely
plot_performance(algorithms, solve_times, backtracks) -	Shows solve times and backtracks comparisons
plot_difficulty_stats(difficulties, solve_times, backtracks, assignments) -	Plots performance vs puzzle difficulty
plot_search_trees() -	Visualizes search trees with/without heuristics
plot_mrv_example() -	Demonstrates MRV heuristic selection visually

Customizing Your Own Puzzle:
- You can edit the example_board inside main() to any custom 9x9 board by replacing 0s where you want empty cells.
- 0s represent an empty cell
