# sudoku_solver.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

class SudokuSolver:
    def __init__(self, board=None):
        # Initialize with a board or an empty one
        if board:
            self.board = [row[:] for row in board]  # Deep copy
        else:
            # Empty 9x9 board
            self.board = [[0 for _ in range(9)] for _ in range(9)]
        
        # Statistics for performance analysis
        self.backtracks = 0
        self.assignments = 0
    
    def solve(self):
        """Main entry point to solve the board"""
        return self._backtrack()
    
    def _backtrack(self):
        """Main backtracking algorithm with heuristics"""
        # Check if board is complete
        if self._is_complete():
            return True
        
        # Select unassigned variable using MRV and Degree heuristic
        row, col, legal_values = self._select_unassigned_variable()
        
        # If no legal values for this cell, this branch fails
        if not legal_values:
            self.backtracks += 1
            return False
        
        # Order values using LCV heuristic
        ordered_values = self._order_domain_values(row, col, legal_values)
        
        # Try each value in the ordered list
        for value in ordered_values:
            # If value is consistent with constraints
            if self._is_valid(row, col, value):
                # Make assignment
                self.board[row][col] = value
                self.assignments += 1
                
                # Recursively try to solve the rest
                if self._backtrack():
                    return True
                
                # If we get here, we need to backtrack
                self.board[row][col] = 0
                self.backtracks += 1
        
        # No solution found in this branch
        return False
    
    def _is_complete(self):
        """Check if the board is completely filled"""
        for row in range(9):
            for col in range(9):
                if self.board[row][col] == 0:
                    return False
        return True
    
    def _get_legal_values(self, row, col):
        """Return a set of legal values for a cell"""
        if self.board[row][col] != 0:  # If cell already filled
            return set()
        
        # Start with all possible values 1-9
        values = set(range(1, 10))
        
        # Remove values that appear in the same row
        for c in range(9):
            if self.board[row][c] != 0:
                values.discard(self.board[row][c])
        
        # Remove values that appear in the same column
        for r in range(9):
            if self.board[r][col] != 0:
                values.discard(self.board[r][col])
        
        # Remove values that appear in the same 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if self.board[r][c] != 0:
                    values.discard(self.board[r][c])
        
        return values
    
    def _get_degree(self, row, col):
        """
        Count how many unfilled cells share a constraint with this cell.
        Used for tie-breaking in the MRV heuristic.
        """
        if self.board[row][col] != 0:  # If cell is already filled
            return 0
        
        degree = 0
        seen = set()  # To avoid double-counting cells
        
        # Count unfilled cells in the same row
        for c in range(9):
            if c != col and self.board[row][c] == 0:
                seen.add((row, c))
                degree += 1
        
        # Count unfilled cells in the same column
        for r in range(9):
            if r != row and self.board[r][col] == 0:
                if (r, col) not in seen:
                    seen.add((r, col))
                    degree += 1
        
        # Count unfilled cells in the same 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if (r != row or c != col) and self.board[r][c] == 0:
                    if (r, c) not in seen:
                        seen.add((r, c))
                        degree += 1
        
        return degree
    
    def _select_unassigned_variable(self):
        """
        Use MRV heuristic to select the cell to fill next.
        Return (row, col, legal_values) of the selected cell.
        """
        min_remaining = 10  # More than maximum possible (9)
        selected_cell = None
        selected_values = set()
        
        # Loop through all cells
        for row in range(9):
            for col in range(9):
                if self.board[row][col] == 0:  # Empty cell
                    legal_values = self._get_legal_values(row, col)
                    num_values = len(legal_values)
                    
                    # If we find a cell with no legal values, the puzzle is unsolvable
                    if num_values == 0:
                        return None, None, set()
                    
                    # If we find a cell with fewer remaining values
                    if num_values < min_remaining:
                        min_remaining = num_values
                        selected_cell = (row, col)
                        selected_values = legal_values
                        
                        # Optimization: If we find a cell with only one possibility, return immediately
                        if min_remaining == 1:
                            return row, col, selected_values
                    
                    # If there's a tie, use degree heuristic as tie-breaker
                    elif num_values == min_remaining and selected_cell:
                        current_degree = self._get_degree(row, col)
                        selected_degree = self._get_degree(selected_cell[0], selected_cell[1])
                        if current_degree > selected_degree:
                            selected_cell = (row, col)
                            selected_values = legal_values
        
        if selected_cell:
            return selected_cell[0], selected_cell[1], selected_values
        return None, None, set()  # No empty cells left
    
    def _order_domain_values(self, row, col, legal_values):
        """
        Order values based on LCV heuristic.
        Return values sorted from least constraining to most constraining.
        """
        # Dictionary to store how constraining each value is
        value_constraints = {}
        
        # For each possible value
        for value in legal_values:
            # Temporarily assign this value
            self.board[row][col] = value
            constraints = 0
            
            # Check impact on related cells
            # We'll count how many values would be eliminated from other cells
            
            # Check cells in same row
            for c in range(9):
                if c != col and self.board[row][c] == 0:
                    # If this value would eliminate an option
                    if not self._is_valid(row, c, value):
                        constraints += 1
            
            # Check cells in same column
            for r in range(9):
                if r != row and self.board[r][col] == 0:
                    if not self._is_valid(r, col, value):
                        constraints += 1
            
            # Check cells in same box
            box_row, box_col = 3 * (row // 3), 3 * (col // 3)
            for r in range(box_row, box_row + 3):
                for c in range(box_col, box_col + 3):
                    if (r != row or c != col) and self.board[r][c] == 0:
                        if not self._is_valid(r, c, value):
                            constraints += 1
            
            # Undo the assignment
            self.board[row][col] = 0
            
            # Store the constraint count
            value_constraints[value] = constraints
        
        # Sort values by constraint count (ascending)
        return sorted(legal_values, key=lambda x: value_constraints[x])
    
    def _is_valid(self, row, col, value):
        """Check if placing value at (row, col) violates any constraint"""
        # Check row
        for c in range(9):
            if self.board[row][c] == value:
                return False
        
        # Check column
        for r in range(9):
            if self.board[r][col] == value:
                return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if self.board[r][c] == value:
                    return False
        
        return True
    
    def print_board(self):
        """Print the current state of the board"""
        horizontal_line = "+-------+-------+-------+"
        print(horizontal_line)
        for i in range(9):
            row_str = "| "
            for j in range(9):
                if j % 3 == 2:
                    row_str += str(self.board[i][j] if self.board[i][j] != 0 else " ") + " | "
                else:
                    row_str += str(self.board[i][j] if self.board[i][j] != 0 else " ") + " "
            print(row_str)
            if i % 3 == 2:
                print(horizontal_line)
    
    def get_stats(self):
        """Return performance statistics"""
        return {
            "assignments": self.assignments,
            "backtracks": self.backtracks
        }


# Example usage
def main():
    import time
    
    # Example puzzle (0 represents empty cells)
    # This is a medium difficulty puzzle
    example_board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    
    # Create solver instance
    solver = SudokuSolver(example_board)
    
    # Print original board
    print("Original Puzzle:")
    solver.print_board()
    
    # Solve and time it
    start_time = time.time()
    solved = solver.solve()
    end_time = time.time()
    
    # Print results
    if solved:
        print("\nSolved Puzzle:")
        solver.print_board()
        
        # Print statistics
        stats = solver.get_stats()
        print(f"\nStatistics:")
        print(f"Time taken: {end_time - start_time:.3f} seconds")
        print(f"Assignments made: {stats['assignments']}")
        print(f"Backtracks: {stats['backtracks']}")
    else:
        print("\nNo solution exists for this puzzle!")

def plot_sudoku(board, title="Sudoku Board"):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create a custom colormap (light blue for original numbers, white for filled)
    cmap = ListedColormap(['white', 'lightblue'])

    # Create a mask for the original numbers (assuming they're marked somehow)
    # For this example, let's assume non-zero values in the initial board are original
    original_mask = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                original_mask[i][j] = 1

    # Plot the grid with colors
    ax.imshow(original_mask, cmap=cmap, alpha=0.3)

    # Add the numbers
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                ax.text(j, i, str(board[i][j]), ha='center', va='center', fontsize=16, 
                        color='black' if original_mask[i][j] == 0 else 'blue')

    # Grid lines
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(i - 0.5, color='black', linewidth=lw)
        ax.axvline(i - 0.5, color='black', linewidth=lw)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add title
    ax.set_title(title, fontsize=16)

    return fig

# Example usage:
# Create a sample Sudoku board (you can replace it with your board)
board = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
])

fig = plot_sudoku(board, title="Sudoku Puzzle")
plt.show()

def plot_performance(algorithms, solve_times, backtracks):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot solve times
    bars1 = ax1.bar(algorithms, solve_times, color=['#E74C3C', '#F39C12', '#3498DB', '#2ECC71'])
    ax1.set_title('Average Solving Time (Hard Puzzles)', fontsize=14)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_yscale('log')  # Log scale to show the dramatic difference

    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}s', ha='center', va='bottom', fontsize=10)

    # Plot backtracks
    bars2 = ax2.bar(algorithms, backtracks, color=['#E74C3C', '#F39C12', '#3498DB', '#2ECC71'])
    ax2.set_title('Average Number of Backtracks', fontsize=14)
    ax2.set_ylabel('Backtracks (log scale)', fontsize=12)
    ax2.set_yscale('log')  # Log scale to show the dramatic difference

    # Add values on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig

# Example usage:
algorithms = ['Backtracking', 'Forward Checking', 'MRV + Forward', 'AC-3 + MRV + Forward']
solve_times = [120.5, 45.2, 15.8, 2.7]  # example times
backtracks = [15000, 5000, 1200, 50]     # example backtrack counts

fig = plot_performance(algorithms, solve_times, backtracks)
plt.show()

def plot_performance(algorithms, solve_times, backtracks):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot solve times
    bars1 = ax1.bar(algorithms, solve_times, color=['#E74C3C', '#F39C12', '#3498DB', '#2ECC71'])
    ax1.set_title('Average Solving Time (Hard Puzzles)', fontsize=14)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_yscale('log')  # Log scale to show the dramatic difference

    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}s', ha='center', va='bottom', fontsize=10)

    # Plot backtracks
    bars2 = ax2.bar(algorithms, backtracks, color=['#E74C3C', '#F39C12', '#3498DB', '#2ECC71'])
    ax2.set_title('Average Number of Backtracks', fontsize=14)
    ax2.set_ylabel('Backtracks (log scale)', fontsize=12)
    ax2.set_yscale('log')  # Log scale to show the dramatic difference

    # Add values on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig

# Example usage:
algorithms = ['Backtracking', 'Forward Checking', 'MRV + Forward', 'AC-3 + MRV + Forward']
solve_times = [120.5, 45.2, 15.8, 2.7]  # example times
backtracks = [15000, 5000, 1200, 50]     # example backtrack counts

fig = plot_performance(algorithms, solve_times, backtracks)
plt.show()

def plot_difficulty_stats(difficulties, solve_times, backtracks, assignments):
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Plot solve times
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(difficulties, solve_times, color=['#ABEBC6', '#82E0AA', '#2ECC71', '#1D8348'])
    ax1.set_title('Average Solving Time', fontsize=14)
    ax1.set_ylabel('Time (seconds)', fontsize=12)

    # Add values on bars
    for i, v in enumerate(solve_times):
        ax1.text(i, v + 0.005, f'{v:.3f}s', ha='center')

    # Plot backtracks
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(difficulties, backtracks, color=['#AED6F1', '#5DADE2', '#3498DB', '#1B4F72'])
    ax2.set_title('Average Backtracks', fontsize=14)
    ax2.set_ylabel('Number of backtracks', fontsize=12)

    # Add values on bars
    for i, v in enumerate(backtracks):
        ax2.text(i, v + 0.5, f'{v}', ha='center')

    # Plot assignments
    ax3 = fig.add_subplot(gs[1, :])
    ax3.bar(difficulties, assignments, color=['#D7BDE2', '#AF7AC5', '#8E44AD', '#4A235A'], width=0.5)
    ax3.set_title('Average Cell Assignments', fontsize=14)
    ax3.set_ylabel('Number of assignments', fontsize=12)

    # Add values on bars
    for i, v in enumerate(assignments):
        ax3.text(i, v + 1, f'{v}', ha='center')

    plt.tight_layout()
    return fig

# Example usage:
difficulties = ['Easy', 'Medium', 'Hard', 'Evil']
solve_times = [0.02, 0.08, 2.5, 25.0]    # example times
backtracks = [30, 200, 5000, 20000]       # example backtrack counts
assignments = [60, 300, 800, 1200]        # example assignment counts

fig = plot_difficulty_stats(difficulties, solve_times, backtracks, assignments)
plt.show()

def plot_search_trees():
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # --- First tree: Without heuristics (more branching) ---
    ax1.set_title('Search Tree Without Heuristics', fontsize=14)
    ax1.set_xlim(-1, 5)
    ax1.set_ylim(0, 10)
    ax1.axis('off')

    # Root
    ax1.plot(2, 9, 'ro', markersize=15)
    ax1.text(2, 9.3, "Start", ha='center')

    # Level 1
    for i, x in enumerate([0, 1, 2, 3, 4]):
        ax1.plot(x, 7, 'bo', markersize=10)
        ax1.plot([2, x], [9, 7], 'k-', alpha=0.7)

    # Level 2 - many branches
    for i in range(15):
        x = i * 0.3
        ax1.plot(x, 5, 'go', markersize=8)
        parent = i // 3
        ax1.plot([parent, x], [7, 5], 'k-', alpha=0.5)

    # Level 3 - even more branches
    for i in range(40):
        x = i * 0.1 + 0.3
        if i % 4 == 0:
            ax1.plot(x, 3, 'mo', markersize=6)
        else:
            ax1.plot(x, 3, 'mo', markersize=6, alpha=0.3)
        parent = (i // 3) * 0.3
        ax1.plot([parent, x], [5, 3], 'k-', alpha=0.3)

    # --- Second tree: With heuristics (focused branching) ---
    ax2.set_title('Search Tree With Heuristics', fontsize=14)
    ax2.set_xlim(-1, 5)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    # Root
    ax2.plot(2, 9, 'ro', markersize=15)
    ax2.text(2, 9.3, "Start", ha='center')

    # Level 1 - fewer branches due to MRV
    for i, x in enumerate([1.5, 2, 2.5]):
        ax2.plot(x, 7, 'bo', markersize=10)
        ax2.plot([2, x], [9, 7], 'k-', alpha=0.7)

    # Level 2 - even fewer branches
    for i, x in enumerate([1.7, 2, 2.3]):
        ax2.plot(x, 5, 'go', markersize=8)
        ax2.plot([2, x], [7, 5], 'k-', alpha=0.6)

    # Level 3 - mostly correct path
    for i, x in enumerate([1.9, 2, 2.1]):
        if i == 1:  # Correct path
            ax2.plot(x, 3, 'mo', markersize=6)
            ax2.plot([2, x], [5, 3], 'k-', alpha=0.9)
            # Continue correct path
            ax2.plot(2, 1, 'co', markersize=8)
            ax2.plot([2, 2], [3, 1], 'k-', linewidth=2)
            ax2.text(2, 0.7, "Solution", ha='center')
        else:
            ax2.plot(x, 3, 'mo', markersize=6, alpha=0.3)
            ax2.plot([2, x], [5, 3], 'k-', alpha=0.3)

    plt.tight_layout()
    return fig

# Example usage:
fig = plot_search_trees()
plt.show()

def plot_mrv_example():
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create a sample board with some filled cells
    board = np.zeros((9, 9))
    
    # Add some numbers
    filled_positions = [
        (0, 0, 5), (0, 4, 7), 
        (1, 0, 6), (1, 3, 1), (1, 4, 9), (1, 5, 5),
        (2, 1, 9), (2, 2, 8), (2, 7, 6),
        (3, 0, 8), (3, 4, 6), (3, 8, 3),
        (4, 0, 4), (4, 3, 8), (4, 5, 3), (4, 8, 1),
        (5, 0, 7), (5, 4, 2), (5, 8, 6),
        (6, 1, 6), (6, 6, 2), (6, 7, 8),
        (7, 3, 4), (7, 4, 1), (7, 5, 9), (7, 8, 5),
        (8, 4, 8), (8, 7, 7), (8, 8, 9)
    ]

    for r, c, v in filled_positions:
        board[r, c] = v

    # Plot the grid background
    ax.imshow(np.zeros((9, 9)), cmap='Greys', alpha=0.1)

    # Add the numbers
    for i in range(9):
        for j in range(9):
            if board[i, j] != 0:
                ax.text(j, i, str(int(board[i, j])), ha='center', va='center', fontsize=16, color='blue')

    # Highlight a cell with few options (MRV would select)
    # Highlight (0, 1)
    rect = plt.Rectangle((1 - 0.5, 0 - 0.5), 1, 1, fill=True, color='yellow', alpha=0.3)
    ax.add_patch(rect)

    # Add legal values text for (0,1)
    ax.text(1, 0, "1,2", ha='center', va='center', fontsize=12, color='red')
    ax.text(1, -0.2, "MRV selects this cell\n(only 2 legal values)", ha='center', va='center', fontsize=11)

    # Grid lines
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(i - 0.5, color='black', linewidth=lw)
        ax.axvline(i - 0.5, color='black', linewidth=lw)

    # Highlight other example "legal value count" cells
    # Cell (2, 3)
    rect = plt.Rectangle((3 - 0.5, 2 - 0.5), 1, 1, fill=True, color='lightgreen', alpha=0.2)
    ax.add_patch(rect)
    ax.text(3, 2, "2,3,4,7", ha='center', va='center', fontsize=10, color='green')

    # Cell (6, 4)
    rect = plt.Rectangle((4 - 0.5, 6 - 0.5), 1, 1, fill=True, color='lightgreen', alpha=0.2)
    ax.add_patch(rect)
    ax.text(4, 6, "3,4,5,7,9", ha='center', va='center', fontsize=10, color='green')

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add title
    ax.set_title('Minimum Remaining Values (MRV) Heuristic', fontsize=16)

    plt.tight_layout()
    return fig

# Example usage:
fig = plot_mrv_example()
plt.show()

if __name__ == "__main__":
    main()
    
