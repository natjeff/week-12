import numpy as np
from IPython.display import clear_output
import time
import seaborn as sns
import matplotlib.pyplot as plt


def update_board(current_board):
    """
    Executes one step of Conway's Game of Life on a binary NumPy array.
    Returns updated board after applying Game of Life rules.
    """
    # Pad edges to get counted correctly
    padded = np.pad(current_board, pad_width=1, mode='constant', constant_values=0)
    neighbors = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +     # upper row
        padded[1:-1, :-2] +                    padded[1:-1, 2:] +    # mid row (left, right)
        padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]          # bottom row
    )

    # Apply rules and update board
    birth = (neighbors == 3) & (current_board == 0)
    survive = ((neighbors == 2) | (neighbors == 3)) & (current_board == 1)
    updated_board = np.zeros_like(current_board)
    updated_board[birth | survive] = 1

    return updated_board


def show_game(game_board, n_steps=10, pause=0.5):
    """
    Show `n_steps` of Conway's Game of Life, given the `update_board` function.

    Parameters
    ----------
    game_board : numpy.ndarray
        A binary array representing the initial starting conditions for Conway's Game of Life. In this array, ` represents a "living" cell and 0 represents a "dead" cell.
    n_steps : int, optional
        Number of game steps to run through, by default 10
    pause : float, optional
        Number of seconds to wait between steps, by default 0.5
    """
    for step in range(n_steps):
        clear_output(wait=True)

        # update board
        game_board = update_board(game_board)

        # show board
        sns.heatmap(game_board, cmap='plasma', cbar=False, square=True)
        plt.title(f'Board State at Step {step + 1}')
        plt.show()

        # wait for the next step
        if step + 1 < n_steps:
            time.sleep(pause)