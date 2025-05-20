#  Agent must collect 1 package located somewhere in the environment
#  Agent starts at a random position in the environment
#  Agent can move in 4 directions: UP, DOWN, LEFT, RIGHT

# Maphuti Shilabje

from FourRooms import FourRooms
import numpy as np

GRID_COLS = 11
GRID_ROWS = 11


ACTIONS = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]
NUM_ACTIONS = len(ACTIONS)


# The learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
NUM_EPOCHS = 1000
STEPS_PER_EPOCH = 250


q_table = np.zeros((GRID_ROWS, GRID_COLS, NUM_ACTIONS)) # Initialize Q-table

def get_state_index(pos, packages_remaining):
    """
    Convert the agent's position and packages remaining to a state index.
    """
    x = pos[0] - 1    
    y = pos[1] - 1

    # Ensure the position is within the grid bounds

    x = np.clip(x, 0, GRID_ROWS - 1) # Ensure x is within bounds
    y = np.clip(y, 0, GRID_COLS - 1) # Ensure y is within bounds

    packages = packages_remaining 


    return (pos[0], pos[1], packages)