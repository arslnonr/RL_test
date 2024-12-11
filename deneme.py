import random
import numpy as np   # CALISIYO , SAKIN " BUNLARIN İCİNDE BOSLUK UNUTMA BİDA"
import matplotlib.pyplot as plt

# Define environment ( grid env kullanılacak )


grid = [['S', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', '', ' ', ' ', ' ', ' '],
        [' ', ' ', 'X', 'X', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' '],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', '', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ''],
        ['', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', '', ' ', ' ', ' ', ' '],
        [' ', ' ', 'X', 'X', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', '', ' ', ' ','X', ' ', ' '],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', '', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ''],
        ['', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', '', ' ', ' ', ' ', ' '],
        [' ', ' ', 'X', 'X', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' '],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', '', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ''],
        ['', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', '', ' ', ' ', ' ', ' '],
        [' ', ' ', 'X', 'X', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' '],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', '', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ''],
['', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', '', ' ', ' ', ' ', ' '],
        [' ', ' ', 'X', 'X', 'X', ' ', ' ', 'X', ' ', 'X', 'X', 'X', ' ', ' ', '', ' ', ' ', 'X', ' ', ' '],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', '', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ''],
        ['', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', '', ' ', ' ', ' ', ' '],
        [' ', ' ', 'X', 'X', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', '', ' ', ' ','X', ' ', ' '],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', '', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ''],
        ['', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', '', ' ', ' ', ' ', ' '],
        [' ', ' ', 'X', 'X', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' '],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', '', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ''],
        ['', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', '', ' ', 'X', ' ', ' '],
        [' ', ' ', 'X', 'X', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' '],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', '', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ''],
['', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', '', ' ', ' ', ' ', ' '],
        [' ', ' ', 'X', 'X', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' '],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', '', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ''],
        ['', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', '', ' ', ' ', ' ', ' '],
        [' ', ' ', 'X', 'X', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', '', ' ', ' ','X', ' ', ' '],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', '', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ''],
        ['', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', '', ' ', ' ', ' ', ' '],
        [' ', ' ', 'X', 'X', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' '],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', '', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', 'X', 'X', 'X', ''],
        ['', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', '', ' ', ' ', ' ', ' '],
        [' ', ' ', 'X', 'X', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' '],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', '', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', 'X'],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ''],
['', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', '', ' ', ' ', ' ', ' '],
        [' ', ' ', 'X', 'X', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', '', ' ', ' ', 'X', ' ', ' '],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'X', 'X', ' ', '', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', 'X', ' ', 'X'],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ''],
        ['', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'X', '', ' ', ' ', ' ', ' '],
        [' ', ' ', 'X', 'X', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', '', ' ', ' ','X', ' ', ' '],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', 'X', ' ', ' ', ' ', ' ', 'X', ' ', '', ' ', 'X', 'X', 'X'],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', 'X', ' ', ' ', ' ', 'X', ' ', 'X', 'X', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ''],
        ['', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', '', ' ', ' ', ' ', ' '],
        [' ', ' ', 'X', 'X', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' '],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', '', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ''],
        [' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ''],
        [' ', ' ', ' ', ' ', '', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', 'X', 'G']
        ]

rows = len(grid)
cols = len(grid[0])
actions = ["up","down","left","right"]


#Q - LEARNİNG PARAMS
alpha = 0.2
gamma = 0.95
epsilon = 0.9
episodes = 5000
epsilon_decrease = 0.001

Q_table = np.zeros((rows * cols,4))


def get_state(row, col):
    return row * cols + col


def get_next_action(state):
    if random.uniform(0,1)<epsilon:
        return random.choice(actions)
    else:
        return actions[np.argmax(Q_table[state, :])]

def get_next_location(row,col,action):
    new_row, new_col = row , col
    if action == 'up' and row > 0:
        new_row -= 1
    if action == 'down' and row < rows -1:
        new_row += 1
    if action == 'left' and col > 0:
        new_col -=1
    if action == 'right' and col < cols -1:
        new_col +=1

    return new_row,new_col

def get_reward(row,col):
    if grid[row][col] == 'G':
        return 1000
    elif grid[row][col] == 'X':
        return -1000

    else:
        return -1

# training loop
for episode in range(episodes):
    row , col = 0 , 0 # starts at top left
    total_reward = 0
    while grid[row][col] != 'G':
        state = get_state(row,col)
        action = get_next_action(state)
        next_row, next_col = get_next_location(row,col,action)
        reward = get_reward(next_row,next_col)
        next_state = get_state(next_row,next_col)

        # q table update burayı anla mutlaka , yuakrıyı da
        old_value = Q_table[state,actions.index(action)]
        next_max = np.max(Q_table[next_state, :])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        Q_table[state, actions.index(action)] = new_value
        total_reward += reward
        row, col = next_row , next_col
    epsilon = epsilon - 0.0003
    print("episode : ", episode+1 , "reward: ", total_reward)





# use the trained q table
row, col = 0, 0  # Start at the top-left corner
steps = 0
while grid[row][col] != 'G':
    state = get_state(row, col)
    action = actions[np.argmax(Q_table[state, :])]  # Exploit the learned policy (no exploration)
    next_row, next_col = get_next_location(row, col, action)

    print(f"Step {steps}: From ({row}, {col}) to ({next_row}, {next_col}) taking action {action}")

    row, col = next_row, next_col
    steps += 1

print(f"\nReached the goal in {steps} steps!")




