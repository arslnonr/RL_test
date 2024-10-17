import gym
import numpy as np
import random
import matplotlib.pyplot as plt


class GridEnv(gym.Env):
    def __init__(self, square_grid_size):
        super(GridEnv, self).__init__()
        self.size = square_grid_size
        self.action_space = gym.spaces.Discrete(4)  # 4 hareket seçeneği
        self.observation_space = gym.spaces.Box(low=0, high=square_grid_size - 1, shape=(2,), dtype=np.int32)

    def reset(self):
        self.state = (0, 0)
        return np.array(self.state)

    def step(self, action):
        # Actions: 0 - up, 1 - down, 2 - left, 3 - right
        if action == 0 and self.state[0] > 0:
            self.state = (self.state[0] - 1, self.state[1])
        if action == 1 and self.state[0] < self.size - 1:
            self.state = (self.state[0] + 1, self.state[1])
        if action == 2 and self.state[1] > 0:
            self.state = (self.state[0], self.state[1] - 1)
        if action == 3 and self.state[1] < self.size - 1:
            self.state = (self.state[0], self.state[1] + 1)

        if self.state == (self.size - 1, self.size - 1):
            reward = 1000
            done = True
        elif self.state == (self.size -1, 0 ):
            reward = -1000
            done = True

        elif self.state == (self.size -3, self.size -3):
            reward = -1000
            done = False
        elif self.state ==  (self.size - 3,self.size - 2):
            reward = -1000
            done = False
        else:
            reward = -1
            done = False

        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        grid = np.zeros((self.size, self.size))
        grid[self.state] = 1  # Ajanın bulunduğu konumu göster
        #print(grid)  # Konsol çıktısı için
        plt.imshow(grid, cmap='Greys', vmin=0, vmax=1)
        plt.xticks(np.arange(-0.5, self.size, 1), [])
        plt.yticks(np.arange(-0.5, self.size, 1), [])
        plt.grid()
        plt.show()


# Parameters
grid_size = 5
env = GridEnv(grid_size)

# Q-table
Q_table = np.zeros((grid_size, grid_size, 4))

# Hyperparameters
alpha = 0.05
gamma = 0.95
epsilon = 0.1
num_episodes = 5000


# Action selection using epsilon-greedy
def epsilon_greedy_policy(state, Q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1, 2, 3])
    else:
        state_x, state_y = state
        return np.argmax(Q_table[state_x, state_y])


# Q-learning algorithm
def q_learning():
    global epsilon
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy_policy(state, Q_table, epsilon)
            next_state, reward, done, _ = env.step(action)

            state_x, state_y = state
            next_state_x, next_state_y = next_state

            best_next_action = np.argmax(Q_table[next_state_x, next_state_y])
            Q_table[state_x, state_y, action] = Q_table[state_x, state_y, action] + alpha * (
                    reward + gamma * Q_table[next_state_x, next_state_y, best_next_action] - Q_table[
                state_x, state_y, action]
            )

            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {episode + 1}, Total Reward: {total_reward}")




# Run the Q-learning algorithm
q_learning()


# Testing the learned policy with rendering
def test_policy():
    state = env.reset()
    done = False
    steps = 0
    while not done:
        env.render()  # Ortamı görselleştir
        action = np.argmax(Q_table[state[0], state[1]])
        next_state, _, done, _ = env.step(action)
        state = next_state
        steps += 1
    print(f"Reached goal in {steps} steps.")


# Test the learned policy after training
test_policy()