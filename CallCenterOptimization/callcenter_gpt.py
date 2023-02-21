import numpy as np

class CallCenter:
    def __init__(self):
        self.num_csrs = 5
        self.exp_levels = np.array([1, 2, 3, 4, 5])
        self.reward_table = np.array([[5, 7, 10, 13, 16],
                                      [4, 6, 9, 12, 15],
                                      [3, 5, 8, 11, 14],
                                      [2, 4, 7, 10, 13],
                                      [1, 3, 6, 9, 12]])
        self.action_space = np.arange(self.num_csrs)
    
    def reset(self):
        # Reset the environment to a random state
        return np.random.choice(self.exp_levels)
    
    def step(self, state, action):
        # Assign the call to the CSR with the highest expected reward
        reward = self.reward_table[state, action]
        next_state = np.random.choice(self.exp_levels)
        done = True
        return next_state, reward, done

class QLearningAgent:
    def __init__(self, env, alpha=0.8, gamma=0.95):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((len(env.exp_levels), len(env.action_space)))
    
    def select_action(self, state, exploration_rate):
        # Select an action using an epsilon-greedy policy
        if np.random.rand() < exploration_rate:
            return np.random.choice(self.env.action_space)
        else:
            return np.argmax(self.q_table[state, :])
    
    def update_q_table(self, state, action, reward, next_state):
        # Update the Q-table using the Q-learning algorithm
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + \
            self.alpha * (reward + self.gamma * np.max(self.q_table[next_state, :]))
            
# Train the Q-learning agent
env = CallCenter()
agent = QLearningAgent(env)

total_episodes = 10000
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

for episode in range(total_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, exploration_rate)
        next_state, reward, done = env.step(state, action)
        agent.update_q_table(state, action, reward, next_state)
        state = next_state
    
    # Decay the exploration rate after each episode
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    
# Test the Q-learning agent
state = env.reset()
done = False
while not done:
    action = agent.select_action(state, 0)
    next_state, reward, done = env.step(state, action)
    state = next_state

print("The call was assigned to CSR #{} with a reward of {}.".format(state, reward))
