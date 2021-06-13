'''

Author: Daman Arora, 13th June, 2021

Implementation as mentioned at: http://www.incompleteideas.net/book/ebook/node53.html


'''

from blackjack import BlackjackEnv
import random
import numpy as np


class ControlAgent:
    def __init__(self, eps=1e-2):
        
        self.Q = {}
        self.counts = {}
        self.eps = eps
        self.logs = {}

        for i in range(32):
            for j in range(11):
                for k in range(2):
                    self.Q[(i, j, k)] = np.random.randn(2)
                    self.counts[(i, j, k)] = [1, 1]
                    
    def policy(self, state, train=False):
        
        if train:
            if np.random.rand() < self.eps:
                return np.random.randint(2)
            else:
                return self.Q[state].argmax()
        
        else:
            return self.Q[state].argmax()
            
    
    def update(self, trajectory, actions, rewards):
        
        assert(len(trajectory) == len(actions) + 1 and len(actions) == len(rewards))
        
        num_steps = len(actions)
        rewards = np.array(rewards)
        rewards = rewards * np.logspace(0, num_steps - 1, num_steps, base=0.99)
        cumulative_rewards = np.cumsum(rewards[::-1])[::-1]
        
        visited = set()
    
        for j, action in enumerate(actions):
            
            state = trajectory[j]
            G = cumulative_rewards[j]
            
            if not state in visited:
                visited.add(state)
                n = self.counts[state][action]
                self.Q[state][action] = self.Q[state][action] * (n / (n + 1)) + G * (1 / (n + 1))   
                self.counts[state][action] += 1
                
    def evaluate(self, num_rollouts):
        wins, draws, losses = 0, 0, 0
        for i in range(num_rollouts):
            state = env.reset()
            done = False

            while not done:
                action = agent.policy(state, train=False)
                state, reward, done, _ = env.step(action)
            
            if reward == 1.0:
                wins += 1
            elif reward == 0.0:
                draws += 1
            else:
                losses += 1
        
        return wins / num_rollouts, draws / num_rollouts, losses / num_rollouts
            

def train(env, agent, num_rollouts):
    
    for i in range(num_rollouts):
        state = env.reset()
        trajectory, actions, rewards = [state], [], []

        done = False

        while not done:
            action = agent.policy(state, train=True)
            state, reward, done, _ = env.step(action)

            actions.append(action)
            rewards.append(reward)
            trajectory.append(state)

        agent.update(trajectory, actions, rewards)

if __name__ == '__main__':
    env = BlackjackEnv()
    agent = ControlAgent(eps=2e-2)
    train(env=env, agent=agent, num_rollouts=100000)
    
    win_rate, draw_rate, loss_rate = agent.evaluate(num_rollouts=1000)
    
    print(win_rate, draw_rate, loss_rate)