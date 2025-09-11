import gym
from env_2D import Environment2D
import random as random

env = Environment2D(width=10, height=10, obstacle_percentage=0.1)
observation = env.reset()
for _ in range(100):
    env.render()
    action = env.get_valid_actions()
    observation, reward, done = env.step(random.choice(action))
    
    if done:
        observation = env.reset()
        
