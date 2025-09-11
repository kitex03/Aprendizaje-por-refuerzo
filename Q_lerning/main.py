from agentesRL import *
from env_2D import *
from env_frozen import *
from env_maze import *
from env_multigoal import *
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def main():
    env = FrozenLakeEnvironment(10, 10,slippery_float=0.4)
    agente1 = Agent(env)
    # agente2 = Agent(env,epsilon= 0.3)
    # agente3 = Agent(env,epsilon=0.6)
    # agente4 = Agent(env,epsilon=0.9)
    env.render()

    rewards_agente1 = agente1.train_q_learning(30000)
    # rewards_agente2 = agente2.train_q_learning(200)
    # rewards_agente3 = agente3.train_q_learning(200)
    # rewards_agente4 = agente4.train_q_learning(200)

    agente1.test_agent(5)
    # agente2.test_agent(5)

    plt.clf()

    window_size = 20  # Tamaño de la ventana para el promedio móvil
    ma_rewards_agente1 = moving_average(rewards_agente1, window_size)
    # ma_rewards_agente2 = moving_average(rewards_agente2, window_size)
    # ma_rewards_agente3 = moving_average(rewards_agente3, window_size)
    # ma_rewards_agente4 = moving_average(rewards_agente4, window_size)
    
    # moving_average = np.convolve(rewards_agente1, np.ones((100))/100, mode='valid')
    # plt.plot(moving_average, label='Q-Learning MA(100)')
    # moving_average_2 = np.convolve(rewards_agente2, np.ones((100))/100, mode='valid')
    # plt.plot(moving_average_2, label='sarsa MA(100)')
    
    # plt.plot(ma_rewards_agente1, label='agente 1')
    # # plt.plot(ma_rewards_agente2, label='epsilon=0.3')
    # # plt.plot(ma_rewards_agente3, label='epsilon=0.6')
    # # plt.plot(ma_rewards_agente4, label='epsilon=0.9')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.title('Rewards per Episode (Moving Average)')
    # plt.legend()
    # plt.show()

main()