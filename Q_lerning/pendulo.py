import numpy as np
import math
import matplotlib.pyplot as plt
import random

class SimplePendulumEnv:
    def __init__(self):
        # Constantes del entorno
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length = 1.5
        self.dt = 0.02
        
        self.max_theta = math.pi / 4  # Límite de 45 grados
        self.max_theta_dot = 10  # Máxima velocidad angular

        # Espacios discretos para el estado y las acciones
        self.theta_bins = np.linspace(-self.max_theta, self.max_theta, 5)  # 10 divisiones del ángulo
        self.theta_dot_bins = np.linspace(-self.max_theta_dot, self.max_theta_dot, 5)  # 10 divisiones de la velocidad angular
        
        # Acciones discretas: Fuerzas aplicadas
        self.actions = [-5, -2, 0, 2, 5]

        self.reset()

    def reset(self):
        self.theta = np.random.uniform(-0.05, 0.05)
        self.theta_dot = 0.0
        self.t = 0
        return self._discretize_state()

    def step(self, action):
        force = np.clip(self.actions[action], -10.0, 10.0)

        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        total_mass = self.mass_cart + self.mass_pole
        pole_mass_length = self.mass_pole * self.length

        theta_double_dot = (self.gravity * sin_theta - cos_theta * (
            force + pole_mass_length * self.theta_dot ** 2 * sin_theta)) / (
                self.length * (4/3 - self.mass_pole * cos_theta ** 2 / total_mass))

        self.theta_dot += theta_double_dot * self.dt
        self.theta += self.theta_dot * self.dt

        self.theta = np.clip(self.theta, -self.max_theta, self.max_theta)
        self.theta_dot = np.clip(self.theta_dot, -self.max_theta_dot, self.max_theta_dot)

        reward = 1.0 if -self.max_theta < self.theta < self.max_theta else -10.0
        done = bool(self.theta <= -self.max_theta or self.theta >= self.max_theta)

        self.t += 1
        return self._discretize_state(), reward, done, {}

    def _discretize_state(self):
        """Discretiza el estado continuo en celdas."""
        theta_idx = np.digitize(self.theta, self.theta_bins) - 1
        theta_dot_idx = np.digitize(self.theta_dot, self.theta_dot_bins) - 1
        return theta_idx, theta_dot_idx

    def render(self):
        """Dibuja el entorno para visualizarlo."""
        plt.clf()
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        # Posición del péndulo
        x = self.length * math.sin(self.theta)
        y = self.length * math.cos(self.theta)  # Cambiar a positivo para que esté 'hacia arriba'

        # Dibujar el carro
        plt.plot([-0.5, 0.5], [0, 0], 'k-', lw=5)  # Base del carro
        # Dibujar el péndulo
        plt.plot([0, x], [0, y], 'r-', lw=2)  # Brazo del péndulo
        plt.plot(x, y, 'bo', markersize=10)  # Masa del péndulo

        plt.pause(0.01)


    def close(self):
        plt.close()

# Agente de Q-learning
class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.q_table = np.zeros(state_size + (action_size,))
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Probabilidad de exploración
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_size = action_size

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Explorar
        return np.argmax(self.q_table[state])  # Explotar

    def learn(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state + (best_next_action,)] * (1 - done)
        td_error = td_target - self.q_table[state + (action,)]
        self.q_table[state + (action,)] += self.alpha * td_error
        
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Entrenamiento del agente
env = SimplePendulumEnv()
agent = QLearningAgent(state_size=(5, 5), action_size=len(env.actions))

episodes = 1000

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    
    for step in range(200):
        #env.render()
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            print(f"Episodio {episode+1}, Recompensa total: {total_reward}")
            break

# Test
for step in range(200):
        env.render()
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            print(f"Episodio {episode+1}, Recompensa total: {total_reward}")
            break

env.close()
