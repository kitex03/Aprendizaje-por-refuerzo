
import numpy as np
import random
import matplotlib.pyplot as plt

# Clase Agente
class Agent:
    def __init__(self, env, alpha=0.3, gamma=0.8, epsilon=0.2, render_training=False, pause_time=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.render_training = render_training  # Flag para renderizar el entrenamiento
        self.pause_time = pause_time  # Tiempo de pausa para el renderizado
        self.Q = np.zeros((env.height ,env.width , 4))  # Tabla Q (ancho, alto, acciones)
        self.max_actions_per_episode = env.width*env.height

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)  # Exploración
        else:
            return np.argmax(self.Q[state[0], state[1]])  # Explotación

    def train_q_learning(self, num_episodes):
        rewards_per_episode = []  # Lista para almacenar recompensas por episodio
        nactions = 0

        for episode in range(num_episodes):
            state = self.env.reset()  # Reiniciar el entorno
            done = False
            total_reward = 0  # Recompensa total para este episodio
            if episode%1000 == 0: print("Training episode: ", episode, nactions)
            nactions = 0    
            while not done and nactions < self.max_actions_per_episode:
                action = self.choose_action(state)  # Elegir acción
                next_state, reward, done = self.env.step(action)  # Realizar acción
                total_reward += reward  # Acumular recompensa
                # if done: print(" ... Done!")
                # Actualizar la tabla Q
                self.Q[state[0], state[1], action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[next_state[0], next_state[1]]) - self.Q[state[0], state[1], action]
                )
                state = next_state  # Avanzar al siguiente estado
                nactions+=1
                # Renderizar si el flag está activado
                if self.render_training:
                    self.env.render()  # Renderizar el entorno

            rewards_per_episode.append(total_reward)  # Almacenar recompensa total del episodio

        return rewards_per_episode  # Devolver las recompensas por episodio

    def train_sarsa(self, num_episodes):
        rewards_per_episode = []  # Lista para almacenar recompensas por episodio
        nactions = 0

        for episode in range(num_episodes):
            state = self.env.reset()  # Reiniciar el entorno
            action = self.choose_action(state)  # Elegir acción
            done = False
            total_reward = 0  # Recompensa total para este episodio
            if episode%1000 == 0: print("Training episode: ", episode, nactions)
            nactions = 0

            while not done and nactions < self.max_actions_per_episode:
                next_state, reward, done = self.env.step(action)  # Realizar acción
                total_reward += reward  # Acumular recompensa
                next_action = self.choose_action(next_state)  # Elegir la siguiente acción
                # Actualizar la tabla Q
                self.Q[state[0], state[1], action] += self.alpha * (
                    reward + self.gamma * self.Q[next_state[0], next_state[1], next_action] - self.Q[state[0], state[1], action]
                )
                state, action = next_state, next_action  # Avanzar al siguiente estado y acción
                nactions+=1

                # Renderizar si el flag está activado
                if self.render_training:
                    self.env.render()  # Renderizar el entorno

            rewards_per_episode.append(total_reward)  # Almacenar recompensa total del episodio

        return rewards_per_episode  # Devolver las recompensas por episodio

    def test_agent(self, num_tests, max_steps_per_test=100):
        """Ejecuta pruebas del agente después de haber aprendido."""
        for test in range(num_tests):
            state = self.env.reset()  # Reiniciar el entorno para cada prueba
            done = False
            step_count = 0  # Contador de pasos
            print(f"Prueba {test + 1}:")
            self.env.render()  # Mostrar el entorno antes de la prueba

            while not done and step_count < max_steps_per_test:
                action = self.choose_action(state)  # Elegir acción basada en Q
                next_state, reward, done = self.env.step(action)  # Realizar acción
                state = next_state  # Avanzar al siguiente estado
                step_count += 1  # Incrementar el contador de pasos
                # Renderizar el entorno después de cada acción
                self.env.render()  # Renderizar el entorno

            if step_count >= max_steps_per_test:
                print(f"Prueba {test + 1} terminada por exceder el límite de pasos ({max_steps_per_test}).")
            else:
                print(f"Prueba {test + 1} completada en {step_count} pasos.")
                

