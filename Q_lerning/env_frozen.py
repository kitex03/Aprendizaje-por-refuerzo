import numpy as np
import matplotlib.pyplot as plt
import random

class FrozenLakeEnvironment:
    def __init__(self, width, height, hole_prob=0.2, slippery=True, slippery_float = 0.1):
        self.width = width
        self.height = height
        self.hole_prob = hole_prob  # Probabilidad de que una celda sea un agujero
        self.slippery = slippery  # Indica si el suelo es resbaladizo
        self.state = (0, 0)  # Posición inicial del agente
        self.goal = (height - 1, width - 1)  # Posición de la meta
        self.lake = self._generate_lake()  # Generar el lago (0: seguro, 1: agujero)
        self.slippery_float = slippery_float
        
    def _generate_lake(self):
        """Genera un lago con agujeros aleatorios según la probabilidad dada."""
        lake = np.zeros((self.height, self.width))

        # Llenar el lago con agujeros aleatorios, dejando el inicio y la meta sin agujero
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) != (0, 0) and (i, j) != self.goal:
                    if random.random() < self.hole_prob:
                        lake[i, j] = 1  # Un agujero

        return lake

    def reset(self):
        """Reinicia el entorno a la posición inicial."""
        self.state = (0, 0)
        return self.state

    def step(self, action):
        """Realiza una acción y devuelve el nuevo estado, recompensa y si terminó el episodio."""
        if self.slippery:
            if random.random() < self.slippery_float:  # Cambia la acción con un 20% de probabilidad
                action = random.choice([0, 1, 2, 3])  # Elegir una acción aleatoria

        # Obtener el nuevo estado basado en la acción
        new_state = self._move(action)

        # Verificar si el nuevo estado es un agujero
        if self.lake[new_state] == 1:
            return new_state, -10, True  # Fuerte penalización por caer en un agujero, termina el episodio

        # Verificar si el nuevo estado es la meta
        if new_state == self.goal:
            return new_state, 10, True  # Recompensa por llegar a la meta, termina el episodio

        # Si no es ni agujero ni meta, actualizar el estado y devolver -1 por cada paso
        self.state = new_state
        return self.state, -1, False  # Actualizar el estado y devolver la recompensa por el paso

    def _move(self, action):
        """Calcula el nuevo estado en función de la acción."""
        x, y = self.state
        if action == 0:  # Arriba
            x = max(0, x - 1)
        elif action == 1:  # Abajo
            x = min(self.height - 1, x + 1)
        elif action == 2:  # Izquierda
            y = max(0, y - 1)
        elif action == 3:  # Derecha
            y = min(self.width - 1, y + 1)
        return (x, y)

    def get_valid_actions(self):
        """Devuelve las acciones válidas: arriba, abajo, izquierda, derecha."""
        return [0, 1, 2, 3]  # Arriba, Abajo, Izquierda, Derecha

    def render(self):
        """Dibuja el entorno del lago congelado."""
        plt.clf()
        plt.xlim(-0.5, self.width - 0.5)
        plt.ylim(-0.5, self.height - 0.5)

        # Dibujar los agujeros en el lago
        hole_positions = np.argwhere(self.lake == 1)
        for pos in hole_positions:
            plt.scatter(pos[1], pos[0], color='black', s=100)  # Agujero

        # Dibujar el agente
        plt.scatter(self.state[1], self.state[0], color='blue', s=100, label='Agente')

        # Dibujar la meta
        plt.scatter(self.goal[1], self.goal[0], color='red', s=100, label='Meta')

        # Etiquetas y leyenda
        plt.xticks(range(self.width))
        plt.yticks(range(self.height))
        plt.gca().invert_yaxis()  # Para que (0,0) esté en la parte superior izquierda
        plt.legend()
        plt.title("Frozen Lake con slippery = " + str(self.slippery_float))
        plt.pause(0.15)

    def set_slippery(self, slippery):
        """Permite modificar si el suelo es resbaladizo o no."""
        self.slippery = slippery
