import numpy as np
import matplotlib.pyplot as plt
import random

class MazeEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.state = (0, 0)  # Posición inicial del agente
        self.goal = (height // 2, width // 2)  # Posición del objetivo (centro del laberinto)
        self.grid = np.ones((height, width))  # Crear una cuadrícula llena de paredes (1)
        self._generate_maze()  # Generar el laberinto con el algoritmo de Prim

    def _generate_maze(self):
        """Genera un laberinto utilizando el algoritmo de Prim."""
        def neighbors(x, y):
            """Devuelve las celdas vecinas que están dentro del laberinto."""
            directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
            result = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    result.append((nx, ny))
            return result

        # Iniciar desde la posición (1, 1) para evitar los bordes del laberinto
        self.grid[1, 1] = 0
        frontier = neighbors(1, 1)

        while frontier:
            # Escoger una celda aleatoria de la frontera
            cx, cy = random.choice(frontier)
            frontier.remove((cx, cy))

            # Escoger una vecina de la celda que ya forme parte del laberinto
            valid_neighbors = [n for n in neighbors(cx, cy) if self.grid[n] == 0]
            if valid_neighbors:
                nx, ny = random.choice(valid_neighbors)

                # Conectar la celda actual con la vecina
                wx, wy = (cx + nx) // 2, (cy + ny) // 2
                self.grid[cx, cy] = 0
                self.grid[wx, wy] = 0

                # Añadir las celdas vecinas de la nueva celda a la frontera
                frontier.extend([n for n in neighbors(cx, cy) if self.grid[n] == 1])

        # Asegurar que el punto inicial y el objetivo están libres
        self.grid[0, 0] = 0  # Liberar la posición inicial del agente
        self.grid[0, 1] = 0  # Liberar la posición inicial del agente
        self.grid[1, 0] = 0  # Liberar la posición inicial del agente
        
        self.grid[self.goal] = 0  # Liberar el objetivo (centro del laberinto)

    def reset(self):
        self.state = (0, 0)  # Reiniciar el estado del agente
        return self.state

    def step(self, action):
        """Mueve al agente en la dirección especificada."""
        if action == 0:  # Arriba
            new_state = (max(self.state[0] - 1, 0), self.state[1])
        elif action == 1:  # Abajo
            new_state = (min(self.state[0] + 1, self.height - 1), self.state[1])
        elif action == 2:  # Izquierda
            new_state = (self.state[0], max(self.state[1] - 1, 0))
        elif action == 3:  # Derecha
            new_state = (self.state[0], min(self.state[1] + 1, self.width - 1))
        else:
            raise ValueError("Acción no válida")

        # Verificar si la nueva posición es una pared
        if self.grid[new_state] == 1:
            new_state = self.state  # Si hay una pared, no se mueve

        self.state = new_state

        # Recompensa: +1 si llega al objetivo, -1 por cada paso
        if self.state == self.goal:
            return self.state, 10, True  # (nuevo estado, recompensa, fin del episodio)
        else:
            return self.state, -1, False  # (nuevo estado, recompensa, fin del episodio)

    def get_valid_actions(self):
        """Devuelve las acciones válidas: Arriba, Abajo, Izquierda, Derecha."""
        return [0, 1, 2, 3]

    def render(self):
        """Dibuja el entorno del laberinto."""
        plt.clf()  # Limpiar la figura actual
        plt.xlim(-0.5, self.width - 0.5)
        plt.ylim(-0.5, self.height - 0.5)

        # Dibujar la cuadrícula
        plt.grid(True)

        # Dibujar las paredes
        wall_positions = np.argwhere(self.grid == 1)
        for pos in wall_positions:
            plt.scatter(pos[1], pos[0], color='black', s=100)  # Pared

        # Dibujar el agente
        plt.scatter(self.state[1], self.state[0], color='blue', s=100, label='Agente')  # Agente
        # Dibujar el objetivo
        plt.scatter(self.goal[1], self.goal[0], color='red', s=100, label='Objetivo')  # Objetivo

        # Etiquetas y leyenda
        plt.xticks(range(self.width))
        plt.yticks(range(self.height))
        plt.gca().invert_yaxis()  # Invertir el eje Y para que (0,0) esté en la esquina superior izquierda
        #plt.legend()
        plt.title("Laberinto")
        plt.pause(0.1)  # Pausa para permitir la visualización
