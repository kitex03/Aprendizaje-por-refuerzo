import numpy as np
import matplotlib.pyplot as plt
import random
import math

class MultiGoalEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.goal = (random.randint(0,width-1) , random.randint(0,height-1))  # Posición del objetivo (centro del laberinto)
        self.pos = (random.randint(0,self.height-1) , random.randint(0,self.width-1))
        self.state = ((self.pos[0]+ self.width * self.pos[1]), self.goal[0] * self.width + self.goal[1]) # Posición inicial del agente
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
        self.grid[self.pos] = 0  # Liberar la posición inicial del agente
        
        self.grid[self.goal] = 0  # Liberar el objetivo (centro del laberinto)

    def reset(self):
        self.goal = (random.randint(0,self.width-1) , random.randint(0,self.height-1))
        while self.grid[self.goal] == 1:
            self.goal = (random.randint(0,self.width-1) , random.randint(0,self.height-1))
        self.state = ((random.randint(0,self.height-1)+ self.width * random.randint(0,self.width-1)), self.goal[0] * self.width + self.goal[1])  # Reiniciar el estado del agente
        while self.grid[self.get_grid(self.state)] == 1:
            self.state = ((random.randint(0,self.height-1)+ self.width * random.randint(0,self.width-1)), self.goal[0] * self.width + self.goal[1])
        # print("estado inicial: ", self.state)
        return self.state

    def step(self, action):
        """Mueve al agente en la dirección especificada."""
        i, j = self.get_grid(self.state)
        # print("estado actual i, j: ", i, j)
        # print("estado actual: ", self.state)
        # print("accion: ", action)
        # print("objetivo: ", self.goal)
        
        if action == 0:  # Arriba
            new_i, new_j = max(i - 1, 0), j
        elif action == 1:  # Abajo
            new_i, new_j = min(i + 1, self.height - 1), j
        elif action == 2:  # Izquierda
            new_i, new_j = i, max(j - 1, 0)
        elif action == 3:  # Derecha
            new_i, new_j = i, min(j + 1, self.width - 1)
        else:
            raise ValueError("Acción no válida")
    
        # Verificar si la nueva posición es una pared
        if self.grid[new_i, new_j] == 1:
            new_i, new_j = i, j  # Si hay una pared, no se mueve
    
        self.state = (new_i + new_j* self.width , self.state[1])
        # print("Valor de la celda: ",new_i," ,",new_j," ", self.grid[new_i, new_j])
        # print("nuevo estado: ", new_i, new_j)
    
        # Recompensa: +1 si llega al objetivo, -1 por cada paso
        distance = math.dist((new_i, new_j), self.goal)
        # print("distancia: ", distance)
        if (new_i, new_j) == self.goal:
            # print("Llego al objetivo ", new_i, new_j, self.goal)
            return self.state, 10, True  # (nuevo estado, recompensa, fin del episodio)
        else:
            if distance != 0:
                return self.state, -distance, False  # (nuevo estado, recompensa, fin del episodio))

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
        i,j = self.get_grid(self.state)
        plt.scatter(j, i, color='blue', s=100, label='Agente')  # Agente
        # Dibujar el objetivo
        plt.scatter(self.goal[1], self.goal[0], color='red', s=100, label='Objetivo')  # Objetivo

        # Etiquetas y leyenda
        plt.xticks(range(self.width))
        plt.yticks(range(self.height))
        plt.gca().invert_yaxis()  # Invertir el eje Y para que (0,0) esté en la esquina superior izquierda
        #plt.legend()
        plt.title("LaberintoMultigoal")
        plt.pause(0.1)  # Pausa para permitir la visualización
    
    def get_grid(self,state):
        return int(state[0] % self.width) , int(state[0] / self.width)