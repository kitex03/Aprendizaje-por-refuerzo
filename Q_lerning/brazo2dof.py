import numpy as np
import math
import matplotlib.pyplot as plt

class Acrobot_Env:
    def __init__(self, target_position=(1.0, 1.0), tolerance=0.05):
        # Parámetros del sistema físico (ajustados para mayor dificultad)
        self.LINK_LENGTH_1 = 1.0  # Longitud del primer enlace
        self.LINK_LENGTH_2 = 1.0  # Longitud del segundo enlace
        self.LINK_MASS_1 = 2.0    # Masa del primer enlace (aumentada)
        self.LINK_MASS_2 = 2.0    # Masa del segundo enlace (aumentada)
        self.LINK_COM_POS_1 = 0.5  # Centro de masa del primer enlace
        self.LINK_COM_POS_2 = 0.5  # Centro de masa del segundo enlace
        self.LINK_MOI_1 = 0.5     # Momento de inercia del primer enlace (aumentado)
        self.LINK_MOI_2 = 0.5     # Momento de inercia del segundo enlace (aumentado)

        self.MAX_VEL_1 = 2 * np.pi  # Velocidad angular máxima de la primera articulación
        self.MAX_VEL_2 = 2 * np.pi  # Velocidad angular máxima de la segunda articulación

        self.dt = 0.2  # Paso de tiempo

        # Limites de torque reducidos para aumentar la dificultad
        self.TORQUE = 0.5
        self.actions = [-self.TORQUE, 0.0, self.TORQUE]  # Menos fuerza aplicada en cada acción

        # Posición objetivo (x, y)
        self.target_position = target_position
        self.tolerance = tolerance  # Tolerancia de la distancia al objetivo

        # Inicialización del estado
        self.reset()

    def reset(self):
        """Resetea el entorno al estado inicial"""
        # Inicializar ángulos y velocidades angulares con valores aleatorios
        self.state = np.array([
           0,# np.random.uniform(-np.pi, np.pi),  # ángulo de la primera articulación
           0,# np.random.uniform(-np.pi, np.pi),  # ángulo de la segunda articulación
           0,# np.random.uniform(-1, 1),          # velocidad angular de la primera articulación
           0,# np.random.uniform(-1, 1)           # velocidad angular de la segunda articulación
        ])
        return self._get_observation()

    def step(self, action):
        """Realiza un paso en el entorno con la acción dada"""
        torque = self.actions[action]

        # Simular la física del sistema usando ecuaciones dinámicas (Euler o Runge-Kutta)
        self._apply_dynamics(torque)

        # Obtener nueva observación
        obs = self._get_observation()

        # Definir la condición de éxito: si el extremo del acrobot está cerca de la posición objetivo
        done = self._terminal()

        # Recompensa basada en la distancia al objetivo
        reward = self._calculate_reward()

        return obs, reward, done, {}

    def _apply_dynamics(self, torque):
        """Aplica las ecuaciones de movimiento del Acrobot"""
        # Extraer el estado actual
        theta1, theta2, theta_dot1, theta_dot2 = self.state

        # Aceleración angular para cada articulación (con aumento de dificultad)
        accel1, accel2 = self._compute_accelerations(theta1, theta2, theta_dot1, theta_dot2, torque)

        # Integrar las aceleraciones para obtener nuevas velocidades y posiciones
        theta_dot1 += accel1 * self.dt
        theta_dot2 += accel2 * self.dt

        # Limitar las velocidades angulares
        theta_dot1 = np.clip(theta_dot1, -self.MAX_VEL_1, self.MAX_VEL_1)
        theta_dot2 = np.clip(theta_dot2, -self.MAX_VEL_2, self.MAX_VEL_2)

        # Actualizar las posiciones angulares
        theta1 += theta_dot1 * self.dt
        theta2 += theta_dot2 * self.dt

        # Normalizar los ángulos entre [-pi, pi]
        theta1 = ((theta1 + np.pi) % (2 * np.pi)) - np.pi
        theta2 = ((theta2 + np.pi) % (2 * np.pi)) - np.pi

        # Guardar el nuevo estado
        self.state = np.array([theta1, theta2, theta_dot1, theta_dot2])

    def _compute_accelerations(self, theta1, theta2, theta_dot1, theta_dot2, torque):
        """Calcula las aceleraciones angulares (con más dificultad debido a las mayores masas y MOI)"""
        # Ecuaciones de movimiento simplificadas para el ejemplo (más dificultad añadida)
        accel1 = (-2.0 * torque + 0.05 * theta_dot1 - 0.1 * np.cos(theta1))  # Simplificación con más masa
        accel2 = (-2.0 * torque + 0.05 * theta_dot2 - 0.1 * np.cos(theta2))  # Simplificación con más masa
        return accel1, accel2

    def _terminal(self):
        """Comprueba si el acrobot ha alcanzado la posición objetivo"""
        x, y = self._get_end_effector_position()
        x_target, y_target = self.target_position

        # Comprobar si está dentro de una tolerancia de la posición objetivo
        distance_to_target = np.sqrt((x - x_target) ** 2 + (y - y_target) ** 2)
        return distance_to_target <= self.tolerance

    def _calculate_reward(self):
        """Calcula la recompensa basada en la distancia al objetivo"""
        x, y = self._get_end_effector_position()
        x_target, y_target = self.target_position

        # Recompensa negativa proporcional a la distancia al objetivo
        distance_to_target = np.sqrt((x - x_target) ** 2 + (y - y_target) ** 2)
        reward = -distance_to_target

        # Si está dentro de la región objetivo, dar una recompensa positiva
        if distance_to_target <= self.tolerance:
            reward = 100.0  # Recompensa positiva alta cuando alcanza el objetivo

        return reward

    def _get_end_effector_position(self):
        """Calcula la posición (x, y) del extremo del brazo del acrobot"""
        theta1, theta2, _, _ = self.state

        # Coordenadas del extremo del segundo enlace
        x = self.LINK_LENGTH_1 * np.sin(theta1) + self.LINK_LENGTH_2 * np.sin(theta1 + theta2)
        y = -self.LINK_LENGTH_1 * np.cos(theta1) - self.LINK_LENGTH_2 * np.cos(theta1 + theta2)

        return x, y

    def _get_observation(self):
        """Obtiene el estado observado"""
        theta1, theta2, theta_dot1, theta_dot2 = self.state
        return np.array([np.cos(theta1), np.sin(theta1), np.cos(theta2), np.sin(theta2), theta_dot1, theta_dot2])

    def render(self):
        """Dibuja el sistema usando Matplotlib"""
        theta1, theta2, _, _ = self.state

        # Posiciones de los enlaces
        x0, y0 = 0, 0
        x1 = self.LINK_LENGTH_1 * np.sin(theta1)
        y1 = -self.LINK_LENGTH_1 * np.cos(theta1)
        x2 = x1 + self.LINK_LENGTH_2 * np.sin(theta1 + theta2)
        y2 = y1 - self.LINK_LENGTH_2 * np.cos(theta1 + theta2)

        # Limpiar la figura
        plt.clf()

        # Dibujar el acrobot
        plt.plot([x0, x1], [y0, y1], 'o-', markersize=8, linewidth=2, color='blue')  # Primer enlace
        plt.plot([x1, x2], [y1, y2], 'o-', markersize=8, linewidth=2, color='green')  # Segundo enlace

        # Dibujar el extremo del acrobot y la posición objetivo
        plt.plot([x2], [y2], 'ro')  # Extremo del acrobot
        plt.plot([self.target_position[0]], [self.target_position[1]], 'rx', markersize=10, markeredgewidth=2)  # Objetivo

        # Configurar el gráfico
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal')
        plt.draw()
        plt.pause(0.001)



# Ejemplo de uso
if __name__ == "__main__":

    env = Acrobot_Env(target_position=(1.5, 1.5), tolerance=0.01)
    obs = env.reset()

    for _ in range(200):
        action = np.random.choice([0, 1, 2])  # Acción aleatoria
        obs, reward, done, _ = env.step(action)
        env.render()

        if done:
            print("¡Objetivo alcanzado!")
            break
        print("Episodio terminado")

    plt.show()
