import numpy as np
import random
import copy
from collections import namedtuple

# Définition des actions possibles
Action = namedtuple("Action", ["dx", "dy"])

# 0: up, 1: right, 2: down, 3: left
ACTIONS = {
    0: Action(0, -1),
    1: Action(1, 0),
    2: Action(0, 1),
    3: Action(-1, 0)
}

class SnakeEnvironment:
    """
    Gère la logique du jeu Snake : mouvement, collisions, score et placement de nourriture.
    """
    def __init__(self, grid_size: int = 12, seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.width = grid_size
        self.height = grid_size
        self.reset()

    def reset(self):
        """Réinitialise le jeu à l'état initial."""
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = 0  # Index de l'action par défaut
        self.done = False
        self.score = 0
        self.place_food()
        return self.get_state()

    def place_food(self):
        """Place la nourriture aléatoirement sur la grille (hors bordure si possible)."""
        while True:
            fx = int(np.random.randint(1, self.width - 1))
            fy = int(np.random.randint(1, self.height - 1))
            f = (fx, fy)
            if f not in self.snake:
                self.food = f
                return

    def legal_actions(self):
        """Retourne la liste des actions qui ne sont pas un demi-tour immédiat."""
        legal = [0, 1, 2, 3]
        if len(self.snake) > 1:
            # On ne peut pas faire demi-tour sur soi-même
            rev = (self.direction + 2) % 4
            if rev in legal:
                legal.remove(rev)
        return legal

    def step(self, action: int):
        """
        Applique une action et retourne (nouvel_état, reward, terminé, info).
        """
        dx, dy = ACTIONS[action].dx, ACTIONS[action].dy
        head = (self.snake[0][0] + dx, self.snake[0][1] + dy)
        info = {}

        # 1. Collision avec les murs
        if head[0] < 0 or head[0] >= self.width or head[1] < 0 or head[1] >= self.height:
            self.done = True
            return self.get_state(), -10.0, True, info

        # 2. Collision avec son propre corps
        if head in self.snake:
            self.done = True
            return self.get_state(), -10.0, True, info

        # 3. Déplacement
        self.snake.insert(0, head)
        reward = 0.0
        
        if head == self.food:
            reward = 10.0
            self.score += 1
            self.place_food()
        else:
            # On retire la queue si pas de nourriture mangée
            self.snake.pop()

        self.direction = action
        return self.get_state(), reward, self.done, info

    def get_state(self):
        """Retourne une représentation matricielle de la grille."""
        grid = np.zeros((self.height, self.width), dtype=int)
        for (x, y) in self.snake:
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y, x] = 1
        fx, fy = self.food
        grid[fy, fx] = 2
        return grid

    def clone(self):
        """Crée une copie profonde de l'environnement pour les simulations MCTS."""
        new_env = SnakeEnvironment(grid_size=self.width)
        new_env.snake = copy.deepcopy(self.snake)
        new_env.direction = self.direction
        new_env.done = self.done
        new_env.score = self.score
        new_env.food = copy.deepcopy(self.food)
        return new_env