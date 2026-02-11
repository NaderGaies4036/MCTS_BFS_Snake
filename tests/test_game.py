import pytest
import numpy as np
from src.snake_mcts.environnement import SnakeEnvironment
from src.snake_mcts.util import safe_actions

def test_snake_eats_food():
    """Vérifie que le serpent grandit et marque un point quand il mange."""
    env = SnakeEnvironment(grid_size=10)
    env.reset()
    
    # Forcer la position de la nourriture juste devant la tête
    head_x, head_y = env.snake[0]
    env.food = (head_x + 1, head_y)
    
    # Action 1 (Right)
    env.step(1)
    
    assert env.score == 1
    assert len(env.snake) == 2
    assert env.snake[0] == (head_x + 1, head_y)

def test_collision_wall():
    """Vérifie que le jeu s'arrête en touchant un mur."""
    env = SnakeEnvironment(grid_size=10)
    env.reset()
    
    # On déplace le serpent vers le mur de gauche jusqu'à collision
    # (Position initiale est au centre 5,5)
    for _ in range(6):
        _, _, done, _ = env.step(3) # Left
        
    assert done is True

def test_safe_actions_filter():
    """Vérifie que safe_actions empêche de foncer dans un mur."""
    env = SnakeEnvironment(grid_size=6)
    env.reset()
    # On place le serpent au bord du mur du haut (y=0)
    env.snake = [(3, 0)]
    
    safe = safe_actions(env)
    # L'action 0 (Up) ne devrait pas être dans la liste safe
    assert 0 not in safe