import numpy as np
from collections import deque
from .environnement import ACTIONS

def safe_actions(env):
    """
    Filtre les actions pour éviter les murs et les collisions immédiates.
    La queue du serpent est exclue car elle va bouger au prochain tour.
    """
    head_x, head_y = env.snake[0]
    # La queue va bouger, donc on ne considère que le corps jusqu'à l'avant-dernière cellule
    snake_body = set(env.snake[:-1])

    safe = []
    for a in env.legal_actions():
        dx, dy = ACTIONS[a].dx, ACTIONS[a].dy
        nx, ny = head_x + dx, head_y + dy

        # Vérification murs
        if nx < 0 or nx >= env.width or ny < 0 or ny >= env.height:
            continue
        # Vérification corps
        if (nx, ny) in snake_body:
            continue
        safe.append(a)

    return safe if safe else env.legal_actions()

def find_path_bfs(env, allow_tail=True, max_search_nodes=10000):
    """
    BFS pour trouver le chemin le plus court vers la nourriture.
    """
    start = env.snake[0]
    target = env.food
    width, height = env.width, env.height

    occupied = set(env.snake[:-1]) if allow_tail else set(env.snake)

    q = deque([start])
    parent = {start: None}
    parent_action = {}

    nodes_visited = 0
    while q and nodes_visited < max_search_nodes:
        nodes_visited += 1
        cur = q.popleft()
        if cur == target:
            break
        
        cx, cy = cur
        # On teste les 4 directions
        for a_idx, a in ACTIONS.items():
            nx, ny = cx + a.dx, cy + a.dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in occupied:
                if (nx, ny) not in parent:
                    parent[(nx, ny)] = cur
                    parent_action[(nx, ny)] = a_idx
                    q.append((nx, ny))

    if target not in parent:
        return None

    # Reconstitution du chemin
    path = []
    node = target
    while parent[node] is not None:
        path.append(parent_action[node])
        node = parent[node]
    path.reverse()
    return path

def grid_to_rgb(grid, cell_size=28):
    """
    Transforme la matrice numpy en une image RGB pour l'affichage Streamlit.
    """
    h, w = grid.shape
    img_h, img_w = h * cell_size + 1, w * cell_size + 1

    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    COLOR_SNAKE = np.array([40, 40, 40])    # Gris foncé
    COLOR_FOOD = np.array([255, 0, 0])      # Rouge
    COLOR_GRID = np.array([200, 200, 200])  # Gris clair

    for i in range(h):
        for j in range(w):
            y1, x1 = i * cell_size, j * cell_size
            y2, x2 = y1 + cell_size, x1 + cell_size
            if grid[i, j] == 1:
                img[y1:y2, x1:x2] = COLOR_SNAKE
            elif grid[i, j] == 2:
                img[y1:y2, x1:x2] = COLOR_FOOD

    # Lignes de la grille
    for i in range(h + 1):
        y = i * cell_size
        img[y:y+1, :] = COLOR_GRID
    for j in range(w + 1):
        x = j * cell_size
        img[:, x:x+1] = COLOR_GRID

    return img