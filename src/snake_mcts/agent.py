import math
import random
import numpy as np
from snake_mcts.environnement import ACTIONS

class MCTSNode:
    """Représente un état dans l'arbre de recherche MCTS."""
    def __init__(self, parent, action_from_parent, env_state):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.env_state = env_state
        self.children = {}  # action -> MCTSNode
        self.visits = 0
        self.value = 0.0
        self.untried_actions = env_state.legal_actions()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        """Sélectionne le meilleur enfant en utilisant la formule UCB1."""
        best_score = -float('inf')
        best_action, best_node = None, None

        for a, child in self.children.items():
            if child.visits == 0:
                score = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(2 * math.log(max(1, self.visits)) / child.visits)
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_action = a
                best_node = child
        return best_action, best_node

class MCTS:
    """L'algorithme Monte Carlo Tree Search adapté au Snake."""
    def __init__(self, simulations=300, rollout_depth=150, c_param=1.4, 
                 guided_prob=0.85, open_space_weight=0.02, trap_penalty=0.15, serpentin_bonus=0.03):
        self.simulations = int(simulations)
        self.rollout_depth = int(rollout_depth)
        self.c_param = c_param
        self.guided_prob = guided_prob
        self.open_space_weight = open_space_weight
        self.trap_penalty = trap_penalty
        self.serpentin_bonus = serpentin_bonus

    def _get_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _future_legal_count(self, env, nx, ny):
        """Compte les cases adjacentes libres pour évaluer l'espace."""
        cnt = 0
        for a in [0, 1, 2, 3]:
            dx, dy = ACTIONS[a].dx, ACTIONS[a].dy
            xx, yy = nx + dx, ny + dy
            if 0 <= xx < env.width and 0 <= yy < env.height and (xx, yy) not in env.snake:
                cnt += 1
        return cnt

    def rollout(self, env):
        """Phase de simulation aléatoire guidée."""
        total_reward = 0.0
        depth = 0
        visited = set()
        
        # Profondeur dynamique selon l'espace restant
        remaining_space = env.width * env.height - len(env.snake)
        dynamic_depth = int(self.rollout_depth * (1.0 if remaining_space > env.width * env.height * 0.4 else 1.4))

        while (not env.done) and (depth < dynamic_depth):
            legal = env.legal_actions()
            head = env.snake[0]
            fx, fy = env.food

            # Filtrage pour éviter les morts stupides en simulation
            safe = [a for a in legal if (head[0]+ACTIONS[a].dx, head[1]+ACTIONS[a].dy) not in list(env.snake)[:-1]]
            if not safe:
                action = random.choice(legal)
            else:
                # Choisir l'action qui rapproche de la nourriture (probabilité guided_prob)
                best_a = min(safe, key=lambda a: self._get_distance((head[0]+ACTIONS[a].dx, head[1]+ACTIONS[a].dy), env.food))
                action = best_a if random.random() < self.guided_prob else random.choice(safe)

            # --- Reward Shaping ---
            nx, ny = head[0] + ACTIONS[action].dx, head[1] + ACTIONS[action].dy
            dist_before = self._get_distance(head, env.food)
            dist_after = self._get_distance((nx, ny), env.food)
            
            total_reward += (dist_before - dist_after) * 0.08  # Attraction vers la nourriture
            total_reward += self._future_legal_count(env, nx, ny) * self.open_space_weight
            
            if (nx, ny) in visited: total_reward -= 0.05
            else: visited.add((nx, ny))

            _, r, _, _ = env.step(action)
            total_reward += r
            depth += 1

        return total_reward

    def choose(self, root_env, forced_legal=None):
        """Exécute les simulations et retourne la meilleure action."""
        root = MCTSNode(parent=None, action_from_parent=None, env_state=root_env.clone())
        
        if forced_legal is not None:
            root.untried_actions = list(forced_legal)

        for _ in range(self.simulations):
            node = root
            env_sim = node.env_state.clone()

            # 1. Selection
            while node.is_fully_expanded() and len(node.children) > 0:
                _, node = node.best_child(c_param=self.c_param)
                env_sim.step(node.action_from_parent)

            # 2. Expansion
            if node.untried_actions and not env_sim.done:
                a = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
                env_sim.step(a)
                child_node = MCTSNode(parent=node, action_from_parent=a, env_state=env_sim.clone())
                node.children[a] = child_node
                node = child_node

            # 3. Simulation
            reward = self.rollout(env_sim.clone())

            # 4. Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

        # Meilleure action basée sur la moyenne des récompenses
        return max(root.children.keys(), key=lambda a: root.children[a].value / root.children[a].visits)