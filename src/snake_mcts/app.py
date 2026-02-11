import sys
from pathlib import Path


root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

import streamlit as st
import time
from .agent import MCTS
from .util import safe_actions, find_path_bfs, grid_to_rgb
from .environnement import SnakeEnvironment
# --- CONFIGURATION UI ---
st.set_page_config(layout="centered", page_title="MCTS Snake Project")
st.title("MCTS Snake — Architecture Modulaire")

# --- SIDEBAR : PARAMÈTRES ---
with st.sidebar:
    st.header("Configuration")
    grid_size = st.slider("Taille de grille", 6, 20, 12)
    speed = st.slider("Vitesse (sec/step)", 0.0, 0.5, 0.05)
    
    st.header("Paramètres IA")
    simulations = st.number_input("MCTS Simulations", 10, 1000, 300)
    follow_path_threshold = st.slider("Seuil BFS (distance)", 1, 50, 20)
    
    with st.expander("Shaping Avancé"):
        guided_prob = st.slider("Probabilité guidée", 0.0, 1.0, 0.85)
        trap_penalty = st.slider("Pénalité de piège", 0.0, 1.0, 0.15)

# --- INITIALISATION ---
if "game_active" not in st.session_state:
    st.session_state.game_active = False

start_button = st.button(" Démarrer la simulation")

# --- ÉLÉMENTS GRAPHIQUES ---
canvas = st.empty()
score_box = st.empty()
info_box = st.empty()

# --- BOUCLE DE JEU ---
if start_button:
    env = SnakeEnvironment(grid_size=grid_size)
    agent = MCTS(
        simulations=simulations,
        guided_prob=guided_prob,
        trap_penalty=trap_penalty
    )
    
    env.reset()
    step = 0
    
    while not env.done:
        # 1. Calcul des actions sûres
        safe = safe_actions(env)
        
        # 2. Stratégie Hybride : BFS d'abord, sinon MCTS
        path = find_path_bfs(env, allow_tail=True)
        
        if path and len(path) <= follow_path_threshold and path[0] in safe:
            action = path[0]
            reason = f"BFS (Chemin: {len(path)})"
        else:
            action = agent.choose(env, forced_legal=safe)
            reason = "MCTS (Exploration)"

        # 3. Mise à jour de l'environnement
        state, reward, done, _ = env.step(action)
        
        # 4. Rendu visuel
        img = grid_to_rgb(state, cell_size=max(10, 400 // grid_size))
        canvas.image(img, width=400)
        
        # 5. Dashboard
        score_box.metric("Score", env.score, f"Step: {step}")
        info_box.info(f"Décision : **{reason}**")
        
        step += 1
        time.sleep(speed)
        
        if done:
            st.error(f"Game Over ! Score final : {env.score}")
            break