#!/usr/bin/env python
\"\"\"Q-learning maze navigation with plots.\"\"\"
import logging, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.io import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", mode="a")
    ],
)
LOGGER = logging.getLogger("04_maze_model")

ACTIONS = [(0,1), (1,0), (0,-1), (-1,0)]  # R, D, L, U

def in_bounds(r, c, rows, cols):
    return 0 <= r < rows and 0 <= c < cols

def main():
    cfg = load_config(\"config.yaml\")
    rl = cfg[\"rl\"]
    rows, cols = rl[\"maze_rows\"], rl[\"maze_cols\"]
    start = tuple(rl[\"start\"])
    goal = tuple(rl[\"goal\"])

    rng = np.random.default_rng(7)
    maze = np.zeros((rows, cols), dtype=int)
    for _ in range(int(rows*cols*0.2)):
        r, c = int(rng.integers(0, rows)), int(rng.integers(0, cols))
        if (r, c) not in [start, goal]:
            maze[r, c] = 1

    Q = np.zeros((rows, cols, len(ACTIONS)))
    alpha, gamma, eps, episodes = rl[\"alpha\"], rl[\"gamma\"], rl[\"epsilon\"], rl[\"episodes\"]
    rewards = []

    for ep in range(episodes):
        s = start
        total = 0.0
        for _ in range(rows*cols*4):
            if np.random.rand() < eps:
                a = np.random.randint(0, len(ACTIONS))
            else:
                a = int(np.argmax(Q[s[0], s[1], :]))
            dr, dc = ACTIONS[a]
            nr, nc = s[0]+dr, s[1]+dc
            if not in_bounds(nr, nc, rows, cols) or maze[nr, nc] == 1:
                rwd = -5.0
                ns = s
            elif (nr, nc) == goal:
                rwd = 10.0
                ns = (nr, nc)
            else:
                rwd = -0.1
                ns = (nr, nc)
            Q[s[0], s[1], a] = (1-alpha)*Q[s[0], s[1], a] + alpha*(rwd + gamma*np.max(Q[ns[0], ns[1], :]))
            s = ns
            total += rwd
            if s == goal:
                break
        rewards.append(total)

    s = start
    path = [s]
    visited = set([s])
    for _ in range(rows*cols*4):
        a = int(np.argmax(Q[s[0], s[1], :]))
        dr, dc = ACTIONS[a]
        nr, nc = s[0]+dr, s[1]+dc
        if not in_bounds(nr, nc, rows, cols) or maze[nr, nc] == 1:
            break
        s = (nr, nc)
        if s in visited:
            break
        visited.add(s)
        path.append(s)
        if s == goal:
            break

    figures = Path(\"report/figures\")
    figures.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(rewards)
    plt.xlabel(\"Episode\")
    plt.ylabel(\"Total Reward\")
    plt.title(\"Q-Learning: Reward per Episode\")
    plt.tight_layout()
    plt.savefig(figures / \"rl_reward_curve.png\", dpi=150)
    plt.close()

    grid = np.ones((rows, cols))
    grid[maze==1] = 0.0

    plt.figure()
    plt.imshow(grid, interpolation=\"none\")
    if len(path) > 1:
        pr, pc = zip(*path)
        plt.plot(pc, pr, linewidth=2)
    plt.scatter([start[1], goal[1]], [start[0], goal[0]])
    plt.title(\"Maze and Learned Path\")
    plt.tight_layout()
    plt.savefig(figures / \"rl_maze_path.png\", dpi=150)
    plt.close()

if __name__ == \"__main__\":
    main()
