"""A tiny Q-learning maze demo saved as PNG under report/figures/maze_policy.png."""
import numpy as np
import matplotlib.pyplot as plt
from src.utils import FIG

def q_learning_maze(n=6, episodes=1500, alpha=0.1, gamma=0.95, eps=0.2, seed=13):
    rng = np.random.default_rng(seed)
    # Build a simple grid with obstacles
    grid = np.zeros((n, n), dtype=int)
    obstacles = [(1,1),(1,2),(2,1),(3,3),(4,2)]
    for r,c in obstacles:
        if 0 <= r < n and 0 <= c < n:
            grid[r,c] = -1  # wall
    start, goal = (0,0), (n-1,n-1)
    A = [(0,1),(1,0),(0,-1),(-1,0)]  # R,D,L,U
    Q = np.zeros((n,n,len(A)))

    def step(s, a):
        r, c = s
        dr, dc = A[a]
        nr, nc = r+dr, c+dc
        if nr<0 or nr>=n or nc<0 or nc>=n or grid[nr,nc]==-1:
            return (r,c), -1.0, False
        if (nr, nc) == goal:
            return (nr,nc), 10.0, True
        return (nr,nc), -0.05, False

    for _ in range(episodes):
        s = start
        done = False
        while not done:
            if rng.random() < eps:
                a = rng.integers(0, len(A))
            else:
                a = int(np.argmax(Q[s[0], s[1]]))
            ns, r, done = step(s, a)
            Q[s[0], s[1], a] += alpha * (r + gamma * np.max(Q[ns[0], ns[1]]) - Q[s[0], s[1], a])
            s = ns

    # Derive greedy policy arrows
    policy = np.full((n,n), '•', dtype=object)
    arrows = {0:'→',1:'↓',2:'←',3:'↑'}
    for r in range(n):
        for c in range(n):
            if (r,c)==start: policy[r,c]='S'; continue
            if (r,c)==goal: policy[r,c]='G'; continue
            if grid[r,c]==-1: policy[r,c]='■'; continue
            policy[r,c] = arrows[int(np.argmax(Q[r,c]))]

    # Plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title("Q-Learning Maze Policy")
    ax.imshow(grid==-1, cmap='gray', vmin=0, vmax=1)
    for (r,c), ch in np.ndenumerate(policy):
        ax.text(c, r, ch, ha='center', va='center', fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = FIG / "maze_policy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

if __name__ == "__main__":
    path = q_learning_maze()
    print(f"🧭  Saved {path}")
