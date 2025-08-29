#!/usr/bin/env python3
"""
Optional: very small Q-learning maze stub to expand later.
"""
import numpy as np

def make_grid(h=5, w=7, walls={(1,1),(1,2)}, start=(0,0), goal=(4,6)):
    return {"h":h,"w":w,"walls":set(walls),"start":start,"goal":goal}

def step(env, s, a):
    y,x = s
    moves = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}  # up,down,left,right
    dy,dx = moves[a]
    ny, nx = max(0,min(env["h"]-1,y+dy)), max(0,min(env["w"]-1,x+dx))
    ns = (ny,nx)
    if ns in env["walls"]:
        ns = s  # bump
    reward = 1.0 if ns == env["goal"] else -0.01
    done = ns == env["goal"]
    return ns, reward, done

def train_qlearning(episodes=500, alpha=0.3, gamma=0.95, eps=0.1):
    env = make_grid()
    Q = np.zeros((env["h"], env["w"], 4))
    for ep in range(episodes):
        s = env["start"]
        done = False
        while not done:
            if np.random.rand() < eps:
                a = np.random.randint(4)
            else:
                a = np.argmax(Q[s[0], s[1]])
            ns, r, done = step(env, s, a)
            Q[s[0],s[1],a] += alpha * (r + gamma * Q[ns[0],ns[1]].max() - Q[s[0],s[1],a])
            s = ns
    return Q

if __name__ == "__main__":
    Q = train_qlearning()
    print("✅ Trained simple Q-table. Shape:", Q.shape)
