import numpy as np, random

class GridWorld:
    def __init__(self, w=6, h=6, walls=None, start=(0,0), goal=(5,5)):
        self.w, self.h = w, h
        self.walls = set(walls or [(1,1),(1,2),(2,2),(3,1),(4,3)])
        self.start, self.goal = start, goal
        self.reset()

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, a):
        x,y = self.pos
        if a==0: y = max(0, y-1)
        elif a==1: x = min(self.w-1, x+1)
        elif a==2: y = min(self.h-1, y+1)
        elif a==3: x = max(0, x-1)
        if (x,y) in self.walls:
            return self.pos, -0.2, False
        if (x,y)==self.goal:
            self.pos = (x,y)
            return self.pos, 1.0, True
        self.pos = (x,y)
        return self.pos, -0.01, False

def train_q(episodes=1500, alpha=0.5, gamma=0.95, eps=0.2):
    env = GridWorld()
    Q = np.zeros((env.w, env.h, 4))
    for _ in range(episodes):
        s = env.reset(); done=False; steps=0
        while not done and steps<300:
            x,y = s
            a = random.randint(0,3) if random.random()<eps else int(np.argmax(Q[x,y]))
            s2, r, done = env.step(a)
            x2,y2 = s2
            Q[x,y,a] = (1-alpha)*Q[x,y,a] + alpha*(r + gamma*np.max(Q[x2,y2]))
            s = s2; steps += 1
    return Q

if __name__ == "__main__":
    Q = train_q()
    print("Q-learning finished. Policy learned.")