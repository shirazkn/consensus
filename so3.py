import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt

import mayavi.mlab as mlab

from circle import AgentSO2

E1 = np.array(
    [[0, 0, 0],
    [0, 0, -1],
    [0, 1, 0]]
)

E2 = np.array(
    [[0, 0, 1],
    [0, 0, 0],
    [-1, 0, 0]]
)

E3 = np.array(
    [[0, -1, 0],
    [1, 0, 0],
    [0, 0, 0]]
)

steps = 5000
threshold = 0.001

timestep = 0.0001
p = 1.0

# Fully connected graph on 4 vertices
A = [
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0]
]

N_agents = len(A)


class AgentSO3(AgentSO2):
    def __init__(self, init):
        super().__init__(init)

    @staticmethod
    def get_grad(x, p):
        return NotImplementedError
    
    def update(self):
            vel = np.zeros_like(self.x)

            for n in self.neighbors:
                grad = AgentSO3.get_grad(NotImplementedError, p)
                vel -= A[self.id][n.id]*grad
            
            self.x += vel * timestep

def simulate():
    for i in range(N_agents):
        # AgentSO2(np.random.rand(1)[0]*np.pi*2 - np.pi)
        AgentSO2((2*np.pi/N_agents)*Agent.count + 0.2*(np.random.rand(1)[0]-0.5))

    for i in range(N_agents):
        for j in range(N_agents):
            if A[i][j] == 1:
                Agent.list[i].add_neighbor(deepcopy(Agent.list[j]))

    plot_vals = [[] for _ in range(N_agents)]
    for i in range(steps):
        [plot_vals[a.id].append(
            AgentSO2.angle_from_number(deepcopy(a.x))) 
            for a in Agent.list]    
        [a.communicate() for a in AgentSO3.list]
        [a.update() for a in AgentSO3.list]

    for i in range(N_agents):
        plt.plot(plot_vals[i], label = f'Agent {i}')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    simulate()
    # plot_stuff()