import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

from euclidean_resilient import Agent

steps = 10000
timestep = 0.01
p = 0.5

# Circle graph
A = [
[0, 1, 0, 0, 1],
[1, 0, 1, 0, 0],
[0, 1, 0, 1, 0],
[0, 0, 1, 0, 1],
[1, 0, 0, 1, 0]
]

N_agents = len(A)

def draw_graph(A):
    rows, cols = np.where(np.array(A) == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    G.add_edges_from(edges)
    nx.draw(G, with_labels = True)
    plt.show()

draw_graph(A)


class AgentSO2(Agent):
    def __init__(self, init):
        super().__init__(init)

    @staticmethod
    def angle_from_number(x):
        if x > np.pi:
            return x - 2*np.pi
        elif x < -np.pi:
            return x + 2*np.pi
        else:
            return x

    @staticmethod       
    def get_grad(x, p):
        return p*(1-np.cos(x))**(p-1)*np.sin(x)

    def update(self):
        vel = np.zeros_like(self.x)

        # Potential = |1 - cos(x)|^p
        for n in self.neighbors:
            grad = AgentSO2.get_grad(self.x - n.x, p)
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
        [a.communicate() for a in Agent.list]
        [a.update() for a in Agent.list]

    for i in range(N_agents):
        plt.plot(plot_vals[i], label = f'Agent {i}')

    plt.legend()
    plt.show()

# def plot_stuff():
#     distances = np.linspace(-0.5, 0.5, 10000)
#     potentials = [np.abs(d)**p if np.abs(d) > threshold else 0.5*p*threshold**(p-2)*d**2
#                 for d in distances]
#     gradients = [Agent.get_pnorm_grad(d, p, threshold) for d in distances]
#     plt.plot(distances, potentials, label = 'Potential')
#     plt.plot(distances, gradients, label = 'Gradient')
#     plt.xlabel('Displacement')
#     plt.ylabel('Gradient')
#     plt.legend()
#     plt.show()

if __name__ == '__main__':
    simulate()
    # plot_stuff()