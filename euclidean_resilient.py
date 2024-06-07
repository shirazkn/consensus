import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

from functions.agent import Agent

steps = 5000
threshold = 0.0005

timestep = 0.0001
p = 0.9

# timestep = 0.001
# p = 2.0

deviation_speed = [0.2, 0.4]


# def random_connected_graph(sparsity_factor = 0.5):
#     A = np.ones([N_agents, N_agents], dtype = np.int)
#     for i in range(N_agents):
#         A[i, i] = 0

#     edges_to_remove = N_agents(N_agents-1)/2 * sparsity_factor
#     for edge_removed_ in range(edges_to_remove):
#         raise(NotImplementedError)


# Slightly asymmetric graph
A = [
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0]
]

N_agents = len(A)


def draw_graph(A):
    rows, cols = np.where(np.array(A) == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    G.add_edges_from(edges)
    nx.draw(G, with_labels = True)
    plt.show()

# draw_graph(A)


def simulate():
    for i in range(N_agents):
        Agent(np.random.rand(1)[0])

    for i in range(N_agents):
        for j in range(N_agents):
            if A[i][j] == 1:
                Agent.list[i].add_neighbor(deepcopy(Agent.list[j]))

    plot_vals = [[] for _ in range(N_agents)]
    for i in range(steps):
        if i == int(steps/2):
            Agent.list[0].update = Agent.list[0].bad_update
            Agent.list[1].update = Agent.list[1].bad_update
            Agent.list[2].update = Agent.list[2].fixed_update
            Agent.list[3].update = Agent.list[3].fixed_update
            Agent.list[4].update = Agent.list[4].fixed_update

        [plot_vals[a.id].append(deepcopy(a.x)) for a in Agent.list]    
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