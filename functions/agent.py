import numpy as np
from deepcopy import deepcopy


class Agent:
    count = 0
    list = []

    def __init__(self, init):
        self.id = Agent.count
        Agent.count += 1

        self.neighbors = []
        self.x = deepcopy(init)
        Agent.list.append(self)

    def add_neighbor(self, neighbor_copy):
        self.neighbors.append(neighbor_copy)

    @staticmethod
    def get_pnorm_grad(disp, p, threshold):
        if np.abs(disp) > threshold:
            grad = p*np.abs(disp)**(p-1)*np.sign(disp)
        else:
            grad = p*threshold**(p-2)*disp
            
        return grad

    def update(self):
        vel = np.zeros_like(self.x)

        # Potential = |n.x - self.x|^p
        for n in self.neighbors:
            grad = Agent.get_pnorm_grad(self.x - n.x, p, threshold)
            vel -= A[self.id][n.id]*grad
        
        self.x += vel * timestep

    def communicate(self):
        for n in self.neighbors:
            n.x = deepcopy(Agent.list[n.id].x)

    def bad_update(self):        
        self.x += deviation_speed[self.id] * timestep

    def fixed_update(self):        
        self.x = self.x