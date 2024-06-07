import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt


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


def random_unit_vector():
    x = np.random.normal(size = 3)
    return x/np.linalg.norm(x)

def random_skew_symmetric_matrix():
    u = random_unit_vector()
    return u[0]*E1 + u[1]*E2 + u[2]*E3

def get_one_param_subgroup(X, num_elem):
    return_list = []
    for t in np.linspace(-np.pi, np.pi, num_elem):
        return_list.append(spl.expm(t*X))

    return return_list

def lie_bracket(A, B):
    return (A @ B - B @ A)

p = 0.5

# Check the potential function
def check_potential():
    num_elem = 1000
    Y = random_skew_symmetric_matrix()
    ops = get_one_param_subgroup(Y, num_elem)
    plot_y_vals = []

    # Trace along OPS
    for i in range(num_elem):
        plot_y_vals.append((np.trace(np.identity(3) - ops[i]))**p)
    plt.plot(plot_y_vals, label = r'$Trace(I-R)$')

    # 'Hessian' along a random direction
    X = random_skew_symmetric_matrix()
    # X = Y
    I = np.identity(3)
    for i in range(num_elem):
        R = ops[i]
        plot_y_vals.append(p*(p-1)*(np.trace(I - R))**(p-1)*(np.trace(-R @ X))**2
                        + p*np.trace(-R @ X @ X)
                        )
    plt.plot(plot_y_vals, label = r'$\langle Hess V (X), X \rangle_R$')

    plt.show()
