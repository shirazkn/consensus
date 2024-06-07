import numpy as np
import scipy.linalg as spl

def get_bases(dim, i):
    mat = np.zeros([dim, dim])
    row = dim - 2
    col = dim - 1
    parity = (-1)**i
    while (i-1) > 0:
        if col == row + 1:
            row -= 1
            col = dim - 1
        else:
            col -= 1
        i -= 1

    mat[row, col] = parity*1
    mat[col, row] = -parity*1

    return mat

def random_unit_vector(dim):
    x = np.random.normal(size = dim)
    return x/np.linalg.norm(x)

def random_skew_symmetric_matrix(dim):
    dim_skew = dim_skew(dim)
    u = random_unit_vector(dim_skew)
    mat = np.zeros([dim, dim])
    for i in range(dim_skew):
        mat += u[i]*get_bases(dim, i+1)
    return mat


def dim_skew(dim):
    return int(dim*(dim-1)/2)

def vee(mat):
    dim = mat.shape[0]
    vec = np.zeros([dim_skew(dim), 1])
    row = dim - 2
    col = dim - 1
    for i in range(dim_skew(dim)):
        vec[i, 0] = mat[row, col]*((-1)**i)
        if col == row + 1:
            row -= 1
            col = dim - 1
        else:
            col -= 1

    return vec

def frobenius_ip(A, B):
    return np.trace(A.T @ B)

def lie_bracket(A, B):
    return (A @ B - B @ A)

def get_one_param_subgroup(X, num_elem):
    return_list = []
    for t in np.linspace(-np.pi, np.pi, num_elem):
        return_list.append(spl.expm(t*X))

    return return_list