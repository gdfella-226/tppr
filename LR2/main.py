import numpy as np

def expert_comp(x):
    n, m = x.shape
    k_t = np.ones(m) / m
    for t in range(1000):
        x_t = np.array([np.sum(x[i, :] * k_t) for i in range(n)])
        lambda_t = np.sum([x[i, j] * x_t[i] for i in range(n) for j in range(m)])
        k_t_new = np.array([np.sum(x[:, j] * x_t) / lambda_t for j in range(m)])
        k_t_new /= np.sum(k_t_new)
        if np.allclose(k_t, k_t_new, atol=1e-6):
            break
        k_t = k_t_new

    return k_t

if __name__ == '__main__':
    y = np.array([
        [1, 8, 1, 1],
        [2, 7, 2, 8],
        [3, 6, 8, 3],
        [4, 5, 4, 4],
        [5, 4, 3, 2],
        [6, 3, 7, 7],
        [7, 2, 6, 6],
        [8, 1, 5, 5]
    ])
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                 [8, 7, 6, 5, 4, 3, 2, 1],
                 [1, 2, 8, 4, 3, 7, 6, 5],
                 [1, 8, 3, 4, 2, 7, 6, 5]])
    k = expert_comp(y)
    print(np.sum(k), k)