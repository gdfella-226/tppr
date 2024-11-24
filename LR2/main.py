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

def kendall_corr(x):
    n, m = x.shape
    sums = np.sum(x, axis=1)
    s = np.sum((sums - np.mean(sums))**2)
    w = (12 * s) / (m**2 * (n**3 - n))
    return w

def spearman_corr(x):
    n, m = x.shape
    correlations = np.zeros((m, m))

    for i in range(m-1):
        for j in range(i + 1, m):
            _sum = 0
            for k in range(n):
                _sum += (x[k, i] - x[k, j])**2
            corr = 1 - (6 * _sum)/(n**3 - n)
            correlations[i, j] = corr
            correlations[j, i] = corr

    return correlations


if __name__ == '__main__':
    x = np.array([
        [1, 8, 1, 1],
        [2, 7, 2, 5],
        [3, 6, 5, 3],
        [4, 5, 4, 4],
        [5, 4, 8, 8],
        [6, 3, 7, 7],
        [7, 2, 6, 6],
        [8, 1, 3, 2]
    ])
    #k = expert_comp(x)
    #kc = kendall_corr(x)
    sc = spearman_corr(x)
    print(sc)