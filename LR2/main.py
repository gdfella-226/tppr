import numpy as np
from itertools import permutations

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

def generalize(x, exp_c):
    n, m = x.shape
    weighted_rankings = np.dot(x, exp_c)
    
    best_rank = None
    best_distance = float('inf')

    for perm in permutations(range(1, n + 1)):
        distance = sum(
            exp_c[i] * sum(abs(perm[j] - x[j, i]) for j in range(n))
            for i in range(m)
        )
        if distance < best_distance:
            best_distance = distance
            best_rank = perm

    return np.argsort(weighted_rankings) + 1, best_rank

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

    k = expert_comp(x)
    kc = kendall_corr(x)
    sc = spearman_corr(x)
    g = generalize(x, k)
    
    print(f'Коэффициенты компетентности экспертов: {k}')
    print(f'Средняя выборка: {g[0]}')
    print(f'Медиана: {g[1]}')
    print(f'Коэффициент конкордации Кендалла: {kc}')
    print(f'Коэффициент ранговой корелляции Спирмена: \n{sc}')
    