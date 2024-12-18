import argparse
import scipy.stats
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

def spearman_corr(x, tr_p):
    n, m = x.shape
    correlations = np.zeros((m, m))

    for i in range(m-1):
        for j in range(i + 1, m):
            _sum = 0
            for k in range(n):
                _sum += (x[k, i] - x[k, j])**2
            corr = 1 - (6 * _sum)/(n**3 - n)
            
            df = n - 2
            t_stat = corr * ((n - 2) ** 0.5) / ((1 - corr**2) ** 0.5)
            if 2 * scipy.stats.t.sf(abs(t_stat), df) < 1 - tr_p:
                correlations[i, j] = corr
                correlations[j, i] = corr
            else:
                correlations[i, j] = 0
                correlations[j, i] = 0
    return correlations

def generalize(x, exp_c):
    n, m = x.shape
    #weighted_rankings = np.dot(x, exp_c)
    
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

    sq_best_rank = None
    best_distance = float('inf')

    for perm in permutations(range(1, n + 1)):
        sq_distance = sum(
            exp_c[i] * sum((abs(perm[j] - x[j, i]))**2 for j in range(n))
            for i in range(m)
        )
        if sq_distance < best_distance:
            best_distance = sq_distance
            sq_best_rank = perm

    #return np.argsort(weighted_rankings) + 1, best_rank
    return sq_best_rank, best_rank


def correct(rankings, weights):
    avg_rank, median_rank = generalize(rankings, weights)
    new_rankings = rankings.copy()
    for i in range(rankings.shape[1]):
        new_rankings[:, i] = 0.8 * rankings[:, i] + 0.2 * np.array(median_rank)

    return new_rankings



def gaz(mtx, tr_p=0.95):
    
    k = expert_comp(mtx)
    kc = kendall_corr(mtx)
    sc = spearman_corr(mtx, float(tr_p))
    g = generalize(mtx, k)

    
    return k, kc, sc, g

def input_data(file_paths="./data/l22.txt"):
    if file_paths:
        for path in file_paths:
            try:
                mtx = np.loadtxt(path)
                return mtx
            except Exception as err:
                print(f"Ошибка чтения файла: {err}\nИспользуются значения по умолчанию")
                break
    mtx = np.array([
        [1, 8, 1, 1],
        [2, 7, 2, 5],
        [3, 6, 5, 3],
        [4, 5, 4, 4],
        [5, 4, 8, 8],
        [6, 3, 7, 7],
        [7, 2, 6, 6],
        [8, 1, 3, 2]
    ])
    return mtx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Оценка ранжированием.")
    parser.add_argument("-f", "--file", nargs="+", help="Путь к файлу с матрицами предпочтений")
    parser.add_argument("-tp", "--trustprob",  help="Доверительная вероятность")
    args = parser.parse_args()

    y = np.array([
        [1, 1],
        [2, 5],
        [5, 3],
        [4, 4],
        [8, 8],
        [7, 7],
        [6, 6],
        [3, 2]
    ])
    mtx = input_data(args.file)
    k, kc, sc, g = gaz(mtx, args.trustprob) 
    
    print(f'Коэффициенты компетентности экспертов: {k}')
    print(f'Средняя выборка: {g[0]}')
    print(f'Медиана: {g[1]}')
    print(f'Коэффициент конкордации Кендалла: {kc}')
    print(f'Коэффициент ранговой корелляции Спирмена: \n{sc}')
    