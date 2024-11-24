import numpy as np
import argparse


def input_data(file_paths=None):
    if file_paths:
        matrices = []
        for path in file_paths:
            try:
                matrices.append(np.loadtxt(path))
            except Exception as err:
                print(f"Ошибка чтения файла: {err}\nИспользуются значения по умолчанию")
                break
    else:
        matrices = [
            np.array([[0, 1  , 1, 1, 0],
                      [0, 0  , 0, 0, 0.5],
                      [0, 1  , 0, 0, 0],
                      [0, 1  , 1, 0, 0],
                      [1, 0.5, 1, 1, 0]]),

            np.array([[0, 1  , 1, 1, 1],
                      [0, 0  , 1, 1, 0.5],
                      [0, 0  , 0, 1, 0],
                      [0, 0  , 0, 0, 1],
                      [0, 0.5, 1, 0, 0]]),

            np.array([[0, 1  , 1, 1, 0],
                      [0, 0  , 1, 1, 0.5],
                      [0, 0  , 0, 1, 1],
                      [0, 0  , 0, 0, 1],
                      [1, 0.5, 0, 0, 0]]),

            np.array([[0  , 1  , 1  , 0.5, 0.5],
                      [0  , 0  , 0.5, 0.5, 1],
                      [0  , 0.5, 0  , 0.5, 0.5],
                      [0.5, 0.5, 0.5, 0  , 0.5],
                      [0.5, 0  , 0.5, 0.5, 0]]),
        ]
    return matrices


def aggregate(matrices):
    n = len(matrices[0])
    aggregated = np.zeros((n, n))
    for matrix in matrices:
        aggregated += matrix
    return aggregated / len(matrices)
    

def calculate(mtx, threshold=0.01):
    n = mtx.shape[0]
    row_sums = np.sum(mtx, axis=1)
    vector = row_sums / np.sum(row_sums)    
    order = len(vector)
    for i in range(1, len(vector)):
        diff = abs(vector[i] - vector[i - 1])
        if diff < threshold:
            order = i
            break
    return {"vector": vector, "order": order}


def main(file_paths=None):
    data = input_data(file_paths)
    mtx = aggregate(data)
    res = calculate(mtx)
    print("Вектор относительной важности объектов:", res["vector"])
    print("Целесообразный порядок t:", res["order"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Парные сравнения для определения важности объектов.")
    parser.add_argument("-f", "--files", nargs="+", help="Пути к файлам с матрицами предпочтений")
    args = parser.parse_args()
    main(args.files)
