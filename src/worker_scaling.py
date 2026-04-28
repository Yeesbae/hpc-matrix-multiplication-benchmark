import time
import csv
import numpy as np
import implementations

MATRIX_SIZES = [16, 32, 64, 128, 256]
WORKERS = [1, 2, 4, 8]
REPEATS = 10
TOLERANCE = 1e-6

def create_matrices(size, seed):
    rng = np.random.default_rng(seed)

    A = rng.random((size, size))
    B = rng.random((size, size))
    
    if A.shape[1] != B.shape[0]:
        raise ValueError('Number of columns in A must be equal to number of rows in B.')
    else:
        return A, B

def run_benchmark(implementation, A, B):
    start = time.perf_counter()
    result = implementation(A, B)
    end = time.perf_counter()
    return result, (end - start)

def correctness_check(result, reference):
    max_error = np.max(np.abs(result - reference))
    correct = max_error < TOLERANCE
    return correct, max_error

def save_result(results):
    with open('results/raw/benchmark_workers_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['implementation', 'matrix_size', 'repeat', 'runtime_seconds', 'correct', 'max_error', 'workers'])
        writer.writerows(results)
    return 0

def main():
    print('Running benchmarks...')
    results = []
    for size in MATRIX_SIZES:
        print(f'Benchmarking size {size}...')
        for repeat in range(REPEATS):
            A, B = create_matrices(size, seed = 42 + size * 100 + repeat)
            reference = implementations.numpy_implementation(A, B)
            for workers in WORKERS:
                name = 'multiprocessing'
                impl = lambda A, B: implementations.multiprocessing_implementation(A, B, workers=workers)
                result, runtime = run_benchmark(impl, A, B)
                correct, max_error = correctness_check(result, reference)
                results.append((name, size, repeat, runtime, correct, max_error, workers))

    save_result(results)
    return 0

if __name__ == '__main__':
    main()
