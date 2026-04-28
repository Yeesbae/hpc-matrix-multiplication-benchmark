import numpy as np
import multiprocessing as mp

# Naive python implementation of matrix multiplication
def naive_python_implementation(A, B):
  if len(A[0]) != len(B):
    raise ValueError("Number of columns in A must be equal to number of rows in B.") 

  C = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
  for i in range(len(A)):
      for j in range(len(B[0])):
        for k in range(len(A[0])):
          C[i][j] += A[i][k] * B[k][j]
  
  return np.array(C)

# Numpy implementation
def numpy_implementation(A, B):
    A = np.array(A)
    B = np.array(B)

    if A.shape[1] != B.shape[0]:
        raise ValueError("Number of columns in A must be equal to number of rows in B.")
  
    return np.dot(A, B)

def multiprocessing_implementation(A, B, workers=2):
    A = np.array(A)
    B = np.array(B)
    
    if A.shape[1] != B.shape[0]:
        raise ValueError("Number of columns in A must be equal to number of rows in B.")
    
    A_parts = np.array_split(A, workers)
    tasks = []
    
    for chunk in A_parts:
        tasks.append((chunk, B))
    with mp.Pool(processes=workers) as pool:
        results = pool.starmap(naive_python_implementation, tasks)
    
    C = np.concatenate(results)
    
    return C
