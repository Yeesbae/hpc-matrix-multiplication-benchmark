# Matrix Multiplication Performance Benchmark

## Overview

This project benchmarks different matrix multiplication implementations and compares their correctness and performance.

The goal is to understand how implementation choices affect runtime, scaling behaviour, and speedup as matrix size increases.

## Motivation

Matrix multiplication is a core operation in scientific computing, machine learning, numerical simulation, and high performance computing.

This project uses it as a first HPC-style benchmark to practice:

- correctness testing
- performance measurement
- experiment automation
- runtime comparison
- scaling analysis

## Implementations

This project currently includes:

1. Naive Python implementation  
   - Uses explicit triple nested loops
   - Serves as the baseline implementation

2. NumPy implementation  
   - Uses optimized library-backed matrix multiplication
   - Serves as the optimized reference implementation

Planned:

3. Multiprocessing implementation  
   - Splits matrix computation across CPU workers
   - Used to study parallel overhead and scaling

## Repository Structure

```text
hpc-matrix-multiplication-benchmark/
├── src/
│   ├── implementations.py
│   ├── benchmark.py
│   └── plot_results.py
├── tests/
│   └── test_implementations.py
├── experiments/
├── scripts/
├── results/
│   ├── raw/
│   ├── logs/
│   └── plots/
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

Create and activate a virtual environment:

```sh
uv venv
source .venv/bin/activate
```

Install dependencies:

```sh
uv pip install -r requirements.txt
```

## Running Tests
Run the correctness tests:

```sh
python -m unittest discover -s tests -v
```

The tests verify that the implementations produce correct results for:

- square matrices
- rectangular matrices
- identity matrices
- zero matrices

## Running Benchmarks

Run the benchmark:

```sh
python -m src.benchmark
```

Results are saved to:

```text
results/raw/benchmark_results.csv
```

## Plotting Results

Generate comparison and speedup plots:

```sh
python src/plot_results.py
```

Plots are saved to:

```text
results/plots/comparison_plot.png
results/plots/speedup_plot.png
```

## Benchmark Design

The benchmark will compare implementations by varying matrix size.

For each matrix size:

1. Generate input matrices
2. Run each implementation
3. Measure execution time
4. Repeat multiple times
5. Validate correctness against NumPy
6. Save results to CSV

Planned metrics:

- runtime
- average runtime
- speedup versus naive baseline
- correctness flag

## Results

Outputs:

- `results/raw/benchmark_results.csv` — raw timing data
- `results/plots/comparison_plot.png` — average runtime vs matrix size (log scale)
- `results/plots/speedup_plot.png` — speedup ratio (naive_python / other) vs matrix size (only has Numpy for now)

## Initial Observations

At this stage:

- naive Python is expected to be much slower because it performs explicit Python-level loops
- NumPy is expected to be significantly faster because it uses optimized low-level numerical routines
- multiprocessing may not always improve performance for small matrices due to process overhead

## Future Improvements

Planned extensions:

- add multiprocessing implementation
- add worker scaling experiments
- add larger benchmark configurations
- add runtime plots (done)
- add speedup plots (done)
- add SLURM-style job script
- add optional GPU implementation using PyTorch or CUDA
