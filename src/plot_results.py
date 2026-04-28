import matplotlib.pyplot as plt
import pandas as pd

def load_results(file_path):
    return pd.read_csv(file_path)

def process_results(df):
    implementation = df['implementation'].unique().tolist()
    runtime = {}
    
    for impl in implementation:
        runtime[f'{impl}'] = {
            "matrix_size": df[df['implementation'] == impl]['matrix_size'].tolist(),
            "runtime_seconds": df[df['implementation'] == impl]['runtime_seconds'].tolist(),
            "matrix_size_grouped": df[df['implementation'] == impl]['matrix_size'].unique().tolist(),
            "average_runtime_seconds": df[df['implementation'] == impl].groupby('matrix_size')['runtime_seconds'].mean().tolist(),
        }
    
    base_impl = 'naive_python'
    speedup = {"matrix_size_grouped": runtime[base_impl]['matrix_size_grouped']}
    for impl in implementation[1:]:
        base_avg = runtime[base_impl]['average_runtime_seconds']
        impl_avg = runtime[impl]['average_runtime_seconds']
        speedup[f'{base_impl}/{impl}'] = [b / i for b, i in zip(base_avg, impl_avg)]
    
    return runtime, speedup

def plot_results(results, comparison):
    plt.figure()
    plt.title("Average Runtime vs Matrix Size")
    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time (s)")
    plt.yscale('log')
    plt.grid(True)
    for impl, data in results.items():
        # plt.plot(data["matrix_size"], data["runtime_seconds"], label=f"{impl}", marker='o')
        plt.plot(data["matrix_size_grouped"], data["average_runtime_seconds"], label=f"{impl} (average)", linestyle='--')
    plt.legend()
    plt.savefig("./results/plots/comparison_plot.png")
    plt.show()
    
    plt.figure()
    plt.title("Speedup of Implementations (fixed workers)")
    plt.xlabel("Matrix Size")
    plt.ylabel("Speedup")
    plt.yscale('log')
    plt.grid(True)
    for impl, data in comparison.items():
        if impl != "matrix_size_grouped":
            plt.plot(comparison["matrix_size_grouped"], data, label=f"{impl}")
    plt.legend()
    plt.savefig("./results/plots/speedup_plot.png")
    plt.show()

    
def main():
    results = load_results("./results/raw/benchmark_results.csv")
    processed_results, speedup = process_results(results)
    print("Processed Results:", processed_results)
    print("Speedup:", speedup)
    plot_results(processed_results, speedup)
    return 0

if __name__ == "__main__":
    main()
