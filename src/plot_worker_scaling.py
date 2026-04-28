import matplotlib.pyplot as plt
import pandas as pd

def load_results(file_path):
    return pd.read_csv(file_path)

def process_results(df):
    df2 = df.groupby(['matrix_size', 'workers'])['runtime_seconds'].mean().rename('average_runtime_seconds').reset_index()
    baseline = df2[df2['workers'] == 1].set_index('matrix_size')['average_runtime_seconds']
    df2['baseline_runtime'] = df2['matrix_size'].map(baseline)
    df2['speedup'] = df2['baseline_runtime'] / df2['average_runtime_seconds']
    return df2

def plot_results(results):
    plt.figure()
    plt.title('Multiprocessing Runtime vs Number of Workers')
    plt.xlabel('Number of Workers')
    plt.ylabel('Average Runtime (s)')
    plt.yscale('log')
    plt.grid(True)
    for size in results['matrix_size'].unique():
        subset = results[results['matrix_size'] == size]
        plt.plot(subset['workers'], subset['average_runtime_seconds'], label=f'Matrix Size {size}', marker='o')
    plt.legend()
    plt.savefig('./results/plots/multiprocessing_workers_plot.png')
    plt.show()
    
    plt.figure()
    plt.title('Speedup of Implementations (varying workers)')
    plt.xlabel('Number of Workers')
    plt.ylabel('Speedup (x)')
    plt.yscale('log')
    plt.grid(True)
    for size in results['matrix_size'].unique():
        subset = results[results['matrix_size'] == size]
        plt.plot(subset['workers'], subset['speedup'], label=f'Matrix Size {size}', marker='o')
    plt.legend()
    plt.savefig('./results/plots/speedup_workers_plot.png')
    plt.show()

    
def main():
    results = load_results('./results/raw/benchmark_workers_results.csv')
    processed_results = process_results(results)
    # print('Processed Results:', processed_results)
    plot_results(processed_results)
    return 0

if __name__ == '__main__':
    main()
