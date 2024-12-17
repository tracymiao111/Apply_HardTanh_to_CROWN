# visualize_results.py

import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualize comparison results.")
    parser.add_argument('--csv', type=str, default='comparison_results.csv', help='Input CSV file with results.')
    parser.add_argument('--output_plot', type=str, default='comparison_results_4.png', help='Output plot filename.')
    args = parser.parse_args()


    df = pd.read_csv(args.csv)

    non_opt = df[df['optimize'] == False]
    opt = df[df['optimize'] == True]

    plt.figure(figsize=(10, 6))
    plt.plot(non_opt['eps'], non_opt['average_bound_width'], label='Non-Optimized', marker='o')
    plt.plot(opt['eps'], opt['average_bound_width'], label='Optimized', marker='x')
    plt.xlabel('Perturbation Epsilon (eps)')
    plt.ylabel('Average Bound Width')
    plt.title('Comparison of Bound Tightness: Optimized vs Non-Optimized')
    plt.legend()
    plt.grid(True)
    plt.savefig(args.output_plot)
    plt.show()

if __name__ == "__main__":
    main()