# compare_bounds.py

import subprocess
import argparse
import re
import csv
import sys
import os
from tqdm import tqdm
import numpy as np

def run_crown(activation, data_file, eps, optimize):
    """Run crown.py with specified parameters and capture the output."""
    # Use sys.executable to ensure the same Python interpreter is used
    cmd = [
        sys.executable, 'crown.py',
        '-a', activation,
        '--eps', str(eps),
        data_file
    ]
    if optimize:
        cmd.append('--optimize')
    
    # Set locale environment variables to prevent locale errors
    env = os.environ.copy()
    env['LC_ALL'] = 'en_US.UTF-8'
    env['LANG'] = 'en_US.UTF-8'
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, env=env)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print("Standard Output:", e.stdout)
        print("Standard Error:", e.stderr)
        sys.exit(1)

def parse_bounds(output):
    """Parse the bounds from the output of crown.py."""
    pattern = re.compile(r'f_(\d+)\(x_(\d+)\):\s+([-+]?\d*\.\d+|\d+)\s+<=\s+f_\d+\(x_\d+\+delta\)\s+<=\s+([-+]?\d*\.\d+|\d+)')
    bounds = []
    for line in output.split('\n'):
        match = pattern.match(line.strip())
        if match:
            f_j = int(match.group(1))
            x_i = int(match.group(2))
            lb = float(match.group(3))
            ub = float(match.group(4))
            bounds.append((f_j, x_i, lb, ub))
    return bounds

def compute_average_width(bounds):
    """Compute the average bound width from parsed bounds."""
    if not bounds:
        return float('inf')
    total_width = 0.0
    for (_, _, lb, ub) in bounds:
        total_width += (ub - lb)
    average_width = total_width / len(bounds)
    return average_width

def main():
    parser = argparse.ArgumentParser(description="Compare CROWN bounds with and without optimization over varying epsilons.")
    parser.add_argument('-a', '--activation', default='hardtanh', choices=['relu', 'hardtanh'],
                        type=str, help='Activation Function')
    parser.add_argument('--data_file', type=str, required=True, help='Input data file (.pth)')
    parser.add_argument('--max_eps', type=float, default=0.01, help='Maximum epsilon value')
    parser.add_argument('--step_eps', type=float, default=0.001, help='Step size for epsilon')
    parser.add_argument('--output_csv', type=str, default='comparison_results_data2.csv', help='Output CSV file for results')
    args = parser.parse_args()

    # Generate eps values from 0 to max_eps inclusive
    num_steps = int(args.max_eps / args.step_eps) + 1
    eps_values = [round(i * args.step_eps, 4) for i in range(num_steps)]

    # Open CSV file to write results
    with open(args.output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['eps', 'optimize', 'average_bound_width']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for eps in tqdm(eps_values, desc="Processing epsilons"):
            for optimize in [False, True]:
                output = run_crown(args.activation, args.data_file, eps, optimize)
                bounds = parse_bounds(output)
                avg_width = compute_average_width(bounds)
                writer.writerow({
                    'eps': eps,
                    'optimize': optimize,
                    'average_bound_width': avg_width
                })
                status = "Optimized" if optimize else "Non-Optimized"
                print(f"eps: {eps:.4f}, {status}, Average Bound Width: {avg_width:.4f}")

    print(f"\nComparison results saved to {args.output_csv}")

if __name__ == "__main__":
    main()