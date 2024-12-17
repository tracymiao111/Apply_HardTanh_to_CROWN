import numpy as np
import matplotlib.pyplot as plt

def hardtanh(x, min_val=-1.0, max_val=1.0):
    """
    HardTanh activation function.
    
    Parameters:
    - x (array-like): Input array.
    - min_val (float): Minimum value to clip to.
    - max_val (float): Maximum value to clip to.
    
    Returns:
    - np.ndarray: Output after applying HardTanh.
    """
    return np.clip(x, min_val, max_val)

def plot_hardtanh(show_bounds=False, lb=-2, ub=2):
    """
    Plots the HardTanh activation function.
    
    Parameters:
    - show_bounds (bool): If True, plots upper and lower linear bounds.
    - lb (float): Lower bound of input range for bounds plotting.
    - ub (float): Upper bound of input range for bounds plotting.
    """
    # Define input range
    x = np.linspace(lb, ub, 1000)
    y = hardtanh(x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='HardTanh', color='black', linewidth=2)
    
    if show_bounds:
        upper_bound = 0.5 * x + 0.5
        lower_bound = -0.5 * x - 0.5
        
        plt.plot(x, upper_bound, label='Upper Bound', linestyle='--', color='red')
        plt.plot(x, lower_bound, label='Lower Bound', linestyle='--', color='blue')
        
        plt.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.2, label='Bounded Region')
    
    plt.axvline(x=-1, color='green', linestyle=':', linewidth=1)
    plt.axvline(x=1, color='green', linestyle=':', linewidth=1)
    plt.text(-1, hardtanh(-1) - 0.2, 'x = -1', color='green', fontsize=9, ha='right')
    plt.text(1, hardtanh(1) - 0.2, 'x = 1', color='green', fontsize=9, ha='left')
    
    plt.title('HardTanh Activation Function')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)
    plt.ylim([min(hardtanh(lb), min(y)) - 0.5, max(hardtanh(ub), max(y)) + 0.5])
    
    plt.show()

if __name__ == "__main__":
    plot_hardtanh(show_bounds=False)
    