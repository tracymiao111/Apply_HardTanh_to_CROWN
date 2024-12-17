# plot_bounds.py
import numpy as np
import matplotlib.pyplot as plt
import torch

def hardtanh(x):
    """HardTanh activation function."""
    return np.clip(x, -1, 1)

def compute_bounds_no_alpha(x, lb, ub):
    """Compute upper and lower bounds without alpha based on the 6-case logic."""
    upper_d = np.zeros_like(x)
    upper_b = np.zeros_like(x)
    lower_d = np.zeros_like(x)
    lower_b = np.zeros_like(x)
    
    # Define masks for each of the 6 cases
    case1 = (ub < -1)  # lb < ub < -1
    case2 = (lb < -1) & (ub < 1) & (ub >= -1)  # lb < -1 < ub < 1
    case3 = (lb < -1) & (ub > 1)  # lb < -1, ub > 1
    case4 = (lb > -1) & (ub < 1)  # -1 < lb < ub < 1
    case5 = (lb > -1) & (lb < 1) & (ub > 1)  # -1 < lb < 1 < ub
    case6 = (lb > 1)  # 1 < lb < ub

    # Case 1: lb < ub < -1
    upper_d[case1] = 0.0
    upper_b[case1] = -1.0
    lower_d[case1] = 0.0
    lower_b[case1] = -1.0

    # Case 2: lb < -1 < ub < 1
    c2 = case2
    upper_d[c2] = (ub[c2] + 1.0) / (ub[c2] - lb[c2])
    upper_b[c2] = -1.0 - ((ub[c2] + 1.0) / (ub[c2] - lb[c2])) * lb[c2]
    lower_d[c2] = (ub[c2] + 1.0) / (ub[c2] - lb[c2])
    lower_b[c2] = -1.0 + ((ub[c2] + 1.0) / (ub[c2] - lb[c2]))

    # Case 3: lb < -1, ub > 1
    c3 = case3
    upper_d[c3] = 2.0 / (ub[c3] - lb[c3])
    upper_b[c3] = 1.0 - (2.0 / (ub[c3] - lb[c3]))
    lower_d[c3] = 2.0 / (ub[c3] - lb[c3])
    lower_b[c3] = -1.0 + (2.0 / (ub[c3] - lb[c3]))

    # Case 4: -1 < lb < ub < 1
    c4 = case4
    upper_d[c4] = 1.0
    upper_b[c4] = 0.0
    lower_d[c4] = 1.0
    lower_b[c4] = 0.0

    # Case 5: -1 < lb < 1 < ub
    c5 = case5
    upper_d[c5] = (1.0 - lb[c5]) / (ub[c5] - lb[c5])
    upper_b[c5] = 1.0 - ((1.0 - lb[c5]) / (ub[c5] - lb[c5]))
    lower_d[c5] = (1.0 - lb[c5]) / (ub[c5] - lb[c5])
    lower_b[c5] = -1.0 + ((1.0 - lb[c5]) / (ub[c5] - lb[c5]))

    # Case 6: 1 < lb < ub
    c6 = case6
    upper_d[c6] = 0.0
    upper_b[c6] = 1.0
    lower_d[c6] = 0.0
    lower_b[c6] = 1.0

    # Compute upper and lower bounds
    upper_bound = upper_d * x + upper_b
    lower_bound = lower_d * x + lower_b

    return upper_bound, lower_bound

def compute_bounds_alpha(x, lb, ub):
    """Compute upper and lower bounds with alpha based on the revised 6-case logic."""
    upper_d = np.zeros_like(x)
    upper_b = np.zeros_like(x)
    lower_d = np.zeros_like(x)
    lower_b = np.zeros_like(x)
    
    # Define masks for each of the 9 cases
    case1 = (ub < -1)  # lb < ub < -1
    case2A = (lb < -1) & (ub < 1) & (ub >= -1) & (lb >= -2)  # Specific to Case 2A
    case2B = (lb < -1) & (ub < 1) & (ub >= -1) & (lb < -2)   # Specific to Case 2B
    case3A = (lb < -1) & (ub > 1) & (lb >= -1.5) & (ub <= 1.3)  # Case 3A
    case3B = (lb < -1) & (ub > 1) & ((lb < -1.5) | (ub > 1.3))     # Case 3B
    case4 = (lb > -1) & (ub < 1)  # -1 < lb < ub < 1
    case5A = (lb > -1) & (lb < 1) & (ub > 1) & (lb >= 0.5) & (ub <= 1.5)  # Case 5A
    case5B = (lb > -1) & (lb < 1) & (ub > 1) & ((lb < 0.5) | (ub > 1.5))  # Case 5B
    case6 = (lb > 1)  # 1 < lb < ub

    # Case 1: lb < ub < -1
    upper_d[case1] = 0.0
    upper_b[case1] = -1.0
    lower_d[case1] = 0.0
    lower_b[case1] = -1.0

    # Case 2A: Specific bounds for Case 2A
    c2A = case2A
    upper_d[c2A] = (ub[c2A] + 1.0) / (ub[c2A] - lb[c2A])
    upper_b[c2A] = -1.0 - ((ub[c2A] + 1.0) / (ub[c2A] - lb[c2A])) * lb[c2A]
    lower_d[c2A] = (upper_d[c2A] > 0.5).astype(float)
    lower_b[c2A] = -1.0 + lower_d[c2A]

    # Case 2B: Specific bounds for Case 2B
    c2B = case2B
    upper_d[c2B] = (ub[c2B] + 1.0) / (ub[c2B] - lb[c2B])
    upper_b[c2B] = -1.0 - ((ub[c2B] + 1.0) / (ub[c2B] - lb[c2B])) * lb[c2B]
    lower_d[c2B] = (upper_d[c2B] > 0.5).astype(float)
    lower_b[c2B] = -1.0 + lower_d[c2B]

    # Case 3A: lb < -1, ub > 1 with specific bounds
    c3A = case3A
    slope_c3A = 2.0 / (ub[c3A] - lb[c3A])
    condition_c3A = slope_c3A > 0.5
    upper_d[c3A] = np.where(condition_c3A, slope_c3A, 0.0)
    lower_d[c3A] = np.where(condition_c3A, slope_c3A, 0.0)
    upper_b[c3A] = 1.0 - upper_d[c3A]
    lower_b[c3A] = -1.0 + lower_d[c3A]

    # Case 3B: lb < -1, ub > 1 with different bounds
    c3B = case3B
    slope_c3B = 2.0 / (ub[c3B] - lb[c3B])
    condition_c3B = slope_c3B > 0.5
    upper_d[c3B] = np.where(condition_c3B, slope_c3B, 0.0)
    lower_d[c3B] = np.where(condition_c3B, slope_c3B, 0.0)
    upper_b[c3B] = 1.0 - upper_d[c3B]
    lower_b[c3B] = -1.0 + lower_d[c3B]

    # Case 4: -1 < lb < ub < 1
    c4 = case4
    upper_d[c4] = 1.0
    upper_b[c4] = 0.0
    lower_d[c4] = 1.0
    lower_b[c4] = 0.0

    # Case 5A: Specific bounds for Case 5A
    c5A = case5A
    slope5A = (1.0 - lb[c5A]) / (ub[c5A] - lb[c5A])
    upper_d[c5A] = (slope5A > 0.5).astype(float)
    upper_b[c5A] = 1.0 - upper_d[c5A]
    lower_d[c5A] = slope5A
    lower_b[c5A] = 1.0 - slope5A * ub[c5A]

    # Case 5B: Specific bounds for Case 5B
    c5B = case5B
    slope5B = (1.0 - lb[c5B]) / (ub[c5B] - lb[c5B])
    upper_d[c5B] = (slope5B > 0.5).astype(float)
    upper_b[c5B] = 1.0 - upper_d[c5B]
    lower_d[c5B] = slope5B
    lower_b[c5B] = 1.0 - slope5B * ub[c5B]

    # Case 6: 1 < lb < ub
    c6 = case6
    upper_d[c6] = 0.0
    upper_b[c6] = 1.0
    lower_d[c6] = 0.0
    lower_b[c6] = 1.0

    # Compute upper and lower bounds
    upper_bound = upper_d * x + upper_b
    lower_bound = lower_d * x + lower_b

    return upper_bound, lower_bound

def plot_bounds_separately_with_enhancements():
    """Plot HardTanh activation function with bounding lines for no alpha and alpha separately,
    including lb and ub markers on the HardTanh curve and subcases for alpha."""
    # Define input range
    x = np.linspace(-3, 3, 600)
    y = hardtanh(x)

    # Define lb and ub for each case
    # No Alpha: 6 cases
    no_alpha_scenarios = {
        'Case 1': {'lb': -2, 'ub': -1.5},
        'Case 2': {'lb': -2, 'ub': 0.5},
        'Case 3': {'lb': -2, 'ub': 2},
        'Case 4': {'lb': 0, 'ub': 0.8},
        'Case 5': {'lb': 0, 'ub': 2},
        'Case 6': {'lb': 1.5, 'ub': 2},
    }

    # Alpha: 9 cases (including subcases for Case 2, 3, and 5)
    alpha_scenarios = {
        'Case 1': {'lb': -2, 'ub': -1.5},
        'Case 2A': {'lb': -1.5, 'ub': 0.5},
        'Case 2B': {'lb': -2.8, 'ub': -0.5},
        'Case 3A': {'lb': -1.3, 'ub': 1.3},
        'Case 3B': {'lb': -2.8, 'ub': 2.8},
        'Case 4': {'lb': 0, 'ub': 0.8},
        'Case 5A': {'lb': 0.5, 'ub': 2.8},
        'Case 5B': {'lb': -0.8, 'ub': 2},
        'Case 6': {'lb': 1.5, 'ub': 2},
    }

    # Initialize two separate figures
    fig_no_alpha, axes_no_alpha = plt.subplots(2, 3, figsize=(18, 10))
    fig_no_alpha.suptitle('HardTanh Activation with Bounding Lines (No Alpha)', fontsize=16)

    fig_alpha, axes_alpha = plt.subplots(3, 3, figsize=(18, 12))  # Adjusted to 3x3 for 9 cases
    fig_alpha.suptitle('HardTanh Activation with Bounding Lines (Alpha)', fontsize=20)

    # Plot No Alpha Bounds
    for ax, (case, bounds) in zip(axes_no_alpha.flatten(), no_alpha_scenarios.items()):
        lb = bounds['lb']
        ub = bounds['ub']

        # Convert lb and ub to arrays matching x
        lb_array = np.full_like(x, lb)
        ub_array = np.full_like(x, ub)

        # Compute bounds for no alpha
        upper_no_alpha, lower_no_alpha = compute_bounds_no_alpha(x, lb_array, ub_array)

        # Define mask for [lb, ub]
        mask = (x >= lb) & (x <= ub)

        # Outside [lb, ub], set bounds to HardTanh
        upper_no_alpha[~mask] = y[~mask]
        lower_no_alpha[~mask] = y[~mask]

        # Plot HardTanh
        ax.plot(x, y, label='HardTanh', color='black', linewidth=2)

        # Plot bounds without alpha
        ax.plot(x, upper_no_alpha, label='Upper Bound (No Alpha)', linestyle='--', color='blue')
        ax.plot(x, lower_no_alpha, label='Lower Bound (No Alpha)', linestyle='--', color='blue')

        # Fill regions for better visualization within [lb, ub]
        ax.fill_between(x, lower_no_alpha, upper_no_alpha, where=mask, color='blue', alpha=0.1, label='Bound (No Alpha)')

        # Compute y positions for lb and ub on HardTanh
        y_lb = hardtanh(lb)
        y_ub = hardtanh(ub)

        # Mark lb and ub as dots on the HardTanh curve
        ax.scatter(lb, y_lb, color='green', marker='o', s=50, label='lb')
        ax.scatter(ub, y_ub, color='orange', marker='o', s=50, label='ub')

        # Annotate lb and ub
        ax.annotate('lb', xy=(lb, y_lb), xytext=(lb, y_lb - 0.5),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    color='green', fontsize=10, ha='center')
        ax.annotate('ub', xy=(ub, y_ub), xytext=(ub, y_ub - 0.5),
                    arrowprops=dict(arrowstyle='->', color='orange'),
                    color='orange', fontsize=10, ha='center')

        # Set plot details
        ax.set_title(f'{case}: lb={lb}, ub={ub}', fontsize=14)
        ax.set_xlabel('Input', fontsize=12)
        ax.set_ylabel('Activation / Bounds', fontsize=12)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-2, 2])
        ax.grid(True)

    # Adjust legends to appear only once for No Alpha
    handles_no_alpha, labels_no_alpha = axes_no_alpha.flatten()[0].get_legend_handles_labels()
    by_label_no_alpha = dict(zip(labels_no_alpha, handles_no_alpha))
    axes_no_alpha.flatten()[0].legend(by_label_no_alpha.values(), by_label_no_alpha.keys())

    # Plot Alpha Bounds
    for ax, (case, bounds) in zip(axes_alpha.flatten(), alpha_scenarios.items()):
        lb = bounds['lb']
        ub = bounds['ub']

        # Convert lb and ub to arrays matching x
        lb_array = np.full_like(x, lb)
        ub_array = np.full_like(x, ub)

        # Compute bounds with alpha
        upper_alpha, lower_alpha = compute_bounds_alpha(x, lb_array, ub_array)

        # Define mask for [lb, ub]
        mask = (x >= lb) & (x <= ub)

        # Outside [lb, ub], set bounds to HardTanh
        upper_alpha[~mask] = y[~mask]
        lower_alpha[~mask] = y[~mask]

        # Plot HardTanh
        ax.plot(x, y, label='HardTanh', color='black', linewidth=2)

        # Plot primary bounds with alpha
        ax.plot(x, upper_alpha, label='Upper Bound (Alpha)', linestyle=':', color='red')
        ax.plot(x, lower_alpha, label='Lower Bound (Alpha)', linestyle=':', color='red')

        # Fill regions for better visualization within [lb, ub]
        ax.fill_between(x, lower_alpha, upper_alpha, where=mask, color='red', alpha=0.1, label='Bound (Alpha)')

        # Plot alternative bounds for specific cases
        if case in ['Case 2A', 'Case 2B']:
            # Alternative Lower Bound: y = -1
            alternative_lower = np.full_like(x, -1.0)
            ax.plot(x, alternative_lower, label='Lower Bound (Alpha, Alt)', linestyle=':', color='green')
            ax.fill_between(x, lower_alpha, alternative_lower, where=mask, color='green', alpha=0.05, label='Bound (Alpha, Alt)')
        if case in ['Case 3A', 'Case 3B']:
            # Alternative Upper Bound: y = 1.0
            alternative_upper = np.full_like(x, 1.0)
            ax.plot(x, alternative_upper, label='Upper Bound (Alpha, Alt)', linestyle=':', color='purple')
            ax.fill_between(x, upper_alpha, alternative_upper, where=mask, color='purple', alpha=0.05, label='Bound (Alpha, Alt)')
        if case in ['Case 5A', 'Case 5B']:
            # Alternative Upper Bound: y = 1.0
            alternative_upper = np.full_like(x, 1.0)
            ax.plot(x, alternative_upper, label='Upper Bound (Alpha, Alt)', linestyle=':', color='brown')
            ax.fill_between(x, upper_alpha, alternative_upper, where=mask, color='brown', alpha=0.05, label='Bound (Alpha, Alt)')

        # Compute y positions for lb and ub on HardTanh
        y_lb = hardtanh(lb)
        y_ub = hardtanh(ub)

        # Mark lb and ub as dots on the HardTanh curve
        ax.scatter(lb, y_lb, color='green', marker='o', s=50, label='lb')
        ax.scatter(ub, y_ub, color='orange', marker='o', s=50, label='ub')

        # Annotate lb and ub
        ax.annotate('lb', xy=(lb, y_lb), xytext=(lb, y_lb - 0.5),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    color='green', fontsize=10, ha='center')
        ax.annotate('ub', xy=(ub, y_ub), xytext=(ub, y_ub - 0.5),
                    arrowprops=dict(arrowstyle='->', color='orange'),
                    color='orange', fontsize=10, ha='center')

        # Set plot details
        ax.set_title(f'{case}: lb={lb}, ub={ub}', fontsize=14)
        ax.set_xlabel('Input', fontsize=12)
        ax.set_ylabel('Activation / Bounds', fontsize=12)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-2, 2])
        ax.grid(True)

    # Adjust legends to appear only once for Alpha
    handles_alpha, labels_alpha = axes_alpha.flatten()[0].get_legend_handles_labels()
    by_label_alpha = dict(zip(labels_alpha, handles_alpha))
    axes_alpha.flatten()[0].legend(by_label_alpha.values(), by_label_alpha.keys())

    # Adjust layout and display
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    plot_bounds_separately_with_enhancements()