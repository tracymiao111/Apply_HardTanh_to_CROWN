# crown.py

import torch
import torch.nn as nn
import numpy as np
from model import SimpleNNRelu, SimpleNNHardTanh
from linear import BoundLinear
from relu import BoundReLU
from hardTanh_question import BoundHardTanh
import time
import argparse


class BoundedSequential(nn.Sequential):
    r"""This class wraps the PyTorch nn.Sequential object with bound computation."""

    def __init__(self, *args):
        super(BoundedSequential, self).__init__(*args)

    @staticmethod
    def convert(seq_model):
        """Convert a PyTorch model to a model with bounds."""
        layers = []
        for l in seq_model:
            if isinstance(l, nn.Linear):
                layers.append(BoundLinear.convert(l))
            elif isinstance(l, nn.ReLU):
                layers.append(BoundReLU.convert(l))
            elif isinstance(l, nn.Hardtanh):
                layers.append(BoundHardTanh.convert(l))
        return BoundedSequential(*layers)

    def compute_bounds(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False):
        """Main function for computing bounds."""
        ub = lb = None
        ub, lb = self.full_boundpropogation(x_U=x_U, x_L=x_L, upper=upper, lower=lower, optimize=optimize)
        return ub, lb

    def full_boundpropogation(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False):
        """A full bound propagation."""
        modules = list(self._modules.values())
        
        # Propagate the 'optimize' flag to all BoundHardTanh layers
        for module in modules:
            if isinstance(module, BoundHardTanh):
                module.use_alpha = optimize

        # CROWN propagation for all layers
        for i in range(len(modules)):
            # We only need the bounds before a ReLU/HardTanh layer
            if isinstance(modules[i], BoundReLU) or isinstance(modules[i], BoundHardTanh):
                if isinstance(modules[i - 1], BoundLinear):
                    # Add a batch dimension and create an identity matrix
                    newC = torch.eye(modules[i - 1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
                    # Use CROWN to compute pre-activation bounds starting from layer i-1
                    ub, lb = self.boundpropogate_from_layer(
                        x_U=x_U, 
                        x_L=x_L, 
                        C=newC, 
                        upper=True, 
                        lower=True,
                        start_node=i - 1
                    )
                # Set pre-activation bounds for layer i (the ReLU/HardTanh layer)
                modules[i].upper_u = ub
                modules[i].lower_l = lb

        # Get the final layer bound
        final_layer = modules[-1]
        finalC = torch.eye(final_layer.out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
        ub, lb = self.boundpropogate_from_layer(
            x_U=x_U, 
            x_L=x_L,
            C=finalC, 
            upper=upper,
            lower=lower, 
            start_node=len(modules) - 1
        )
        return ub, lb

    def boundpropogate_from_layer(self, x_U=None, x_L=None, C=None, upper=False, lower=True, start_node=None):
        """The bound propagation starting from a given layer."""
        modules = list(self._modules.values()) if start_node is None else list(self._modules.values())[:start_node + 1]
        upper_A = C if upper else None
        lower_A = C if lower else None
        upper_sum_b = lower_sum_b = x_U.new_zeros((x_U.size(0), C.size(1)))
        
        for i, module in enumerate(reversed(modules)):
            upper_A, upper_b, lower_A, lower_b = module.boundpropogate(upper_A, lower_A, start_node)
            print(f"Layer {start_node - i} - upper_A shape: {upper_A.shape}")
            print(f"Layer {start_node - i} - upper_b shape: {upper_b.shape}")
            upper_sum_b = upper_b + upper_sum_b
            lower_sum_b = lower_b + lower_sum_b

        # Define a helper function to compute concrete bounds
        def _get_concrete_bound(A, sum_b, sign=-1):
            if A is None:
                return None
            A = A.view(A.size(0), A.size(1), -1)
            # A has shape (batch, specification_size, flattened_input_size)
            x_ub = x_U.view(x_U.size(0), -1, 1)
            x_lb = x_L.view(x_L.size(0), -1, 1)
            center = (x_ub + x_lb) / 2.0
            diff = (x_ub - x_lb) / 2.0
            bound = A.bmm(center) + sign * A.abs().bmm(diff)
            bound = bound.squeeze(-1) + sum_b
            return bound

        lb = _get_concrete_bound(lower_A, lower_sum_b, sign=-1)
        ub = _get_concrete_bound(upper_A, upper_sum_b, sign=+1)
        if ub is None:
            ub = x_U.new_full((x_U.size(0), C.size(1)), float('inf'))
        if lb is None:
            lb = x_L.new_full((x_L.size(0), C.size(1)), float('-inf'))
        return ub, lb


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--activation', default='relu', choices=['relu', 'hardtanh'],
                        type=str, help='Activation Function')
    parser.add_argument('--optimize', action='store_true', help='Use alpha bound strategy')
    parser.add_argument('--eps', type=float, default=0.08, help='Perturbation epsilon')
    parser.add_argument('data_file', type=str, help='input data, a tensor saved as a .pth file.')
    # Parse the command line arguments
    args = parser.parse_args()

    # Load the test data
    x_test, label = torch.load(args.data_file)

    # Load the appropriate model
    if args.activation == 'relu':
        print('Use ReLU model')
        model = SimpleNNRelu()
        model.load_state_dict(torch.load('models/relu_model.pth'))
    else:
        print('Use HardTanh model')
        model = SimpleNNHardTanh()
        model.load_state_dict(torch.load('models/hardtanh_model.pth'))

    # Prepare the input
    batch_size = x_test.size(0)
    x_test = x_test.reshape(batch_size, -1)
    output = model(x_test)
    y_size = output.size(1)
    print("Network prediction: {}".format(output))

    # Set perturbation
    eps = args.eps
    x_u = x_test + eps
    x_l = x_test - eps
    print(f"Verifying Perturbation - {eps}")

    # Start bound computation
    start_time = time.time()
    boundedmodel = BoundedSequential.convert(model)
    ub, lb = boundedmodel.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True, optimize=args.optimize)
    end_time = time.time()
    print(f"Bound computation time: {end_time - start_time:.2f} seconds")

    # Print the bounds
    for i in range(batch_size):
        for j in range(y_size):
            print(f'f_{j}(x_{i}): {lb[i][j].item():8.4f} <= f_{j}(x_{i}+delta) <= {ub[i][j].item():8.4f}')