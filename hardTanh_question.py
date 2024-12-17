import torch
import torch.nn as nn

class BoundHardTanh(nn.Hardtanh):
    def __init__(self):
        super(BoundHardTanh, self).__init__()

    @staticmethod
    def convert(act_layer):
        # Convert an nn.Hardtanh layer into BoundHardTanh
        l = BoundHardTanh()
        l.min_val = act_layer.min_val
        l.max_val = act_layer.max_val
        return l

    def _boundpropogate_alpha(self, last_uA, last_lA):
        """Bound propagation without alpha using the revised 6-case logic."""
        lb = self.lower_l  # (batch, neurons)
        ub = self.upper_u  # (batch, neurons)
        # ub = torch.max(ub, lb + 1e-8)

        # Initialize slopes and intercepts
        upper_d = torch.zeros_like(lb)
        upper_b = torch.zeros_like(lb)
        lower_d = torch.zeros_like(lb)
        lower_b = torch.zeros_like(lb)

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
        upper_b[c2] = -1.0 - ((ub[c2] + 1.0) / (ub[c2] - lb[c2]))*lb[c2]
        lower_d[c2] = (upper_d[c2] > 0.5).float()
        lower_b[c2] = -1.0 + lower_d[c2]

        # *** Revised Case 3: lb < -1, ub > 1 ***
        c3 = case3
        slope_c3 = 2.0 / (ub[c3] - lb[c3])  # Compute slope where case3 is True
        # Condition: slope > 0.5
        condition_c3 = slope_c3 > 0.5
        # If condition_c3 is True, set to slope; else, set to 0.0
        upper_d[c3] = torch.where(condition_c3, slope_c3, torch.zeros_like(slope_c3))
        lower_d[c3] = torch.where(condition_c3, slope_c3, torch.zeros_like(slope_c3))
        # Set upper_b and lower_b as before
        upper_b[c3] = 1.0 - upper_d[c3]
        lower_b[c3] = -1.0 + lower_d[c3]
        # *** End of Revised Case 3 ***

        # Case 4: -1 < lb < ub < 1
        # upper_d = 1, upper_b = 0
        # lower_d = 1, lower_b = 0
        c4 = case4
        upper_d[c4] = 1.0
        upper_b[c4] = 0.0
        lower_d[c4] = 1.0
        lower_b[c4] = 0.0

        # Case 5: -1 < lb < 1 < ub
        # upper_d = (1 - lb) / (ub - lb)
        # upper_b = 1 - ((1 - lb) / (ub - lb))
        # lower_d = (1 - lb) / (ub - lb)
        # lower_b = 1 - ((1 - lb) / (ub - lb)) * ub
        c5 = case5
        upper_d[c5] = (((1.0 - lb[c5]) / (ub[c5] - lb[c5])) > 0.5).float()
        upper_b[c5] = 1.0 - upper_d[c5]
        lower_d[c5] = (1.0 - lb[c5]) / (ub[c5] - lb[c5])
        lower_b[c5] = 1.0 - ((1.0 - lb[c5]) / (ub[c5] - lb[c5])) * ub[c5]

        # Case 6: 1 < lb < ub
        # upper_d = 0, upper_b = 1
        # lower_d = 0, lower_b = 1
        c6 = case6
        upper_d[c6] = 0.0
        upper_b[c6] = 1.0
        lower_d[c6] = 0.0
        lower_b[c6] = 1.0

        # Reshape for broadcasting
        upper_d = upper_d.unsqueeze(1)  # (batch, 1, neurons)
        upper_b = upper_b.unsqueeze(1)  # (batch, 1, neurons)
        lower_d = lower_d.unsqueeze(1)  # (batch, 1, neurons)
        lower_b = lower_b.unsqueeze(1)  # (batch, 1, neurons)

        # Initialize outputs
        uA = ubias = lA = lbias = None

        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0)
            neg_uA = last_uA.clamp(max=0)
            # Compute upper bound coefficients
            uA = upper_d * pos_uA + lower_d * neg_uA
            # Compute upper bias: sum over neurons
            ubias = (pos_uA * upper_b + neg_uA * lower_b).sum(dim=-1)

        if last_lA is not None:
            pos_lA = last_lA.clamp(min=0)
            neg_lA = last_lA.clamp(max=0)
            # Compute lower bound coefficients
            lA = upper_d * neg_lA + lower_d * pos_lA
            # Compute lower bias: sum over neurons
            lbias = (neg_lA * upper_b + pos_lA * lower_b).sum(dim=-1)

        return uA, ubias, lA, lbias

    def _boundpropogate_no_alpha(self, last_uA, last_lA):
        """Bound propagation without alpha using the revised 6-case logic."""
        lb = self.lower_l  # (batch, neurons)
        ub = self.upper_u  # (batch, neurons)

        # Initialize slopes and intercepts
        upper_d = torch.zeros_like(lb)
        upper_b = torch.zeros_like(lb)
        lower_d = torch.zeros_like(lb)
        lower_b = torch.zeros_like(lb)

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
        upper_b[c2] = -1.0 - ((ub[c2] + 1.0) / (ub[c2] - lb[c2]))*lb[c2]
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
        lower_b[c5] = 1.0 - ((1.0 - lb[c5]) / (ub[c5] - lb[c5])) * ub[c5]

        # Case 6: 1 < lb < ub
        c6 = case6
        upper_d[c6] = 0.0
        upper_b[c6] = 1.0
        lower_d[c6] = 0.0
        lower_b[c6] = 1.0

        # Reshape for broadcasting
        upper_d = upper_d.unsqueeze(1)  # (batch, 1, neurons)
        upper_b = upper_b.unsqueeze(1)  # (batch, 1, neurons)
        lower_d = lower_d.unsqueeze(1)  # (batch, 1, neurons)
        lower_b = lower_b.unsqueeze(1)  # (batch, 1, neurons)

        # Initialize outputs
        uA = ubias = lA = lbias = None

        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0)
            neg_uA = last_uA.clamp(max=0)
            # Compute upper bound coefficients
            uA = upper_d * pos_uA + lower_d * neg_uA
            # Compute upper bias: sum over neurons
            ubias = (pos_uA * upper_b + neg_uA * lower_b).sum(dim=-1)

        if last_lA is not None:
            pos_lA = last_lA.clamp(min=0)
            neg_lA = last_lA.clamp(max=0)
            # Compute lower bound coefficients
            lA = upper_d * neg_lA + lower_d * pos_lA
            # Compute lower bias: sum over neurons
            lbias = (neg_lA * upper_b + pos_lA * lower_b).sum(dim=-1)

        return uA, ubias, lA, lbias

    def boundpropogate(self, last_uA, last_lA, start_node=None):
        if self.use_alpha:
            return self._boundpropogate_alpha(last_uA, last_lA)
        else:
            return self._boundpropogate_no_alpha(last_uA, last_lA)
        