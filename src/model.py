"""
The neural network used by all three problems.

A PINN is just a normal fully connected network (an MLP):
    input  -> (x, t) or (x, y) or (t)
    hidden -> several layers with tanh activation
    output -> the physical quantity (pressure, temperature, displacement)

What makes it "physics-informed" is NOT the network, it's the loss function
(see the experiments/ files). This file only defines the network itself.
"""
import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_width, n_hidden, lb, ub,
                 init="glorot_normal", output_transform=None):
        """
        in_dim           : number of inputs, e.g. 2 for (x, t)
        out_dim          : number of outputs, e.g. 1 for p(x, t)
        hidden_width     : neurons per hidden layer (paper: 50 or 32)
        n_hidden         : number of hidden layers (paper: 5, 4 or 3)
        lb, ub           : lower / upper bound of each input, used to scale
                           inputs to [-1, 1] (helps tanh networks train)
        init             : "glorot_normal" or "glorot_uniform" (as in the paper)
        output_transform : optional function f(X, raw_output) -> final output.
                           Used to build conditions into the output, e.g.
                           consolidation multiplies the output by x so that
                           p(0, t) = 0 is always satisfied.
                           (Part 2 / Fix 1 also plugs in here.)
        """
        super().__init__()

        dims = [in_dim] + [hidden_width] * n_hidden
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.Tanh()]
        layers.append(nn.Linear(dims[-1], out_dim))  # linear output layer
        self.net = nn.Sequential(*layers)

        # Weight initialisation, exactly as described in the paper
        for m in self.net:
            if isinstance(m, nn.Linear):
                if init == "glorot_normal":
                    nn.init.xavier_normal_(m.weight)
                elif init == "glorot_uniform":
                    nn.init.xavier_uniform_(m.weight)
                else:
                    raise ValueError(f"Unknown init: {init}")
                nn.init.zeros_(m.bias)

        # Buffers are saved with the model but are not trained
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32))
        self.output_transform = output_transform

    def forward(self, X):
        # Scale every input column to [-1, 1]
        X_scaled = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        out = self.net(X_scaled)
        if self.output_transform is not None:
            out = self.output_transform(X, out)
        return out
