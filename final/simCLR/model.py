import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class SimCLR(nn.Module):

    def __init__(self,projection_dim, n_features):
        super(SimCLR, self).__init__()

        self.n_features = n_features

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, h_i, h_j):
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return z_i, z_j