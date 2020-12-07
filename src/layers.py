import torch
import torch.nn as nn


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

class FCPlanenet(nn.Module):
    """
    For reducing point clouds we use SimplePointNet to reduce to c_dim features,
    and on it we define our FC to derive planes we need.
    Input:
        size N x n_dim
    Output:
        size n_channels x n_dim
    Parameters :
        n_channels (int) : number of planes/channels
        n_dim (int) : dimension of points (3 for 3D, 2 for 2D)
        n_points (int) : number of points
        c_dim (int) : dimension of out
    """

    def __init__(self,
                 n_dim=3,
                 n_points=300,
                 c_dim=32,
                 hidden_dim=32):
        super(FCPlanenet, self).__init__()

        # Simple PointNet
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(n_dim, 2 * hidden_dim)
        self.fc_0 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

        # MLP
        # uses c_dim = hidden_dim to give accordingly proper output for further reducing to plane parameters
        # self.point_net = SimplePointnet(c_dim=hidden_dim,hidden_dim=hidden_dim, dim=n_dim)

        self.mlp0 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, p):
        batch_size, T, D = p.size()

        # Simple Point Net
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        # Reduce to B x hidden_dim
        net = self.pool(net, dim=1)
        net = self.fc_c(self.actvn(net))

        # MLP
        net = self.mlp0(self.actvn(net))
        net = self.mlp1(self.actvn(net))
        net = self.mlp2(self.actvn(net))

        return (net)


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
