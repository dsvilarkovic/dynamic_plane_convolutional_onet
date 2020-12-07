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
                 n_channels=3,
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
        # converts point cloud to it's n_channels plane that best explain it
        self.to_planes = nn.Linear(hidden_dim, n_channels * 3)

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
        net = self.to_planes(self.actvn(net))
        return (net)

class FCPlanenet_conv1d(nn.Module):
    """
    Basically a linear mapping without bias
    In : batch_size x num_points x 3
    Out: batch_size x 32
    """

    def __init__(self,
                 n_dim=3,
                 n_channels=3,
                 n_points=3000,
                 c_dim=32,
                 hidden_dim=16):
        super(FCPlanenet_conv1d, self).__init__()

        self.conv1d_0 = nn.Conv1d(n_dim, hidden_dim, kernel_size = 3, stride = 1, padding=1) #3000
        self.maxpool1d_1 = nn.MaxPool1d(kernel_size=3, stride=3) #1000
        self.conv1d_1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1) #1000
        self.maxpool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # maxpool_1 500
        self.conv1d_2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)  # 500
        # maxpool_2 250
        self.conv1d_3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1) # 250
        # maxpool_2 125
        self.conv1d_4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)  # 63
        self.conv1d_5 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)  # 32
        # maxpool_2 16
        self.conv1d_6 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1) # 16
        # maxpool_2 8
        self.conv1d_7 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)  # 8
        # maxpool_2 4
        self.conv1d_8 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)  # 4
        # maxpool_2 2
        self.conv1d_9 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)  # 2
        # maxpool_2 1

        self.to_planes = nn.Linear(hidden_dim, n_channels * 3)

        self.actvn = nn.ReLU()

    def forward(self, p):

        p = p.transpose(1,2)
        net = self.conv1d_0(p) # B x 32 x 3000
        net = self.maxpool1d_1(net) # B x 32 x 1000
        net = self.conv1d_1(self.actvn(net)) # 1000
        net = self.maxpool1d_2(net) # 500
        net = self.conv1d_2(self.actvn(net)) #500
        net = self.maxpool1d_2(net) #250
        net = self.conv1d_3(self.actvn(net)) #250
        net = self.maxpool1d_2(net)  # 125

        net = self.conv1d_4(self.actvn(net)) # 63
        net = self.conv1d_5(self.actvn(net)) # 32
        net = self.maxpool1d_2(net) # 16

        net = self.maxpool1d_2(self.conv1d_6(self.actvn(net))) # 8
        net = self.maxpool1d_2(self.conv1d_7(self.actvn(net)))  # 4
        net = self.maxpool1d_2(self.conv1d_8(self.actvn(net)))  # 2
        net = self.maxpool1d_2(self.conv1d_9(self.actvn(net))) # B x 32 x 1

        net = self.actvn(net)
        net = net.squeeze(2) # B x hidden_dim
        net = self.to_planes(net)

        return (net)
        
class ResnetBlockConv1d(nn.Module):
    ''' 1D-Convolutional ResNet block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_h=None, size_out=None):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = nn.BatchNorm1d(size_in)
        self.bn_1 = nn.BatchNorm1d(size_h)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


# Utility modules
class AffineLayer(nn.Module):
    ''' Affine layer class.

    Args:
        c_dim (tensor): dimension of latent conditioned code c
        dim (int): input dimension
    '''

    def __init__(self, c_dim, dim=3):
        super().__init__()
        self.c_dim = c_dim
        self.dim = dim
        # Submodules
        self.fc_A = nn.Linear(c_dim, dim * dim)
        self.fc_b = nn.Linear(c_dim, dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_A.weight)
        nn.init.zeros_(self.fc_b.weight)
        with torch.no_grad():
            self.fc_A.bias.copy_(torch.eye(3).view(-1))
            self.fc_b.bias.copy_(torch.tensor([0., 0., 2.]))

    def forward(self, x, p):
        assert(x.size(0) == p.size(0))
        assert(p.size(2) == self.dim)
        batch_size = x.size(0)
        A = self.fc_A(x).view(batch_size, 3, 3)
        b = self.fc_b(x).view(batch_size, 1, 3)
        out = p @ A + b
        return out