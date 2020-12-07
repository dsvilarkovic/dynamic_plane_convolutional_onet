import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import ResnetBlockFC, FCPlanenet
from torch_scatter import scatter_mean, scatter_max
from src.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, positional_encoding, normalize_dynamic_plane_coordinate, ChangeBasis
from src.encoder.unet import UNet
import pdb

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

class LocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks
        for each local point on the ground plane.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', unet=False, unet_kwargs=None, 
                 plane_resolution=None, grid_resolution=None, plane_type='xz', padding=0.1,
                 n_blocks=5, pos_encoding=False):
        super().__init__()
        self.c_dim = c_dim
        
        if pos_encoding == True:
            dim = 60 # hardcoded

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        # self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        # self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        # self.block_2 = ResnetBlockFC(2*hidden_dim, c_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        # self.pool = maxpool
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None


        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pe = positional_encoding()

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane,
                                      self.reso_plane)  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane


    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid ** 3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p):
        batch_size, T, D = p.size()

        # acquire the index for each point
        coord = {}
        index = {}
        if 'xz' in self.plane_type:
            coord['xz'] = normalize_coordinate(p.clone(), plane='xz', padding=self.padding)
            index['xz'] = coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = normalize_coordinate(p.clone(), plane='xy', padding=self.padding)
            index['xy'] = coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = normalize_coordinate(p.clone(), plane='yz', padding=self.padding)
            index['yz'] = coordinate2index(coord['yz'], self.reso_plane)

        ##################
        if self.pos_encoding:
            pp = self.pe(p)
            net = self.fc_pos(pp)
        else:
            net = self.fc_pos(p)
        ##################

        # net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}

        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c, plane='xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')

        return fea


class DynamicLocalPoolPointnet(nn.Module):
    """PointNet-based encoder network with ResNet blocks
    for each local point on the ground plane. Learns n_channels dynamic planes 

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension 
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
        pos_encoding (bool): positional encoding  Defaults to False.
        n_channels (int): number of learning planes Defaults to 3.
        plane_net (str): type of plane-prediction network. Defaults to 'FCPlanenet'.
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', unet=False, unet_kwargs=None,
                 plane_resolution=None,
                 grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5, pos_encoding=False, n_channels=3, plane_net='FCPlanenet'):
        super().__init__()
        self.c_dim = c_dim
        self.num_channels = n_channels

        if pos_encoding==True:
            dim = 60

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        # self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        # self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        # self.block_2 = ResnetBlockFC(2*hidden_dim, c_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        planenet_hidden_dim = hidden_dim
        self.fc_plane_net = FCPlanenet(n_dim=dim, hidden_dim=hidden_dim)

    
        # Create FC layers based on the number of planes
        self.plane_params = nn.ModuleList([
            nn.Linear(planenet_hidden_dim, 3) for i in range(n_channels)
        ])

        self.plane_params_hdim = nn.ModuleList([
            nn.Linear(3, hidden_dim) for i in range(n_channels)
        ])

        self.actvn = nn.ReLU()
        # self.pool = maxpool
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None


        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pe = positional_encoding()

    def generate_dynamic_plane_features(self, p, c, normal_feature, basis_normalizer_matrix):
        # acquire indices of features in plane
        xy = normalize_dynamic_plane_coordinate(p.clone(), basis_normalizer_matrix,
                                                padding=self.padding)  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)

        c = c.permute(0, 2, 1)  # B x 512 x T
        c = c + normal_feature.unsqueeze(2)
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane,
                                      self.reso_plane)  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid ** 3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p):
        batch_size, T, D = p.size()
        self.device = 'cpu'

        ##################
        if self.pos_encoding:
            pp = self.pe(p)
            net = self.fc_pos(pp)
            net_pl = self.fc_plane_net(pp)
        else:
            net = self.fc_pos(p)
            net_pl = self.fc_plane_net(p)
        ##################

        normal_fea = []
        normal_fea_hdim = {}

        for l in range(self.num_channels):
            normal_fea.append(self.plane_params[l](self.actvn(net_pl)))
            normal_fea_hdim['plane{}'.format(l)] = self.plane_params_hdim[l](normal_fea[l])

        self.plane_parameters = torch.stack(normal_fea, dim=1) # plane parameter (batch_size x L x 3)
        C_mat = ChangeBasis(self.plane_parameters,
                                 device=self.device)  # change of basis and normalizer matrix (concatenated)
        num_planes = C_mat.size()[1]

        # acquire the index for each point
        coord = {}
        index = {}

        for l in range(num_planes):
            coord['plane{}'.format(l)] = normalize_dynamic_plane_coordinate(p.clone(), C_mat[:, l],
                                                                            padding=self.padding)
            index['plane{}'.format(l)] = coordinate2index(coord['plane{}'.format(l)], self.reso_plane)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}

        for l in range(C_mat.size()[1]):
            fea['plane{}'.format(l)] = self.generate_dynamic_plane_features(p, c, normal_fea_hdim['plane{}'.format(l)], C_mat[:, l])

        fea['c_mat'] = C_mat

        # Normalize plane params for similarity loss calculation
        self.plane_parameters = self.plane_parameters.reshape([batch_size * num_planes, 3])
        self.plane_parameters = self.plane_parameters / torch.norm(self.plane_parameters, p=2, dim=1).view(batch_size * num_planes,
                                                                                            1)  # normalize
        self.plane_parameters = self.plane_parameters.view(batch_size, -1)
        self.plane_parameters = self.plane_parameters.view(batch_size, -1, 3)
        # print("just fea", type(fea))
        return fea

class HybridLocalPoolPointnet(nn.Module):
    """PointNet-based Hybrid encoder network with ResNet blocks
        for each local point on the ground plane. Has 3 predefined canonical planes 
        + n_channels dynamic planes

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension 
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
        pos_encoding (bool): positional encoding  Defaults to False.
        n_channels (int): number of learning planes Defaults to 3.
        plane_net (str): type of plane-prediction network. Defaults to 'FCPlanenet'.
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', unet=False, unet_kwargs=None, plane_resolution=None,
                 grid_resolution=None, padding=0.1, n_blocks=5, pos_encoding=False, n_channels=3,
                 plane_net='FCPlanenet'):
        super().__init__()
        self.c_dim = c_dim
        self.num_channels = n_channels
        n_dynamic_channels = n_channels - 3

        if pos_encoding == True:
            dim = 60

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)
        ])

        self.fc_c = nn.Linear(hidden_dim, c_dim)

        planenet_hidden_dim = hidden_dim
        self.fc_plane_net = FCPlanenet(n_dim=dim, hidden_dim=hidden_dim)

        # Create FC layers based on the number of planes
        self.plane_params = nn.ModuleList([
            nn.Linear(planenet_hidden_dim, 3) for i in range(n_dynamic_channels)
        ])

        self.plane_params_hdim = nn.ModuleList([
            nn.Linear(3, hidden_dim) for i in range(n_dynamic_channels)
        ])

        self.actvn = nn.ReLU()
        # self.pool = maxpool
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None



        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pe = positional_encoding()

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane,
                                      self.reso_plane)  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_dynamic_plane_features(self, p, c, normal_feature, basis_normalizer_matrix):
        # acquire indices of features in plane
        xy = normalize_dynamic_plane_coordinate(p.clone(), basis_normalizer_matrix,
                                                padding=self.padding)  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)

        c = c.permute(0, 2, 1)  # B x 512 x T
        c = c + normal_feature.unsqueeze(2)
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane,
                                      self.reso_plane)  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()
        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid ** 3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p):
        batch_size, T, D = p.size()
        self.device = 'cuda'

        ##################
        if self.pos_encoding:
            pp = self.pe(p)
            net = self.fc_pos(pp)
            net_pl = self.fc_plane_net(pp)
        else:
            net = self.fc_pos(p)
            net_pl = self.fc_plane_net(p)
        ##################

        normal_fea = []
        normal_fea_hdim = {}

        num_planes = self.num_channels
        num_dynamic_planes = self.num_channels - 3

        for l in range(num_dynamic_planes):
            normal_fea.append(self.plane_params[l](self.actvn(net_pl)))
            normal_fea_hdim['dynamic_plane{}'.format(l)] = self.plane_params_hdim[l](normal_fea[l])

        if(self.num_channels == 3):
            raise Exception(f"Number of channels is {self.num_channels}, no point in using Hybrid approach")
        self.plane_parameters = torch.stack(normal_fea, dim=1)  # plane parameter (batch_size x num_dynamic_planes x 3)
        C_mat = ChangeBasis(self.plane_parameters,
                                 device=self.device)  # change of basis and normalizer matrix (concatenated)

        # acquire the index for each point
        coord = {}
        index = {}

        for l in range(num_planes):
            if l == 0:
                coord['plane{}'.format(l)] = normalize_coordinate(p.clone(), plane='xz', padding=self.padding)
                index['plane{}'.format(l)] = coordinate2index(coord['plane{}'.format(l)], self.reso_plane)
            elif l ==1:
                coord['plane{}'.format(l)] = normalize_coordinate(p.clone(), plane='xy', padding=self.padding)
                index['plane{}'.format(l)] = coordinate2index(coord['plane{}'.format(l)], self.reso_plane)
            elif l == 2:
                coord['plane{}'.format(l)] = normalize_coordinate(p.clone(), plane='yz', padding=self.padding)
                index['plane{}'.format(l)] = coordinate2index(coord['plane{}'.format(l)], self.reso_plane)
            else:
                dynamic_plane_id = l-3
                coord['plane{}'.format(l)] = normalize_dynamic_plane_coordinate(p.clone(), C_mat[:, dynamic_plane_id],
                                                                            padding=self.padding)
                index['plane{}'.format(l)] = coordinate2index(coord['plane{}'.format(l)], self.reso_plane)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}

        for l in range(num_planes):
            if l == 0:
                fea['plane{}'.format(l)] = self.generate_plane_features(p, c, plane='xz')
            elif l==1:
                fea['plane{}'.format(l)] = self.generate_plane_features(p, c, plane='xy')
            elif l == 2:
                fea['plane{}'.format(l)] = self.generate_plane_features(p, c, plane='yz')
            else:
                dynamic_plane_id = l - 3
                fea['plane{}'.format(l)] = self.generate_dynamic_plane_features(p, c, normal_fea_hdim['dynamic_plane{}'.format(dynamic_plane_id)],
                                                                             C_mat[:, dynamic_plane_id])

        fea['c_mat'] = C_mat
        # Normalize plane params for similarity loss calculation
        eye_basis = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]).to(self.device)
        canonical_plane_parameters = torch.cat(batch_size * [eye_basis]).view(batch_size, 3, 3)

        self.plane_parameters = self.plane_parameters.reshape([batch_size * num_dynamic_planes, 3])
        self.plane_parameters = self.plane_parameters / torch.norm(self.plane_parameters, p=2, dim=1).view(
            batch_size * num_dynamic_planes,
            1)  # normalize
        self.plane_parameters = self.plane_parameters.view(batch_size, -1)
        self.plane_parameters = self.plane_parameters.view(batch_size, -1, 3)

        # Concatenate canonical plane normals and dynamic plane parameters
        self.plane_parameters = torch.cat([canonical_plane_parameters, self.plane_parameters], dim=1)

        return fea
