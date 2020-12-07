import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
from src.utils import libmcubes
from src.common import make_3d_grid
from src.utils.libsimplify import simplify_mesh
from src.utils.libmise import MISE
import time
import pdb
import sys
from math import sin,cos,radians,sqrt
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class Generator3D(object):
    '''  Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        simplify_nfaces (int): number of faces the mesh should be simplified to
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 simplify_nfaces=None):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces

    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}

        # Encode inputs
        t0 = time.time()
        with torch.no_grad():
            semantic_map = data.get('semantic_map', None)
            if semantic_map is not None:
                c = self.model.encoder(semantic_map.to(device))
            else:
                c = self.model.encode_inputs(inputs)

        stats_dict['time (encode inputs)'] = time.time() - t0

        z = self.model.get_z_from_prior((1,), sample=self.sample).to(device)
        #q_z = self.model.infer_z(None, None, c, inputs=inputs, **kwargs)
        #z = q_z.rsample()
 
        mesh = self.generate_from_latent(z, c, stats_dict=stats_dict, **kwargs)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def generate_rotated_mesh(self, data, return_stats=True, DEGREES = 0, save_rotation_tensor = False):
        ''' Generates the output mesh which is rotated by DEGREES interval. 
        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}
        # for random prints, there are 4048 files in each class
        random_counter = 1
        # random_counter = int(random.random()* 10000000000000 % 256)
        if(random_counter == 0):
            print('Found one!')
            fig = plt.figure()
            ax = plt.axes(projection='3d')
        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        inputs_np = inputs
        if(random_counter == 0):
            #Visualization code for generating images of input and output rotations
            inputs_np = inputs.to('cpu')
            ax.scatter(inputs_np[0,:,0], inputs_np[0,:,1], inputs_np[0,:,2], color = 'red', s=0.1)
        # This function rotates object pointcloud that will be put in the encoder.
        inputs, rotation = self.rotate_points(inputs, DEGREES = DEGREES, save_rotation_tensor = save_rotation_tensor) #This function rotates object pointcloud that will be put in the encoder.
        if(random_counter == 0):
            #Visualization code for generating images of input and output rotations
            inputs_np = inputs.to('cpu')
            ax.scatter(inputs_np[0,:,0],inputs_np[0,:,1], inputs_np[0,:,2], color = 'blue', s=0.1)
        
        kwargs = {}

        # Encode inputs
        t0 = time.time()
        with torch.no_grad():	
            semantic_map = data.get('semantic_map', None)	
            if semantic_map is not None:	
                c = self.model.encoder(semantic_map.to(device))	
            else:	
                c = self.model.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = time.time() - t0
        z = self.model.get_z_from_prior((1,), sample=self.sample).to(device)
        mesh = self.generate_from_latent(z, c, stats_dict=stats_dict, **kwargs)	
        vertices = None
        if(random_counter == 0):
            #Visualization code for generating images of input and output rotations
            vertices = torch.tensor(mesh.vertices).cpu()
            ax.scatter(vertices[:,0],vertices[:,1], vertices[:,2], color = 'green', s=0.01)
            plt.savefig(str(data['idx']) + '_' + str(DEGREES) + '_degrees.jpg')
            plt.close()
        del vertices, inputs_np
        if return_stats:
            return mesh, stats_dict, rotation
        else:
            return mesh, rotation

    def generate_from_latent(self, z, c=None, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.

        Args:
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
            values = self.eval_points(pointsf, z, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                # Evaluate model and update
                values = self.eval_points(
                    pointsf, z, c, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, z, c, stats_dict=stats_dict)
        return mesh

    def generate_from_rotated_latent(self, z, c=None, stats_dict={}, **kwargs):
        ''' Generates mesh from latent by using rotated query points.
        Args:
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding
        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
            values = self.eval_points(pointsf, z, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx) #This function rotates object pointcloud that will be put in the encoder.
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)
            points = mesh_extractor.query()
            
            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                pointsf = self.rotate_points(pointsf,query_points = True) #This function is used to rotate query points.    
                # Evaluate model and update
                values = self.eval_points(	
                    pointsf, z, c, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()
            value_grid = mesh_extractor.to_dense()
        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0
        mesh = self.extract_mesh(value_grid, z, c, stats_dict=stats_dict)	
        return mesh

    def eval_points(self, p, z, c=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points 
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model.decode(pi, z, c, **kwargs).logits	
            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def extract_mesh(self, occ_hat, z, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, z, c)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, z, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, z, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        z, c = z.unsqueeze(0), c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, z, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, z, c=None):
        ''' Refines the predicted mesh.
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode(face_point.unsqueeze(0), z, c).logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh

    def rotate_points(self, pointcloud_model,query_points = False, DEGREES = 0, save_rotation_tensor = False):
        """
                Link: https://en.wikipedia.org/wiki/Rotation_matrix
            Function for rotating points
                pointcloud_model (numpy 3d array) - batch_size x pointcloud_size x 3d channel sized numpy array which presents pointcloud
                DEGREES (int) - range of rotations to be used
                query_points (boolean) - used for rotating query points with already existing rotation matrix
                use_rotation_tensor (boolean) - asking whether DEGREES should be used for generating new rotation matrix, or use the already established one
                save_rotation_tensor (boolean) - asking to keep rotation matrix in a pytorch .pt file
        """
        if(query_points != True):
            angle_range = DEGREES
            x_angle = radians(random.random() * angle_range)
            y_angle = radians(random.random() * angle_range)
            z_angle = radians(random.random() * angle_range)

            rot_x = torch.Tensor([[1,0,0,0],[0, cos(x_angle),-sin(x_angle),0], [0, sin(x_angle), cos(x_angle),0], [0,0,0,1]])
            rot_y = torch.Tensor([[cos(y_angle),0,sin(y_angle), 0],[0, 1, 0,0], [-sin(y_angle),0,cos(y_angle),0], [0,0,0,1]])
            rot_z = torch.Tensor([[cos(z_angle), -sin(z_angle),0,0],[sin(z_angle), cos(z_angle),0,0],[0,0,1,0], [0,0,0,1]])
            
            pointcloud_model = pointcloud_model[0,:,:]
       
            point_cloud_size, _ = pointcloud_model.shape
            pointcloud_model = torch.cat([pointcloud_model, torch.ones(point_cloud_size,1).to(self.device)], dim = 1)
            
            rotation_matrix = torch.mm(rot_y, rot_z)
            rotation_matrix = torch.mm(rot_x,rotation_matrix)
            pointcloud_model_rotated = torch.mm(pointcloud_model, rotation_matrix.to(self.device))
            self.rotation_matrix = rotation_matrix
            
            if(save_rotation_tensor):
                #used for experiemnt on rotating manually rotated planes
                torch.save(rotation_matrix, 'rotation_matrix.pt') #used for plane prediction, change it at your will 
            return pointcloud_model_rotated[None,:,0:3], (x_angle, y_angle, z_angle)
        else:
            point_cloud_size, _ = pointcloud_model.shape
            pointcloud_model = torch.cat([pointcloud_model, torch.ones(point_cloud_size,1).to(self.device)], dim = 1)
            pointcloud_model_rotated =torch.mm(pointcloud_model, self.rotation_matrix.to(self.device))
            return pointcloud_model_rotated[:,0:3]
