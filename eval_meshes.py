import argparse
# import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import trimesh
import torch
from src import config, data
from src.eval import MeshEvaluator
from src.utils.io import load_pointcloud
import pickle 	
from math import sin,cos,radians,sqrt	
import random	
from mpl_toolkits.mplot3d import Axes3D	
import matplotlib.pyplot as plt	
def rotate_points(pointcloud_model,rotations):	
    ## https://en.wikipedia.org/wiki/Rotation_matrix	
    (x_angle, y_angle, z_angle) = rotations	
    # print(x_angle,y_angle, z_angle)	
    rot_x = torch.Tensor([[1,0,0,0],[0, cos(x_angle),-sin(x_angle),0], [0, sin(x_angle), cos(x_angle),0], [0,0,0,1]])	
    rot_y = torch.Tensor([[cos(y_angle),0,sin(y_angle), 0],[0, 1, 0,0], [-sin(y_angle),0,cos(y_angle),0], [0,0,0,1]])	
    rot_z = torch.Tensor([[cos(z_angle), -sin(z_angle),0,0],[sin(z_angle), cos(z_angle),0,0],[0,0,1,0], [0,0,0,1]])	

    pointcloud_model = torch.tensor(pointcloud_model)	
    point_cloud_size, _ = pointcloud_model.shape	
    pointcloud_model = torch.cat([pointcloud_model, torch.ones(point_cloud_size,1)], dim = 1)	
    	
    rotation_matrix = torch.mm(rot_y, rot_z)	
    rotation_matrix = torch.mm(rot_x,rotation_matrix)	
    pointcloud_model_rotated = torch.mm(pointcloud_model, rotation_matrix)	

    return pointcloud_model_rotated[:,0:3].numpy()

parser = argparse.ArgumentParser(
    description='Evaluate mesh algorithms.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--eval_input', action='store_true',
                    help='Evaluate inputs instead.')
parser.add_argument('--eval-rotations', action='store_true',	
                    help='Evaluate inputs instead.')
args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
eval_rotations = args.eval_rotations	
# Shorthands
out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
if not args.eval_input:
    out_file = os.path.join(generation_dir, 'eval_meshes_full.pkl')
    out_file_class = os.path.join(generation_dir, 'eval_meshes.csv')
    if(eval_rotations):	
        #For evaluating rotaitons	
        out_file_class = os.path.join(generation_dir, 'eval_meshes_with_rotation.csv')	
    else:	
        out_file_class = os.path.join(generation_dir, 'eval_meshes.csv')
else:
    out_file = os.path.join(generation_dir, 'eval_input_full.pkl')
    out_file_class = os.path.join(generation_dir, 'eval_input.csv')

# Dataset
points_field = data.PointsField(
    cfg['data']['points_iou_file'],
    unpackbits=cfg['data']['points_unpackbits'],
    multi_files=cfg['data']['multi_files']
)
pointcloud_field = data.PointCloudField(
    cfg['data']['pointcloud_chamfer_file'],
    multi_files=cfg['data']['multi_files']
)
fields = {
    'points_iou': points_field,
    'pointcloud_chamfer': pointcloud_field,
    'idx': data.IndexField(),
}

print('Test split: ', cfg['data']['test_split'])

dataset_folder = cfg['data']['path']
dataset = data.Shapes3dDataset(
    dataset_folder, fields,
    cfg['data']['test_split'],
    categories=cfg['data']['classes'])

# Evaluator
evaluator = MeshEvaluator(n_points=100000)

# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

# Evaluate all classes
eval_dicts = []
print('Evaluating meshes...')
for it, data in enumerate(tqdm(test_loader)):
    if data is None:
        print('Invalid data.')
        continue

    # Output folders
    if not args.eval_input:
        mesh_dir = os.path.join(generation_dir, 'meshes')
        pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    else:
        mesh_dir = os.path.join(generation_dir, 'input')
        pointcloud_dir = os.path.join(generation_dir, 'input')

    # Get index etc.
    idx = data['idx'].item()

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}

    modelname = model_dict['model']
    category_id = model_dict['category']

    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    if category_id != 'n/a':
        mesh_dir = os.path.join(mesh_dir, category_id)
        pointcloud_dir = os.path.join(pointcloud_dir, category_id)

    # Evaluate
    pointcloud_tgt = data['pointcloud_chamfer'].squeeze(0).numpy()
    normals_tgt = data['pointcloud_chamfer.normals'].squeeze(0).numpy()
    points_tgt = data['points_iou'].squeeze(0).numpy()
    occ_tgt = data['points_iou.occ'].squeeze(0).numpy()

    # Evaluating mesh and pointcloud
    # Start row and put basic informatin inside
    eval_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }
    eval_dicts.append(eval_dict)

    # Evaluate mesh
    if cfg['test']['eval_mesh']:
        mesh_file = os.path.join(mesh_dir, '%s.off' % modelname)

        if(eval_rotations):	
            rotation_out_file = os.path.join(mesh_dir, '%s_rotation.pckl' % modelname)	
            rotations = None	
            with open(rotation_out_file, 'rb') as f:	
                rotations = pickle.load(f)	
            pointcloud_tgt = rotate_points(pointcloud_model = pointcloud_tgt,rotations = rotations)	
            normals_tgt = rotate_points(pointcloud_model = normals_tgt,rotations = rotations)	
            points_tgt = rotate_points(pointcloud_model = points_tgt,rotations = rotations)	

        if os.path.exists(mesh_file):
            try:
                mesh = trimesh.load(mesh_file, process=False)
                eval_dict_mesh = evaluator.eval_mesh(
                    mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt)
                for k, v in eval_dict_mesh.items():
                    eval_dict[k + ' (mesh)'] = v
            except Exception as e:
                print("Error: Could not evaluate mesh: %s" % mesh_file)
        else:
            print('Warning: mesh does not exist: %s' % mesh_file)

    # Evaluate point cloud
    if cfg['test']['eval_pointcloud']:
        pointcloud_file = os.path.join(
            pointcloud_dir, '%s.ply' % modelname)

        if os.path.exists(pointcloud_file):
            pointcloud = load_pointcloud(pointcloud_file)
            eval_dict_pcl = evaluator.eval_pointcloud(
                pointcloud, pointcloud_tgt)
            for k, v in eval_dict_pcl.items():
                eval_dict[k + ' (pcl)'] = v
        else:
            print('Warning: pointcloud does not exist: %s'
                    % pointcloud_file)

# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
eval_df.to_pickle(out_file)

# Create CSV file  with main statistics
eval_df_class = eval_df.groupby(by=['class name']).mean()
eval_df_class.to_csv(out_file_class)

# Print results
eval_df_class.loc['mean'] = eval_df_class.mean()
print(eval_df_class)
