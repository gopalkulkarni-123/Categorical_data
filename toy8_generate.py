import torch
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

verts = [
         (-2.4142, 1.),
         (-1., 2.4142),
         (1.,  2.4142),
         (2.4142,  1.),
         (2.4142, -1.),
         (1., -2.4142),
         (-1., -2.4142),
         (-2.4142, -1.)
        ]

label_maps = {
              'all':  [0, 1, 2, 3, 4, 5, 6, 7],
              'some': [0, 0, 0, 0, 1, 1, 2, 3],
              'none': [0, 0, 0, 0, 0, 0, 0, 0],
             }

def generate(labels, tot_dataset_size):
    # print('Generating artifical data for setup "%s"' % (labels))

    np.random.seed(0)
    N = tot_dataset_size
    mapping = label_maps[labels]

    pos = np.random.normal(size=(N, 2), scale=0.2)
    labels = np.zeros((N, 8))
    n = N//8

    data = []

    for i, v in enumerate(verts):
        a = {'cloud':0, 'eval_cloud':0, 'label':0}
        
        pos[i*n:(i+1)*n, :] += v
        labels[i*n:(i+1)*n, mapping[i]] = 1.
        
        entire_cloud = pos[i*n:(i+1)*n, :]
        indices = np.random.permutation(entire_cloud.shape[0])
        split_idx = int(entire_cloud.shape[0]/2)
        indices_1 = indices[:split_idx]
        indices_2 = indices[split_idx:] 
        
        a['cloud'] = torch.tensor(entire_cloud[indices_1], dtype=float)
        a['eval_cloud'] = torch.tensor(entire_cloud[indices_2], dtype=float)
        a['label'] = i
        data.append(a)

    return data

batch_size = 1600
test_split = 2**15

point_clouds = generate(
    labels='all',
    tot_dataset_size=2**15
)