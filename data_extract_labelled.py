import numpy as np
import h5py as h5
from Plotters.plotter_nparray import plot
import torch
import os

###
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
###

def sample_cloud(vertices_c, faces_vc, label, size=2**10, return_eval_cloud=False):

    polygons = vertices_c[faces_vc]
    cross = np.cross(polygons[:, 2] - polygons[:, 0], polygons[:, 2] - polygons[:, 1])
    areas = np.sqrt((cross**2).sum(1)) / 2.0

    probs = areas / areas.sum()
    p_sample = np.random.choice(np.arange(polygons.shape[0]), size=2 * size if return_eval_cloud else size, p=probs)

    sampled_polygons = polygons[p_sample]

    s1 = np.random.random((2 * size if return_eval_cloud else size, 1)).astype(np.float32)
    s2 = np.random.random((2 * size if return_eval_cloud else size, 1)).astype(np.float32)
    cond = (s1 + s2) > 1.
    s1[cond] = 1. - s1[cond]
    s2[cond] = 1. - s2[cond]

    sample = {
        'cloud': (sampled_polygons[:, 0] +
                  s1 * (sampled_polygons[:, 1] - sampled_polygons[:, 0]) +
                  s2 * (sampled_polygons[:, 2] - sampled_polygons[:, 0])).astype(np.float32)
    }


    sample['eval_cloud'] = sample['cloud'][1::2].copy().T
    sample['cloud'] = sample['cloud'][::2].T
    sample['label'] = label

    #plot(sample['cloud'])

    return sample

def data():

    with h5.File((r"D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\data\ShapeNetCore55v2_meshes_resampled.h5"), 'r', libver='latest', swmr=True) as fin:
        vertices_c_bounds = np.empty(fin['test_vertices_c_bounds'].shape, dtype=np.uint64)
        fin['test_vertices_c_bounds'].read_direct(vertices_c_bounds)

        faces_bounds = np.empty(fin['test_faces_bounds'].shape, dtype=np.uint64)
        fin['test_faces_bounds'].read_direct(faces_bounds)

        sample = []
        airplane_indices = np.arange(0,10)  
        chair_indices = np.arange(3725, 3735, 1)
        iterator = np.concatenate((airplane_indices, chair_indices))

        for i in iterator:
            vertices_c = np.array(fin['test_vertices_c'][vertices_c_bounds[i]:vertices_c_bounds[i + 1]],dtype=np.float32)
            faces_vc = np.array(fin['test_faces_vc'][faces_bounds[i]:faces_bounds[i + 1]],dtype=np.uint32)

            if i<3725:
                label = 0
            else:
                label = 1
            
            sample_single = sample_cloud(vertices_c, faces_vc, label, size=2048, return_eval_cloud=True)
            sample.append(sample_single)
        
    return sample

dataset_airplanes = data()
#plot(dataset_airplanes, 'airplanes')
#print("end")