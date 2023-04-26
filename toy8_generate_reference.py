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

    for i, v in enumerate(verts):
        pos[i*n:(i+1)*n, :] += v
        labels[i*n:(i+1)*n, mapping[i]] = 1.

    #shuffling = np.random.permutation(N)
    #pos = torch.tensor(pos[shuffling], dtype=torch.float)
    #labels = torch.tensor(labels[shuffling], dtype=torch.float)
    
    pos = torch.tensor(pos, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.float)

    return pos, labels

batch_size = 1600
test_split = 2**15

pos, labels = generate(
    labels='all',
    tot_dataset_size=2**15
)
c = np.where(labels[:test_split])[1]
plt.figure(figsize=(6, 6))
plt.scatter(pos[:test_split, 0], pos[:test_split, 1], c=c, cmap='Set1', s=0.25)
plt.xticks([])
plt.yticks([])
plt.show()

"""
print("end")
plt.figure(figsize=(6, 6))
for i in range(len(point_clouds)):
    plt.scatter(point_clouds[i]['cloud'].T[0], point_clouds[i]['cloud'].T[1], label='cloud')
    plt.scatter(point_clouds[i]['eval_cloud'].T[0], point_clouds[i]['eval_cloud'].T[1], label='eval_cloud')
    plt.title(point_clouds[i]['label'])
    plt.legend()
    plt.show()
"""