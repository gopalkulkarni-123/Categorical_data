import h5py
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import os

def plot(dataset_1, title):

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    for i in range(dataset_1.shape[0]):
        ax.scatter(dataset_1[0], dataset_1[1], dataset_1[2])
    
    plt.show()

filename = r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\Categorical_data\results\airplane_gen_model\chairs_v7_128.h5'

with h5py.File(filename, 'r') as file:
    sampled_clouds=np.array(file.get('sampled_clouds'))

for i in range(sampled_clouds.shape[0]):
    plot(sampled_clouds[i], f'sampled clouds {i}')