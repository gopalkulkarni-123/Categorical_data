import h5py
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import os

#####
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#####

def plot(dataset_1, title):

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    for i in range(10):
        #ax.scatter(dataset[i][0],dataset[i][1],dataset[i][2])

        ax.scatter(dataset_1[2], dataset_1[3], dataset_1[4])
        #ax.scatter(dataset_2[0], dataset_2[1], dataset_2[2])
        plt.show()

filename = r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\Categorical_data\results\airplane_gen_model\airplanes_and_chairs.h5'

with h5py.File(filename, 'r') as file:
    sampled_clouds=np.array(file.get('sampled_clouds'))

def main():
    #reference_data = data()
    for i in range (10):
        #ref_cloud = reference_data[i].get('cloud')
        plot (sampled_clouds[i], 'sampled clouds')

main()