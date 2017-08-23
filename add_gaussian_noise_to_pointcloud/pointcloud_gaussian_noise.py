import sys
import numpy as np
import math
import numpy.matlib as M
from matplotlib.mlab import PCA
import tensorflow as tf
######################################################lets go
import gzip
import glob
import os
import random
from os.path import basename
from tensorflow.examples.tutorials.mnist import input_data
def convert_file_name_into_label_idx(label):
    return {
        'airplane': [1],
        'bathtub' : [2],
        'bed':      [3],
        'bench':     [4],
        'bookshelf': [5],
        'bottle': [6],
        'bowl': [7],
        'car': [8],
        'chair': [9],
        'cone': [10],
        'cup': [11],
        'curtain': [12],
        'desk': [13],
        'door': [14],
        'dresser': [15],
        'flower': [16],
        'glass': [17],
        'guitar': [18],
        'keyboard': [19],
        'lamp': [20],
        'laptop': [21],
        'mantel': [22],
        'monitor': [23],
        'night': [24],
        'person': [25],
        'piano': [26],
        'plant': [27],
        'radio': [28],
        'range': [29],
        'sink': [30],
        'sofa': [31],
        'stairs': [32],
        'stool': [33],
        'table': [34],
        'tent': [35],
        'toilet': [36],
        'tv': [37],
        'vase': [38],
        'wardrobe': [39],
        'xbox': [40],
    }[label]
def convert_label_idx_into_label_array(label_idx):
    label_array=np.zeros([1,40])
    label_array[0,label_idx[0]-1]=1
    return label_array

def plot_pointcloud (sorted_pcl):
    from matplotlib import pyplot
    import pylab
    from mpl_toolkits.mplot3d import Axes3D
    import random
    fig = pylab.figure()
    ax = Axes3D(fig)
    ax.scatter(sorted_pcl[:, 0], sorted_pcl[:, 1], sorted_pcl[:, 2])
    pyplot.show()

def read_npy_from_directory_and_add_noise_and_save(path,input_areas,training_or_testing):
    import pypcd
    area_num = len(input_areas)
    for idx in range(area_num):
        if idx==0:
            folders=[path+input_areas[idx]+'/*.npy']
        else:
            temp_str=[path+input_areas[idx]+'/*.npy']
            folders.extend(temp_str)
    file_counter = 1
    for folder in folders:
        files = glob.glob(folder)
        for file in files:
            # disable reading proposed feature
            #lines = np.load(file)
            #temp = np.array([lines])
            #temp=lines.transpose()
            base = os.path.basename(file)
            label = base.rsplit('_', 2)[0]
            object_shor_file_name=base.rsplit('.', 1)[0]
            unsorted_pcl_file_path=path+'../'+object_shor_file_name+'.pcd'
            pc = pypcd.PointCloud.from_path(
                unsorted_pcl_file_path)
            #add gaussian noise here
            for kdx in range(pc.width):
                mu=0
                sigma=0.
                pc.pc_data['x'][kdx] = np.random.normal(mu, sigma, 1)+ pc.pc_data['x'][kdx]
                pc.pc_data['y'][kdx] = np.random.normal(mu, sigma, 1)+ pc.pc_data['y'][kdx]
                pc.pc_data['z'][kdx] = np.random.normal(mu, sigma, 1)+ pc.pc_data['z'][kdx]
            pc_original=pc.pc_data
            file_counter=file_counter+1
            print file_counter
            if(training_or_testing=='training'):
               pc.save_pcd('gaussian_noise_0_1_training/'+object_shor_file_name+'.pcd', compression='binary')
            else:
               pc.save_pcd('gaussian_noise_0_1_testing/'+object_shor_file_name + '.pcd', compression='binary')
            '''
            line_counter=1
            for jdx in range(pc.width):
                pc_xyz=np.array(list(pc_original[jdx])).reshape(1,8)#8here is literal...should be size....
                pc_xyz=pc_xyz[0]
                pc_xyz=pc_xyz[0:3]
                if line_counter == 1:
                    unsorted_pcl=pc_xyz
                    line_counter=line_counter+1
                else:
                    #unsorted_pcl=np.hstack((unsorted_pcl,pc_xyz ))
                    unsorted_pcl = np.vstack((unsorted_pcl, pc_xyz))
                    line_counter = line_counter + 1
            '''
            #plot_pointcloud(unsorted_pcl)
            feature_set_training=1
            label_set_training=1
    return feature_set_training,label_set_training

folder_testing  ='/home/ga48woj/Work/Work/datasets/ModelNet40/dima_sampled_test/pcd_files_test/npy_proposed_feature/'
folder_training = '/home/ga48woj/Work/Work/datasets/ModelNet40/dima_sampled_train/pcd_files_train/npy_proposed_feature/'

#####debug
###########
feature_set_testing,label_set_testing=read_npy_from_directory_and_add_noise_and_save(folder_testing,['descriptors'],'testing')
feature_set_training,label_set_training=read_npy_from_directory_and_add_noise_and_save(folder_training,['descriptors'],'training')
