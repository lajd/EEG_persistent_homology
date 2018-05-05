"""
This file computes persistent homological features of of EEG data

The EEG data is expected to be a MatLab matrix file (.m extension)
with the column ['EEG_data'], which gives a matrix of size
(num_samples, num_time_points, num_channels)

Samples represent individual EEG trials. We compute
persistent homology (PH) features for each
(num_time_points, num_channels) sample in the data.

PH features are computed over a sliding time window over
the (num_time_points, num_channels) data. Each frame
of the sliding window generates its own set of PH features.
By computing PH over time, we can quantify how the 'holes' in
the dataset vary in time. We suspect that the size and quantity
of 'holes' in the dataset, as well as how they vary in time, are
important characteristics of EEG data based on the results of
Petri et. al. in their work "Homological scaffolds of brain functional networks"
found at http://rsif.royalsocietypublishing.org/content/11/101/20140873.
"""

from __future__ import print_function
import cPickle as pickle
import pyximport
pyximport.install()
from cython_helpers import filtrations, partial_cor
import networkx as nx
import Holes as ho
import os
from os.path import dirname, abspath
from scipy.io import loadmat
import pandas as pd
from os.path import join
import numpy as np


def create_edges(data, edge_folder, time_window, hz=500, col='EEG_data'):
    """
    Running this method requires you to make a folder with the following structure in /data.

    /data
        /time_of_{}_s # Where {} is the time_window variable
            /edge_vals
            /gen
            /clique_dictionaries

    :param data: path to EEG data [num_trials]
    :param data_path:
    :param time_window:
    :param hz: Frequency EEG data was sampled at
    :return: Writes partial correlations of data over time window to output folders
    """
    if type(data) is str:
        # Load data from path if path is given
        mat = loadmat(data)  # load mat-file
        mdata = mat[col]  # variable in mat file
        df = pd.DataFrame(mdata)
        data = df.as_matrix()

    # Get data shape
    m, n = data.shape
    # Get timesteps per time window
    timesteps = int(hz * time_window)

    # Initialize start/end indices and slice index
    start_idx = 0
    end_idx = 0
    slice_idx = 0

    # Slice the data
    while end_idx < n:
        end_idx = start_idx + timesteps
        data_slice = data[:, start_idx: end_idx]
        start_idx = end_idx

        if (np.zeros(data_slice.shape) == data_slice).all() == True:
            # This data slice is empty -- can't compute partial correlation
            break
        else:
            # This slice has data and partial correlation can be computed
            pcor_slice = partial_cor.partial_corr(data_slice)
            edge_path = join(edge_folder, '_{0}.txt'.format(slice_idx))
            # Write partial correlation to edge path
            with open(edge_path, 'w') as f:
                _m, _n = pcor_slice.shape
                for i in range(_m):
                    for j in range(i + 1):
                        f.write('{0} {1} {2}\n'.format(str(float(i)), str(float(j)), str(pcor_slice[i, j])))
                        slice_idx += 1
    return


def weight_rank_filter(edge_file, save_file):
    """
    :param edge_file: Path to edge folder where partial correlation data is located
    :param save_file: Path to weight-rank-filtration output to
    :return: weight rank filtration of each partial-correlation time window written to file
    """
    G = nx.read_weighted_edgelist(edge_file)
    fil = filtrations.dense_graph_weight_clique_rank_filtration(G, 1)
    with open(save_file, 'wb') as f:
        pickle.dump(fil, f)
    return save_file


def calculate_persistent_homology(edge_dir, output_dir, hom_dim=1):
    """
    :param edge_dir:
    :param output_dir:
    :param hom_dim:
    :return:
    """
    import time
    t1 = time.time()
    fls = os.listdir(edge_dir)
    kv_files = [(i, i.split('_')[-1].split('.')[0]) for i in fls]
    edge_paths = [edge_dir + '/{0}'.format(i[0]) for i in sorted(kv_files, key=lambda line: int(line[1]))]

    for ep in edge_paths:
        dataset_tag = ep.split('/_')[-1].split('.txt')[0]
        clique_save_file = os.path.join(output_dir, 'clique_dictionaries/_{}.pkl'.format(dataset_tag))
        fil_file = weight_rank_filter(ep, clique_save_file)
        ho.persistent_homology_calculation(fil_file, hom_dim, dataset_tag, output_dir + '/', m1=512, m2=2048)

    dt = time.time() - t1
    print("Total time taken is: {}".format(dt))


def get_birth_and_death(gen_list, W=None, normalized=True, factor_l=20.0, factor_p=1.0, show=False):
    b = []
    d = []
    l = []
    p = []
    if normalized == True:
        if W == None:
            W = 0
            for cycle in gen_list:
                if float(cycle.end) > W:
                    W = float(cycle.end)

        for cycle in gen_list:
            b.append(float(cycle.start) / float(W))
            d.append(float(cycle.end) / float(W))
            if len(cycle.composition) > 0:
                l.append(float(len(cycle.composition)) * factor_l)
            else:
                l.append(4.0 * factor_l)
            p.append(float(cycle.persistence_interval()) / float(W) * factor_p)
    else:
        for cycle in gen_list:
            b.append(float(cycle.start))
            d.append(float(cycle.end))
            if len(cycle.composition) > 0:
                l.append(float(len(cycle.composition)) * factor_l)
            else:
                l.append(4.0 * factor_l)
            p.append(float(cycle.persistence_interval()) * factor_p)
    if show == True:
        import matplotlib.pyplot as plt
        plt.scatter(b, d, l, p)
        plt.xlim(0, 1.1 * np.max([np.max(b), np.max(d)]))
        plt.ylim(0, 1.1 * np.max([np.max(b), np.max(d)]))
        plt.xlabel('Birth')
        plt.ylabel('Death')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    return b, d


if __name__ == '__main__':

    """
    Analysis Parameters
    """
    time_window = 0.15

    """
    Path parameters
    """
    # Where the data is located
    repo_dir = dirname(abspath(__file__))
    data_dir = join(repo_dir, 'data')
    subject_1_session_1_pth = join(repo_dir, 'data/raw_data/sub1_ses1.mat')  # Path to data

    # Where to write results
    project_folder = 'final_sample_of_0.15_s'
    directory = join(repo_dir, 'data', project_folder)
    if not os.path.exists(directory):
        os.makedirs(directory)

    """
    load the [num_samples, num_time_points, num_channels] data
    """
    mat = loadmat(subject_1_session_1_pth)  # load mat-file
    mdata = mat['EEG_data']  # (150, 1002, 63) = (num_samples, num_time_points, num_channels)
    
    """
    Reshape data to be [num_samples, num_channels, num_time_points]
    """
    mdata = mdata.transpose(0, 2, 1)

    """
    Compute persistent homology for each sample of size [num_channels, num_time)
    """

    def compute_persistence_features(sample_data, sample_idx, time_window):
        """
        
        :param sample_data: 2d array of size (num_channels, num_time_points)
        :param sample_idx: The sample index (to label output file)
        :return: Written output files -- persistent homology features corresponding to imput sample
        """

        print('sample data shape is: {}'.format(sample_data.shape))

        # Create a directory to store sample output files
        sample_parent_directory = join(directory, 'sample_{0}'.format(sample_idx))
        if not os.path.exists(sample_parent_directory):
            os.makedirs(sample_parent_directory)

        # Create the directory for writing sample-generator files
        sample_generator_file = join(directory, 'sample_{0}/gen'.format(sample_idx))
        if not os.path.exists(sample_generator_file):
            os.makedirs(sample_generator_file)

        # Create the directory for writing edge-values 
        sample_edge_values_file = join(directory, 'sample_{0}/edge_vals'.format(sample_idx))
        if not os.path.exists(sample_edge_values_file):
            os.makedirs(sample_edge_values_file)

        # Create the directory for writing clique dictionaries
        sample_clique_dictionary_files = join(directory, 'sample_{0}/clique_dictionaries'.format(sample_idx))
        if not os.path.exists(sample_clique_dictionary_files):
            os.makedirs(sample_clique_dictionary_files)

        # Write edges to edge-folder
        create_edges(sample_data, sample_edge_values_file, time_window=time_window)

        # Calculate persistent homology on written edge-values
        # Compute up to the first homology dimension

        calculate_persistent_homology(sample_edge_values_file, sample_parent_directory, hom_dim=1)
        return

    sample_idx = 0
    for sample_data in mdata:
        compute_persistence_features(sample_data, sample_idx, time_window)
        sample_idx += 1
        
