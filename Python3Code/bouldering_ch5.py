##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 5                                               #
#                                                            #
##############################################################

from Chapter5.DistanceMetrics import InstanceDistanceMetrics
from Chapter5.DistanceMetrics import PersonDistanceMetricsNoOrdering
from Chapter5.DistanceMetrics import PersonDistanceMetricsOrdering
from Chapter5.Clustering import NonHierarchicalClustering
from Chapter5.Clustering import HierarchicalClustering
import util.util as util
from util.VisualizeDataset import VisualizeDataset

import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def main():
    DATA_PATH = Path('./intermediate_datafiles_bouldering/')

    # Get all chapter 4 final result files as input
    input_files = list(DATA_PATH.glob('chapter4_result_*.csv'))

    if not input_files:
        print("No Chapter 4 result files found. Please run bouldering_ch4.py with --mode final first.")
        return

    clusteringNH = NonHierarchicalClustering()
    clusteringH = HierarchicalClustering()

    for input_file_path in input_files:
        print(f"\n--- Processing file: {input_file_path.name} ---")
        try:
            dataset = pd.read_csv(input_file_path, index_col=0)
            dataset.index = pd.to_datetime(dataset.index)
        except IOError as e:
            print(f'File not found: {input_file_path.name}. Skipping.')
            continue # Skip to the next file

        # Initialize a visualization object for this dataset.
        DataViz = VisualizeDataset(__file__)

        # Construct the base name for the output file
        # Example: chapter4_result_participant1_Easy1_250.csv -> chapter5_result_participant1_Easy1_250.csv
        base_output_name = input_file_path.name.replace('chapter4_result_', 'chapter5_result_')

        if FLAGS.mode == 'kmeans':
            # Let us look at k-means first.
            k_values = range(2, 10)
            silhouette_values = []

            # Do some initial runs to determine the right number for k

            print('===== kmeans clustering =====')
            for k in k_values:
                print(f'k = {k}')
                dataset_cluster = clusteringNH.k_means_over_instances(copy.deepcopy(
                    dataset), ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], k, 'default', 20, 10)
                silhouette_score = dataset_cluster['silhouette'].mean()
                print(f'silhouette = {silhouette_score}')
                silhouette_values.append(silhouette_score)

            DataViz.plot_xy(x=[k_values], y=[silhouette_values], xlabel='k', ylabel='silhouette score',
                            ylim=[0, 1], line_styles=['b-'])

            # And run the knn with the highest silhouette score

            k = k_values[np.argmax(silhouette_values)]
            print(f'Highest K-Means silhouette score: k = {k}')
            print('Use this value of k to run the --mode=final --k=?')

        if FLAGS.mode == 'kmediods':

            # Do some initial runs to determine the right number for k
            k_values = range(2, 10)
            silhouette_values = []
            print('===== k medoids clustering =====')

            for k in k_values:
                print(f'k = {k}')
                dataset_cluster = clusteringNH.k_medoids_over_instances(copy.deepcopy(
                    dataset), ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], k, 'default', 20, n_inits=10)
                silhouette_score = dataset_cluster['silhouette'].mean()
                print(f'silhouette = {silhouette_score}')
                silhouette_values.append(silhouette_score)

            DataViz.plot_xy(x=[k_values], y=[silhouette_values], xlabel='k', ylabel='silhouette score',
                            ylim=[0, 1], line_styles=['b-'])

            # And run k medoids with the highest silhouette score

            k = k_values[np.argmax(silhouette_values)]
            print(f'Highest K-Medoids silhouette score: k = {k}')

            dataset_kmed = clusteringNH.k_medoids_over_instances(copy.deepcopy(dataset), [
                'acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], k, 'default', 20, n_inits=50)
            DataViz.plot_clusters_3d(dataset_kmed, [
                'acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], 'cluster', ['label'])
            DataViz.plot_silhouette(dataset_kmed, 'cluster', 'silhouette')
            util.print_latex_statistics_clusters(dataset_kmed, 'cluster', [
                'acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], 'label')

        # And the hierarchical clustering is the last one we try
        if FLAGS.mode == 'agglomerative':

            k_values = range(2, 10)
            silhouette_values = []

            # Do some initial runs to determine the right number for the maximum number of clusters.

            print('===== agglomerative clustering =====')
            for k in k_values:
                print(f'k = {k}')
                dataset, l = clusteringH.agglomerative_over_instances(dataset, [
                    'acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], k, 'euclidean', use_prev_linkage=True,
                                                                      link_function='ward')
                silhouette_score = dataset['silhouette'].mean()
                print(f'silhouette = {silhouette_score}')
                silhouette_values.append(silhouette_score)
                if k == k_values[0]:
                    DataViz.plot_dendrogram(dataset, l)

            DataViz.plot_xy(x=[k_values], y=[silhouette_values], xlabel='k', ylabel='silhouette score',
                            ylim=[0, 1], line_styles=['b-'])

        if FLAGS.mode == 'final':
            # And we select the outcome dataset of the knn clustering....
            clusteringNH = NonHierarchicalClustering()

            dataset = clusteringNH.k_means_over_instances(dataset, ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], FLAGS.k,
                                                          'default', 50, 50)
            DataViz.plot_clusters_3d(dataset, ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], 'cluster', ['label'])
            DataViz.plot_silhouette(dataset, 'cluster', 'silhouette')
            util.print_latex_statistics_clusters(
                dataset, 'cluster', ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], 'label')
            del dataset['silhouette']

            # Store the final outcome for this dataset
            output_file = DATA_PATH / f'{base_output_name}'
            dataset.to_csv(output_file)
            print(f"Final results for {input_file_path.name} saved to: {output_file}")


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final, kmeans, kmediods, hierarchical or aggloromative. \
                        'kmeans' to study the effect of kmeans on a selection of variables \
                        'kmediods' to study the effect of kmediods on a selection of variables \
                        'agglomerative' to study the effect of agglomerative clustering on a selection of variables  \
                        'final' kmeans with an optimal level of k is used for the next chapter",
                        choices=['kmeans', 'kmediods', 'agglomerative', 'final'])

    parser.add_argument('--k', type=int, default=6,
                        help="The selected k number of means used in 'final' mode of this chapter' \
                        ")

    FLAGS, unparsed = parser.parse_known_args()

    main()