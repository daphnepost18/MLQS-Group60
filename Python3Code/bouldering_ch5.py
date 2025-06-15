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

    USE_ALL_FILES = (FLAGS.source == 'all')

    if USE_ALL_FILES:
        print("Mode: Processing all individual chapter 4 final result files...")
        all_chapter4_files = list(DATA_PATH.glob('chapter4_result_*.csv'))
        # We explicitly filter out the combined file to avoid processing it along with the individuals.
        input_files = [f for f in all_chapter4_files if 'combined' not in f.name]
    else:
        print("Mode: Processing only the combined chapter 4 final result file...")
        combined_file = DATA_PATH / 'chapter4_result_combined.csv'
        if combined_file.exists():
            input_files = [combined_file]
        else:
            input_files = []

    # Check if any files were found to process.
    if not input_files:
        if USE_ALL_FILES:
            print(
                "No individual Chapter 4 final result files found. Please run bouldering_ch4.py with '--mode final' first.")
        else:
            print(
                f"Combined file not found at '{combined_file}'. Please run bouldering_ch4.py with '--mode final' and '--source combined'.")
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
            continue

        DataViz = VisualizeDataset(__file__)
        base_output_name = input_file_path.name.replace('chapter4_result_', 'chapter5_result_')
        dataset_name = input_file_path.name.replace('chapter4_result_', '').replace('.csv', '')

        if FLAGS.mode == 'kmeans':
            k_values = range(2, 10)
            silhouette_values = []
            print('===== kmeans clustering =====')
            for k in k_values:
                print(f'k = {k}')
                dataset_cluster = clusteringNH.k_means_over_instances(copy.deepcopy(
                    dataset), ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], k, 'default', 20, 10)
                silhouette_score = dataset_cluster['silhouette'].mean()
                print(f'silhouette = {silhouette_score}')
                silhouette_values.append(silhouette_score)
            DataViz.plot_xy(x=[k_values], y=[silhouette_values], xlabel='k', ylabel='silhouette score',
                            ylim=[0, 1], line_styles=['b-'], dataset_name=dataset_name, methodch3='Kmeans')
            k = k_values[np.argmax(silhouette_values)]
            print(f'Highest K-Means silhouette score: k = {k}')
            print('Use this value of k to run the --mode=final --k=?')

        if FLAGS.mode == 'kmediods':
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
                            ylim=[0, 1], line_styles=['b-'], dataset_name=dataset_name, methodch3='Kmedoids')
            k = k_values[np.argmax(silhouette_values)]
            print(f'Highest K-Medoids silhouette score: k = {k}')
            dataset_kmed = clusteringNH.k_medoids_over_instances(copy.deepcopy(dataset), [
                'acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], k, 'default', 20, n_inits=50)
            DataViz.plot_clusters_3d(dataset_kmed, ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], 'cluster',
                                     ['label'],
                                     dataset_name=dataset_name, method='Kmediods')
            DataViz.plot_silhouette(dataset_kmed, 'cluster', 'silhouette', dataset_name=dataset_name, method='Kmediods')
            util.print_latex_statistics_clusters(dataset_kmed, 'cluster', [
                'acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], 'label')

        if FLAGS.mode == 'agglomerative':
            k_values = range(2, 10)
            silhouette_values = []
            print('===== agglomerative clustering =====')
            for k in k_values:
                print(f'k = {k}')
                dataset, l = clusteringH.agglomerative_over_instances(copy.deepcopy(dataset), [
                    'acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], k, 'euclidean', use_prev_linkage=True,
                                                                      link_function='ward')
                silhouette_score = dataset['silhouette'].mean()
                print(f'silhouette = {silhouette_score}')
                silhouette_values.append(silhouette_score)
                if k == k_values[0]:
                    DataViz.plot_dendrogram(dataset, l, dataset_name=dataset_name, method='agglomerative')
            DataViz.plot_xy(x=[k_values], y=[silhouette_values], xlabel='k', ylabel='silhouette score',
                            ylim=[0, 1], line_styles=['b-'], dataset_name=dataset_name, methodch3='agglomerative')

        if FLAGS.mode == 'final':
            dataset = clusteringNH.k_means_over_instances(dataset, ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'],
                                                          FLAGS.k,
                                                          'default', 50, 50)
            DataViz.plot_clusters_3d(dataset, ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], 'cluster', ['label'],
                                     dataset_name=dataset_name, method='Final')
            DataViz.plot_silhouette(dataset, 'cluster', 'silhouette', dataset_name=dataset_name, method='Final')
            util.print_latex_statistics_clusters(
                dataset, 'cluster', ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'], 'label')
            del dataset['silhouette']

            output_file = DATA_PATH / f'{base_output_name}'
            dataset.to_csv(output_file)
            print(f"Final results for {input_file_path.name} saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='combined',
                        help="Specify source: 'all' for individual files, or 'combined' for the single combined file.",
                        choices=['all', 'combined'])

    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final, kmeans, kmediods, or agglomerative.",
                        choices=['kmeans', 'kmediods', 'agglomerative', 'final'])

    parser.add_argument('--k', type=int, default=6,
                        help="The selected k number of clusters to apply in 'final' mode.")

    FLAGS, unparsed = parser.parse_known_args()

    main()