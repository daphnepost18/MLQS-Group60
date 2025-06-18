##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

from util.VisualizeDataset import VisualizeDataset
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
import sys
import copy
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Set up file names and locations.
DATA_PATH = Path('./intermediate_datafiles_bouldering/')


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    print_flags()

    # Create a boolean variable based on the new --source argument.
    USE_ALL_FILES = (FLAGS.source == 'all')

    if USE_ALL_FILES:
        print("Mode: Processing all individual chapter 2 result files...")
        all_chapter2_files = list(DATA_PATH.glob('chapter2_result_*.csv'))
        input_files = [f for f in all_chapter2_files if 'combined' not in f.name]
    else:
        print("Mode: Processing only the combined chapter 2 result file...")
        combined_file = DATA_PATH / 'chapter2_result_combined.csv'
        if combined_file.exists():
            input_files = [combined_file]
        else:
            input_files = []

    if not input_files:
        if USE_ALL_FILES:
            print("No individual Chapter 2 result files found. Please run Chapter 2 first.")
        else:
            print(
                f"Combined file not found at '{DATA_PATH / 'chapter2_result_combined.csv'}'. Please run Chapter 2 to generate it.")
        return

    # Create the outlier classes once
    OutlierDistr = DistributionBasedOutlierDetection()
    OutlierDist = DistanceBasedOutlierDetection()

    # The rest of the script remains the same, as it can loop through one or many files.
    for input_file_path in input_files:
        print(f"\n--- Processing file: {input_file_path.name} ---")
        try:
            dataset = pd.read_csv(input_file_path, index_col=0)
            dataset.index = pd.to_datetime(dataset.index)

        except IOError as e:
            print(f'File not found: {input_file_path.name}. Skipping.')
            continue

        DataViz = VisualizeDataset(__file__)
        outlier_columns = ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)']
        base_output_name = input_file_path.name.replace('chapter2_result_', 'chapter3_result_outliers_')
        dataset_name = input_file_path.name.replace('chapter2_result_', '').replace('.csv', '')

        if FLAGS.mode == 'chauvenet':
            for col in outlier_columns:
                print(f"Applying Chauvenet outlier criteria for column {col}")
                dataset = OutlierDistr.chauvenet(dataset, col, FLAGS.C)
                DataViz.plot_binary_outliers(
                    dataset, col, col + '_outlier', dataset_name=dataset_name, method='Chauvenet')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_chauvenet_C{FLAGS.C}.csv'
            dataset.to_csv(output_file)
            print(f"Results saved to: {output_file}")

        elif FLAGS.mode == 'mixture':
            for col in outlier_columns:
                print(f"Applying mixture model for column {col}")
                dataset = OutlierDistr.mixture_model(dataset, col)
                DataViz.plot_dataset(dataset, [
                    col, col + '_mixture'], ['exact', 'exact'], ['line', 'points'], dataset_name=dataset_name,
                                     method='Mixture')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_mixture.csv'
            dataset.to_csv(output_file)
            print(f"Results saved to: {output_file}")

        elif FLAGS.mode == 'distance':
            for col in outlier_columns:
                try:
                    dataset = OutlierDist.simple_distance_based(
                        dataset, [col], 'euclidean', FLAGS.dmin, FLAGS.fmin)
                    DataViz.plot_binary_outliers(
                        dataset, col, 'simple_dist_outlier', dataset_name=dataset_name, method='DistanceBased')
                except MemoryError as e:
                    print('Not enough memory available for simple distance-based outlier detection...')
                    print('Skipping.')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_distance_dmin{FLAGS.dmin}_fmin{FLAGS.fmin}.csv'
            dataset.to_csv(output_file)
            print(f"Results saved to: {output_file}")

        elif FLAGS.mode == 'LOF':
            for col in outlier_columns:
                try:
                    dataset = OutlierDist.local_outlier_factor(
                        dataset, [col], 'euclidean', FLAGS.K)
                    DataViz.plot_dataset(dataset, [col, 'lof'], [
                        'exact', 'exact'], ['line', 'points'], dataset_name=dataset_name, method=f"LOF_K{FLAGS.K}")
                except MemoryError as e:
                    print('Not enough memory available for lof...')
                    print('Skipping.')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_lof_K{FLAGS.K}.csv'
            dataset.to_csv(output_file)
            print(f"Results saved to: {output_file}")

        elif FLAGS.mode == 'final':
            for col in [c for c in dataset.columns if 'label' not in c]:
                print(f'Applying Local Outlier Factor (LOF) to: {col}')
                try:
                    dataset = OutlierDist.local_outlier_factor(dataset, [col], 'euclidean', FLAGS.K)
                    # Identify outliers: a rule of thumb is to consider points with an LOF score > 1.5 as outliers.
                    # You can adjust this threshold if needed.
                    outlier_indices = dataset[dataset['lof'] > 1.5].index
                    # Replace the identified outlier values in the original column with NaN.
                    dataset.loc[outlier_indices, col] = np.nan
                    del dataset['lof']

                except MemoryError as e:
                    print(f'Not enough memory available for LOF on column {col}... Skipping.')

            output_file_name = f'{base_output_name.replace(".csv", "")}_final_lof_K{FLAGS.K}.csv'
            output_file = DATA_PATH / output_file_name
            dataset.to_csv(output_file)
            print(f"Final results for {input_file_path.name} saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='combined',
                        help="Specify the source files to process: 'all' for individual files, or 'combined' for the single combined file.",
                        choices=['all', 'combined'])

    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: LOF, distance, mixture, chauvenet or final",
                        choices=['LOF', 'distance', 'mixture', 'chauvenet', 'final'])

    parser.add_argument('--C', type=float, default=2,
                        help="Chauvenet: C parameter")

    parser.add_argument('--K', type=int, default=5,
                        help="Local Outlier Factor: K is the number of neighboring points considered")

    parser.add_argument('--dmin', type=float, default=0.10,
                        help="Simple distance based: dmin is ... ")

    parser.add_argument('--fmin', type=float, default=0.99,
                        help="Simple distance based: fmin is ... ")

    FLAGS, unparsed = parser.parse_known_args()

    main()