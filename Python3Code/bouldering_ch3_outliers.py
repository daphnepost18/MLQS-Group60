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

    # Get all chapter 2 result files as input
    input_files = list(DATA_PATH.glob('chapter2_result_*.csv'))

    if not input_files:
        print("No Chapter 2 result files found. Please run Chapter 2 first.")
        return

    # Create the outlier classes once
    OutlierDistr = DistributionBasedOutlierDetection()
    OutlierDist = DistanceBasedOutlierDetection()

    for input_file_path in input_files:
        print(f"\n--- Processing file: {input_file_path.name} ---")
        try:
            dataset = pd.read_csv(input_file_path, index_col=0)
            dataset.index = pd.to_datetime(dataset.index)

        except IOError as e:
            print(f'File not found: {input_file_path.name}. Skipping.')
            continue # Skip to the next file

        # We'll create an instance of our visualization class to plot the results for each dataset.
        DataViz = VisualizeDataset(__file__)

        # Determine the columns we want to experiment on.
        outlier_columns = ['acc_X (m/s^2)','acc_Y (m/s^2)','acc_Z (m/s^2)']

        # Construct the base name for the output file
        base_output_name = input_file_path.name.replace('chapter2_result_', 'chapter3_result_outliers_')
        dataset_name = input_file_path.name.replace('chapter2_result_', '')

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
                    col, col + '_mixture'], ['exact', 'exact'], ['line', 'points'], dataset_name=dataset_name, method='Mixture')
                # This requires:
                # n_data_points * n_data_points * point_size =
                # 31839 * 31839 * 32 bits = ~4GB available memory
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
                    print(
                        'Not enough memory available for simple distance-based outlier detection...')
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
                        'exact', 'exact'], ['line', 'points'], dataset_name=dataset_name, method='LOF')
                except MemoryError as e:
                    print('Not enough memory available for lof...')
                    print('Skipping.')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_lof_K{FLAGS.K}.csv'
            dataset.to_csv(output_file)
            print(f"Results saved to: {output_file}")

        elif FLAGS.mode == 'final':
            # We use Chauvenet's criterion for the final version and apply it to all but the label data...
            for col in [c for c in dataset.columns if not 'label' in c]:
                print(f'Applying Chauvenet to: {col}')
                dataset = OutlierDistr.chauvenet(dataset, col, FLAGS.C)
                dataset.loc[dataset[f'{col}_outlier'] == True, col] = np.nan
                del dataset[col + '_outlier']

            output_file = DATA_PATH / f'{base_output_name}' # Uses the base name directly for final output
            dataset.to_csv(output_file)
            print(f"Final results for {input_file_path.name} saved to: {output_file}")


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: LOF, distance, mixture, chauvenet or final \
                        'LOF' applies the Local Outlier Factor to a single variable \
                        'distance' applies a distance based outlier detection method to a single variable \
                        'mixture' applies a mixture model to detect outliers for a single variable\
                        'chauvenet' applies Chauvenet outlier detection method to a single variable \
                        'final' is used for the next chapter",
                        choices=['LOF', 'distance', 'mixture', 'chauvenet', 'final'])

    parser.add_argument('--C', type=float, default=2,
                        help="Chauvenet: C parameter")

    parser.add_argument('--K', type=int, default=5,
                        help="Local Outlier Factor:  K is the number of neighboring points considered")

    parser.add_argument('--dmin', type=float, default=0.10,
                        help="Simple distance based:  dmin is ... ")

    parser.add_argument('--fmin', type=float, default=0.99,
                        help="Simple distance based:  fmin is ... ")

    FLAGS, unparsed = parser.parse_known_args()

    main()
