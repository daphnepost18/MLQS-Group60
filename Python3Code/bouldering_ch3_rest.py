##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

import sys
import copy
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters

# Set up the file names and locations.
DATA_PATH = Path('./intermediate_datafiles_bouldering/')


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    print_flags()

    # Get all chapter 3 outlier result files as input
    input_files = list(DATA_PATH.glob('chapter3_result_outliers_*.csv'))

    if not input_files:
        print("No Chapter 3 outlier result files found. Please run bouldering_ch3_outliers.py first.")
        return

    # Create instances of the transformation classes once
    MisVal = ImputationMissingValues()
    LowPass = LowPassFilter()
    PCA = PrincipalComponentAnalysis()

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

        # Compute the number of milliseconds covered by an instance based on the first two rows
        milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000

        # Construct the base name for the output file
        base_output_name = input_file_path.name.replace('chapter3_result_outliers_', 'chapter3_result_final_')

        if FLAGS.mode == 'imputation':
            # Let us impute the missing values and plot an example.

            # Note: For multiple files, this plot will overwrite the previous one.
            # You might want to save plots with unique names or view them one by one.
            imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset), 'acc_X (m/s^2)')
            imputed_median_dataset = MisVal.impute_median(copy.deepcopy(dataset), 'acc_X (m/s^2)')
            imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), 'acc_X (m/s^2)')

            DataViz.plot_imputed_values(dataset, ['original', 'mean', 'median', 'interpolation'], 'acc_X (m/s^2)',
                                        imputed_mean_dataset['acc_X (m/s^2)'],
                                        imputed_median_dataset['acc_X (m/s^2)'],
                                        imputed_interpolation_dataset['acc_X (m/s^2)'])
            # No specific output CSV for this mode, it's illustrative.

        elif FLAGS.mode == 'kalman':
            # Using the result from Chapter 2, let us try the Kalman filter on the attribute and study the result.

            # Dynamically derive the original Chapter 2 file path
            # Example: chapter3_result_outliers_participant1_Easy1_250.csv -> chapter2_result_participant1_Easy1_250.csv
            original_chapter2_fname = input_file_path.name.replace('chapter3_result_outliers_', 'chapter2_result_')
            original_dataset_path = DATA_PATH / original_chapter2_fname

            try:
                original_dataset = pd.read_csv(original_dataset_path, index_col=0)
                original_dataset.index = pd.to_datetime(original_dataset.index)
            except IOError as e:
                print(f'Original Chapter 2 file not found: {original_dataset_path.name}. Skipping Kalman filter for this dataset.')
                continue

            KalFilter = KalmanFilters()
            kalman_dataset = KalFilter.apply_kalman_filter(
                original_dataset, 'acc_X (m/s^2)')
            DataViz.plot_imputed_values(kalman_dataset, [
                'original', 'kalman'], 'acc_X (m/s^2)', kalman_dataset['acc_X (m/s^2)_kalman'])
            DataViz.plot_dataset(kalman_dataset, ['acc_X (m/s^2)', 'acc_X (m/s^2)_kalman'], [
                'exact', 'exact'], ['line', 'line'])

            # We ignore the Kalman filter output for now...

        elif FLAGS.mode == 'lowpass':
            # Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz

            # Determine the sampling frequency.
            fs = float(1000) / milliseconds_per_instance
            cutoff = 1.5
            # Let us study acc_X (m/s^2):
            new_dataset = LowPass.low_pass_filter(copy.deepcopy(
                dataset), 'acc_X (m/s^2)', fs, cutoff, order=10)
            DataViz.plot_dataset(new_dataset.iloc[int(0.4 * len(new_dataset.index)):int(0.43 * len(new_dataset.index)), :],
                                 ['acc_X (m/s^2)', 'acc_X (m/s^2)_lowpass'], ['exact', 'exact'], ['line', 'line'])

        elif FLAGS.mode == 'PCA':
            # first impute again, as PCA can not deal with missing values
            for col in [c for c in dataset.columns if not 'label' in c]:
                dataset = MisVal.impute_interpolate(dataset, col)

            selected_predictor_cols = [c for c in dataset.columns if not ('label' in c)]
            pc_values = PCA.determine_pc_explained_variance(
                dataset, selected_predictor_cols)

            # Plot the variance explained.
            DataViz.plot_xy(x=[range(1, len(selected_predictor_cols) + 1)], y=[pc_values],
                            xlabel='principal component number', ylabel='explained variance',
                            ylim=[0, 1], line_styles=['b-'])

            # We select 7 as the best number of PC's as this explains most of the variance

            n_pcs = 7

            dataset = PCA.apply_pca(copy.deepcopy(
                dataset), selected_predictor_cols, n_pcs)

            # And we visualize the result of the PC's
            DataViz.plot_dataset(dataset, ['pca_', 'label'], [
                'like', 'like'], ['line', 'points'])

        elif FLAGS.mode == 'final':
            # Now, for the final version.
            # We first start with imputation by interpolation

            for col in [c for c in dataset.columns if not 'label' in c]:
                dataset = MisVal.impute_interpolate(dataset, col)

            # And now let us include all LOWPASS measurements that have a form of periodicity (and filter them):
            periodic_measurements = ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)',
                                     "gyr_X (rad/s)","gyr_Y (rad/s)","gyr_Z (rad/s)",
                                     "mag_X (µT)","mag_Y (µT)","mag_Z (µT)",
                                     "loc_Height (m)","loc_Velocity (m/s)"]

            # Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz

            # Determine the sampling frequency.
            fs = float(1000) / milliseconds_per_instance
            cutoff = 1.5

            for col in periodic_measurements:
                dataset = LowPass.low_pass_filter(
                    dataset, col, fs, cutoff, order=10)
                dataset[col] = dataset[col + '_lowpass']
                del dataset[col + '_lowpass']

            # We used the optimal found parameter n_pcs = 7, to apply PCA to the final dataset

            selected_predictor_cols = [c for c in dataset.columns if not ('label' in c)]

            n_pcs = 7

            dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)

            # And the overall final dataset:
            DataViz.plot_dataset(dataset,
                                 ['acc_', 'gyr_', 'mag_', 'loc_', 'pca_','label'],
                                 ['like', 'like', 'like', 'like', 'like', 'like'],
                                 ['line', 'line', 'line', 'line', 'points', 'points'])

            # Store the final outcome.
            output_file = DATA_PATH / f'{base_output_name}' # Uses the base name directly for final output
            dataset.to_csv(output_file)
            print(f"Final results for {input_file_path.name} saved to: {output_file}")


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final, imputation, lowpass or PCA \
                        'lowpass' applies the lowpass-filter to a single variable \
                        'imputation' is used for the next chapter \
                        'PCA' is to study the effect of PCA and plot the results\
                        'final' is used for the next chapter", choices=['lowpass', 'imputation', 'PCA', 'final'])

    FLAGS, unparsed = parser.parse_known_args()

    main()
