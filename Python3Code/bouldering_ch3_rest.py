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
            continue

        # We'll create an instance of our visualization class to plot the results for each dataset.
        DataViz = VisualizeDataset(__file__)

        # Ensure there are enough data points and a valid time difference
        if len(dataset.index) < 2:
            print(f"Dataset {input_file_path.name} has less than 2 data points. Cannot determine sampling frequency. Skipping.")
            continue

        time_difference_microseconds = (dataset.index[1] - dataset.index[0]).microseconds
        if time_difference_microseconds == 0:
            print(f"Dataset {input_file_path.name} has zero time difference between first two points. Cannot determine sampling frequency. Skipping.")
            continue

        milliseconds_per_instance = time_difference_microseconds / 1000
        if milliseconds_per_instance == 0:
            print(f"Dataset {input_file_path.name} has effectively zero milliseconds per instance. Cannot determine sampling frequency. Skipping.")
            continue


        # Construct the base name for the output file
        base_output_name = input_file_path.name.replace('chapter3_result_outliers_', 'chapter3_result_final_')
        dataset_name = input_file_path.name.replace('chapter3_result_outliers_', '').replace('.csv', '')

        if FLAGS.mode == 'imputation':
            # Let us impute the missing values and plot an example.
            imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset), 'acc_X (m/s^2)')
            imputed_median_dataset = MisVal.impute_median(copy.deepcopy(dataset), 'acc_X (m/s^2)')
            imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), 'acc_X (m/s^2)')

            DataViz.plot_imputed_values(dataset, ['original', 'mean', 'median', 'interpolation'], 'acc_X (m/s^2)',
                                        imputed_mean_dataset['acc_X (m/s^2)'],
                                        imputed_median_dataset['acc_X (m/s^2)'],
                                        imputed_interpolation_dataset['acc_X (m/s^2)'],
                                        dataset_name=dataset_name, method='Imputation')

        elif FLAGS.mode == 'kalman':
            # Using the result from Chapter 2, let us try the Kalman filter on the attribute and study the result.

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
                'original', 'kalman'], 'acc_X (m/s^2)', kalman_dataset['acc_X (m/s^2)_kalman'],
                                        dataset_name=dataset_name, method='Kalman')
            DataViz.plot_dataset(kalman_dataset, ['acc_X (m/s^2)', 'acc_X (m/s^2)_kalman'], [
                'exact', 'exact'], ['line', 'line'], dataset_name=dataset_name, method='Kalman')

        elif FLAGS.mode == 'lowpass':
            fs = float(1000) / milliseconds_per_instance
            cutoff = 1.5
            normalized_cutoff = cutoff / (fs / 2)

            new_dataset = LowPass.low_pass_filter(copy.deepcopy( dataset), 'acc_X (m/s^2)', fs, normalized_cutoff, order=10)

            DataViz.plot_dataset(
                new_dataset.iloc[int(0.4 * len(new_dataset.index)):int(0.43 * len(new_dataset.index)), :],
                ['acc_X (m/s^2)', 'acc_X (m/s^2)_lowpass'], ['exact', 'exact'], ['line', 'line'],
                        dataset_name=dataset_name, method='Lowpass')

        elif FLAGS.mode == 'PCA':
            for col in [c for c in dataset.columns if not 'label' in c]:
                dataset = MisVal.impute_interpolate(dataset, col)

            # Ensure selected_predictor_cols contains ONLY numeric, non-boolean, non-label columns for PCA
            selected_predictor_cols = [c for c in dataset.columns
                                       if not ('label' in c) and
                                       pd.api.types.is_numeric_dtype(dataset[c]) and # Must be numeric
                                       not pd.api.types.is_bool_dtype(dataset[c])] # Must not be boolean

            if not selected_predictor_cols:
                print("No numeric columns found for PCA. Skipping PCA for this dataset.")
                continue

            pc_values = PCA.determine_pc_explained_variance(
                dataset, selected_predictor_cols)

            DataViz.plot_xy(x=[range(1, len(selected_predictor_cols) + 1)], y=[pc_values],
                            xlabel='principal component number', ylabel='explained variance',
                            ylim=[0, 1], line_styles=['b-'], dataset_name=dataset_name, methodch3='PCA')

            n_pcs = 7
            if n_pcs > len(selected_predictor_cols):
                print(f"Warning: n_pcs ({n_pcs}) is greater than the number of available numeric columns ({len(selected_predictor_cols)}). Adjusting n_pcs to {len(selected_predictor_cols)}.")
                n_pcs = len(selected_predictor_cols)
            if n_pcs == 0:
                print("No principal components can be computed as there are no numeric columns available. Skipping PCA application.")
                continue

            dataset = PCA.apply_pca(copy.deepcopy(
                dataset), selected_predictor_cols, n_pcs)

            DataViz.plot_dataset(dataset, ['pca_', 'label'], [
                'like', 'like'], ['line', 'points'], dataset_name=dataset_name, method='PCA')

        elif FLAGS.mode == 'final':
            for col in [c for c in dataset.columns if not 'label' in c]:
                dataset = MisVal.impute_interpolate(dataset, col)

            periodic_measurements = ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)',
                                     "gyr_X (rad/s)","gyr_Y (rad/s)","gyr_Z (rad/s)",
                                     "mag_X (µT)","mag_Y (µT)","mag_Z (µT)",
                                     "loc_Height (m)","loc_Velocity (m/s)"]

            if len(dataset.index) < 2:
                print(f"Dataset {input_file_path.name} has less than 2 data points. Cannot determine sampling frequency. Skipping lowpass filter.")
                pass
            else:
                time_difference_microseconds = (dataset.index[1] - dataset.index[0]).microseconds
                if time_difference_microseconds == 0:
                    print(f"Dataset {input_file_path.name} has zero time difference. Cannot determine sampling frequency. Skipping lowpass filter.")
                    pass
                else:
                    milliseconds_per_instance = time_difference_microseconds / 1000
                    fs = float(1000) / milliseconds_per_instance
                    cutoff = 1.5
                    normalized_cutoff = cutoff / (fs / 2)

                    for col in periodic_measurements:
                        if col in dataset.columns and pd.api.types.is_numeric_dtype(dataset[col]):
                            dataset = LowPass.low_pass_filter(
                                dataset, col, fs, normalized_cutoff, order=10)
                            dataset[col] = dataset[col + '_lowpass']
                            del dataset[col + '_lowpass']
                        else:
                            print(f"Warning: Periodic measurement column '{col}' not found or not numeric in dataset. Skipping lowpass filter for this column.")


            # Ensure selected_predictor_cols contains ONLY numeric, non-boolean, non-label columns for final PCA
            selected_predictor_cols = [c for c in dataset.columns
                                       if not ('label' in c) and
                                       pd.api.types.is_numeric_dtype(dataset[c]) and # Must be numeric
                                       not pd.api.types.is_bool_dtype(dataset[c])] # Must not be boolean

            if not selected_predictor_cols:
                print("No numeric columns found for final PCA. Skipping PCA application for this dataset.")
                output_file = DATA_PATH / f'{base_output_name}'
                dataset.to_csv(output_file)
                print(f"Results for {input_file_path.name} (PCA skipped) saved to: {output_file}")
                continue

            n_pcs = 7
            if n_pcs > len(selected_predictor_cols):
                print(f"Warning: n_pcs ({n_pcs}) is greater than the number of available numeric columns ({len(selected_predictor_cols)}). Adjusting n_pcs to {len(selected_predictor_cols)}.")
                n_pcs = len(selected_predictor_cols)
            if n_pcs == 0:
                print("No principal components can be computed as there are no numeric columns available. Skipping PCA application.")
                output_file = DATA_PATH / f'{base_output_name}'
                dataset.to_csv(output_file)
                print(f"Results for {input_file_path.name} (PCA skipped) saved to: {output_file}")
                continue

            dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)

            DataViz.plot_dataset(dataset,
                                 ['acc_', 'gyr_', 'mag_', 'loc_', 'pca_','label'],
                                 ['like', 'like', 'like', 'like', 'like', 'like'],
                                 ['line', 'line', 'line', 'line', 'points', 'points'],
                                 dataset_name=dataset_name, method='final')

            output_file = DATA_PATH / f'{base_output_name}'
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