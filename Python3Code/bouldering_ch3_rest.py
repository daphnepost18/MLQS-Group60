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

    # NEW: Create a boolean variable based on the --source argument.
    USE_ALL_FILES = (FLAGS.source == 'all')

    # NEW: Conditional logic to select which files to process.
    if USE_ALL_FILES:
        print("Mode: Processing all individual chapter 3 outlier result files...")
        all_chapter3_files = list(DATA_PATH.glob('chapter3_result_outliers_*.csv'))
        # We explicitly filter out the combined file to avoid processing it along with the individuals.
        input_files = [f for f in all_chapter3_files if 'combined' not in f.name]
    else:
        print("Mode: Processing only the combined chapter 3 outlier result file...")
        combined_file = DATA_PATH / 'chapter3_result_outliers_combined.csv'
        if combined_file.exists():
            input_files = [combined_file]
        else:
            input_files = []

    # Check if any files were found to process.
    if not input_files:
        if USE_ALL_FILES:
            print("No individual Chapter 3 outlier result files found. Please run bouldering_ch3_outliers.py first.")
        else:
            print(
                f"Combined file not found at '{DATA_PATH / 'chapter3_result_outliers_combined.csv'}'. Please run bouldering_ch3_outliers.py with the '--source combined' flag.")
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

        DataViz = VisualizeDataset(__file__)

        if len(dataset.index) < 2 or (dataset.index[1] - dataset.index[0]).microseconds == 0:
            print(
                f"Dataset {input_file_path.name} has insufficient data or time resolution. Skipping frequency-dependent operations.")
            milliseconds_per_instance = -1  # Invalid value
        else:
            milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000

        base_output_name = input_file_path.name.replace('chapter3_result_outliers_', 'chapter3_result_final_')
        dataset_name = input_file_path.name.replace('chapter3_result_outliers_', '').replace('.csv', '')

        if FLAGS.mode == 'imputation':
            imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset), 'acc_X (m/s^2)')
            imputed_median_dataset = MisVal.impute_median(copy.deepcopy(dataset), 'acc_X (m/s^2)')
            imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), 'acc_X (m/s^2)')
            DataViz.plot_imputed_values(dataset, ['original', 'mean', 'median', 'interpolation'], 'acc_X (m/s^2)',
                                        imputed_mean_dataset['acc_X (m/s^2)'],
                                        imputed_median_dataset['acc_X (m/s^2)'],
                                        imputed_interpolation_dataset['acc_X (m/s^2)'],
                                        dataset_name=dataset_name, method='Imputation')

        elif FLAGS.mode == 'kalman':
            original_chapter2_fname = input_file_path.name.replace('chapter3_result_outliers_',
                                                                   'chapter2_result_').replace('_final', '')
            original_dataset_path = DATA_PATH / original_chapter2_fname
            try:
                original_dataset = pd.read_csv(original_dataset_path, index_col=0)
                original_dataset.index = pd.to_datetime(original_dataset.index)
                KalFilter = KalmanFilters()
                kalman_dataset = KalFilter.apply_kalman_filter(original_dataset, 'acc_X (m/s^2)')
                DataViz.plot_imputed_values(kalman_dataset, ['original', 'kalman'], 'acc_X (m/s^2)',
                                            kalman_dataset['acc_X (m/s^2)_kalman'],
                                            dataset_name=dataset_name, method='Kalman')
                DataViz.plot_dataset(kalman_dataset, ['acc_X (m/s^2)', 'acc_X (m/s^2)_kalman'], ['exact', 'exact'],
                                     ['line', 'line'],
                                     dataset_name=dataset_name, method='Kalman')
            except IOError as e:
                print(f'Original Chapter 2 file not found: {original_dataset_path.name}. Skipping Kalman filter.')
                continue

        elif FLAGS.mode == 'lowpass':
            if milliseconds_per_instance <= 0:
                print("Cannot perform lowpass filter due to invalid sampling frequency. Skipping.")
                continue
            fs = float(1000) / milliseconds_per_instance
            cutoff = 1.5
            new_dataset = LowPass.low_pass_filter(copy.deepcopy(dataset), 'acc_X (m/s^2)', fs, cutoff, order=10)
            DataViz.plot_dataset(
                new_dataset.iloc[int(0.4 * len(new_dataset.index)):int(0.43 * len(new_dataset.index)), :],
                ['acc_X (m/s^2)', 'acc_X (m/s^2)_lowpass'], ['exact', 'exact'], ['line', 'line'],
                dataset_name=dataset_name, method='Lowpass')

        elif FLAGS.mode == 'PCA':
            for col in [c for c in dataset.columns if 'label' not in c]:
                dataset = MisVal.impute_interpolate(dataset, col)
            selected_predictor_cols = [c for c in dataset.columns if 'label' not in c and pd.api.types.is_numeric_dtype(
                dataset[c]) and not pd.api.types.is_bool_dtype(dataset[c])]
            if not selected_predictor_cols:
                print("No numeric columns for PCA. Skipping.")
                continue
            pc_values = PCA.determine_pc_explained_variance(dataset, selected_predictor_cols)
            DataViz.plot_xy(x=[range(1, len(selected_predictor_cols) + 1)], y=[pc_values],
                            xlabel='principal component number',
                            ylabel='explained variance', ylim=[0, 1], line_styles=['b-'], dataset_name=dataset_name,
                            methodch3='PCA')
            n_pcs = min(7, len(selected_predictor_cols))
            if n_pcs > 0:
                dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)
                DataViz.plot_dataset(dataset, ['pca_', 'label'], ['like', 'like'], ['line', 'points'],
                                     dataset_name=dataset_name, method='PCA')

        elif FLAGS.mode == 'final':
            for col in [c for c in dataset.columns if 'label' not in c]:
                dataset = MisVal.impute_interpolate(dataset, col)

            if milliseconds_per_instance > 0:
                fs = float(1000) / milliseconds_per_instance
                cutoff = 1.5
                periodic_measurements = ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)', "gyr_X (rad/s)",
                                         "gyr_Y (rad/s)", "gyr_Z (rad/s)", "mag_X (µT)", "mag_Y (µT)", "mag_Z (µT)"]
                for col in periodic_measurements:
                    if col in dataset.columns and pd.api.types.is_numeric_dtype(dataset[col]):
                        dataset = LowPass.low_pass_filter(dataset, col, fs, cutoff, order=10)
                        dataset[col] = dataset[col + '_lowpass']
                        del dataset[col + '_lowpass']

            selected_predictor_cols = [c for c in dataset.columns if 'label' not in c and pd.api.types.is_numeric_dtype(
                dataset[c]) and not pd.api.types.is_bool_dtype(dataset[c])]
            if not selected_predictor_cols:
                print("No numeric columns for final PCA. Saving file without PCA.")
            else:
                n_pcs = min(7, len(selected_predictor_cols))
                if n_pcs > 0:
                    dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)

            output_file = DATA_PATH / f'{base_output_name}'
            dataset.to_csv(output_file)
            print(f"Final results for {input_file_path.name} saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='combined',
                        help="Specify source: 'all' for individual files, or 'combined' for the single combined file.",
                        choices=['all', 'combined'])

    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final, imputation, lowpass, kalman or PCA",
                        choices=['lowpass', 'imputation', 'PCA', 'final', 'kalman'])

    FLAGS, unparsed = parser.parse_known_args()

    main()