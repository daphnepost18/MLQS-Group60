##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import sys
import copy
import pandas as pd
import time
from pathlib import Path
import argparse

from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction

# Read the result from the previous chapter
DATA_PATH = Path('./intermediate_datafiles_bouldering/')


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    print_flags()

    start_time = time.time()

    USE_ALL_FILES = (FLAGS.source == 'all')

    if USE_ALL_FILES:
        print("Mode: Processing all individual chapter 3 final result files...")
        all_chapter3_files = list(DATA_PATH.glob('chapter3_result_final_*.csv'))
        input_files = [f for f in all_chapter3_files if 'combined' not in f.name]
    else:
        print("Mode: Processing only the combined chapter 3 final result file...")
        combined_file = DATA_PATH / 'chapter3_result_final_combined.csv'
        if combined_file.exists():
            input_files = [combined_file]
        else:
            input_files = []

    if not input_files:
        if USE_ALL_FILES:
            print(
                "No individual Chapter 3 final result files found. Please run bouldering_ch3_rest.py with '--mode final' first.")
        else:
            print(
                f"Combined file not found at '{DATA_PATH / 'chapter3_result_final_combined.csv'}'. Please run bouldering_ch3_rest.py with '--mode final' and '--source combined'.")
        return

    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()

    for input_file_path in input_files:
        print(f"\n--- Processing file: {input_file_path.name} ---")
        try:
            dataset = pd.read_csv(input_file_path, index_col=0)
            dataset.index = pd.to_datetime(dataset.index)
        except IOError as e:
            print(f'File not found: {input_file_path.name}. Skipping.')
            continue

        DataViz = VisualizeDataset(__file__)

        if len(dataset.index) < 2:
            print(f"Dataset {input_file_path.name} has less than 2 data points. Skipping.")
            continue

        milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).total_seconds() * 1000
        if milliseconds_per_instance == 0:
            print(f"Dataset {input_file_path.name} has effectively zero milliseconds per instance. Skipping.")
            continue

        base_output_name = input_file_path.name.replace('chapter3_result_final_', 'chapter4_result_')
        dataset_name = input_file_path.name.replace('chapter3_result_final_', '').replace('.csv', '')

        if FLAGS.mode == 'aggregation':
            ws_s = [5, 30, 300]  # window sizes in seconds
            window_sizes = [int(s * 1000 / milliseconds_per_instance) for s in ws_s]
            ACC_X = 'acc_X (m/s^2)'
            for ws in window_sizes:
                if ws > 0:
                    dataset = NumAbs.abstract_numerical(dataset, [ACC_X], ws, 'mean')
                    dataset = NumAbs.abstract_numerical(dataset, [ACC_X], ws, 'std')
            DataViz.plot_dataset(dataset, [ACC_X, f'{ACC_X}_temp_mean', f'{ACC_X}_temp_std', 'label'],
                                 ['exact', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
                                 dataset_name=dataset_name, method='Aggregation')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_aggregation.csv'
            dataset.to_csv(output_file)
            print(f"Results for {input_file_path.name} saved to: {output_file}")

        if FLAGS.mode == 'frequency':
            fs = 1000 / milliseconds_per_instance
            ws = int(10000 / milliseconds_per_instance)
            ACC_X = 'acc_X (m/s^2)'
            if ws > 0:
                dataset = FreqAbs.abstract_frequency(dataset, [ACC_X], ws, fs)
                DataViz.plot_dataset(dataset, [f'{ACC_X}_max_freq', f'{ACC_X}_freq_weighted', f'{ACC_X}_pse', 'label'],
                                     ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
                                     dataset_name=dataset_name, method='Frequency')
                output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_frequency.csv'
                dataset.to_csv(output_file)
                print(f"Results for {input_file_path.name} saved to: {output_file}")

        if FLAGS.mode == 'final':
            ws = int(3000 / milliseconds_per_instance)
            if ws == 0:
                print("Window size is zero, skipping final processing.")
                continue

            fs = 1000 / milliseconds_per_instance

            selected_predictor_cols = [c for c in dataset.columns
                                       if 'label' not in c and pd.api.types.is_numeric_dtype(
                    dataset[c]) and not pd.api.types.is_bool_dtype(dataset[c])]
            dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
            dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')

            CatAbs = CategoricalAbstraction()
            label_cols = [col for col in dataset.columns if col.startswith('label')]
            if label_cols:
                dataset = CatAbs.abstract_categorical(dataset, label_cols, ['exact'] * len(label_cols), 0.03, ws, 2)

            periodic_predictor_cols = ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)',
                                       "gyr_X (rad/s)", "gyr_Y (rad/s)", "gyr_Z (rad/s)",
                                       "mag_X (µT)", "mag_Y (µT)", "mag_Z (µT)"]

            periodic_measurements_for_freq = [col for col in periodic_predictor_cols if
                                              col in dataset.columns and pd.api.types.is_numeric_dtype(dataset[col])]

            if periodic_measurements_for_freq:
                dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), periodic_measurements_for_freq, ws, fs)

            window_overlap = 0.9
            skip_points = int((1 - window_overlap) * ws)
            if skip_points == 0: skip_points = 1
            dataset = dataset.iloc[::skip_points, :]

            output_file = DATA_PATH / f'{base_output_name}'
            dataset.to_csv(output_file)
            print(f"Final results for {input_file_path.name} saved to: {output_file}")

    print("--- Total processing time: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='combined',
                        help="Specify source: 'all' for individual files, or 'combined' for the single combined file.",
                        choices=['all', 'combined'])

    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final, aggregation or frequency.",
                        choices=['aggregation', 'frequency', 'final'])

    FLAGS, unparsed = parser.parse_known_args()

    main()