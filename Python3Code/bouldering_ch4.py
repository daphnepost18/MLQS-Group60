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

    input_files = list(DATA_PATH.glob('chapter3_result_final_*.csv'))

    if not input_files:
        print("No Chapter 3 final result files found. Please run bouldering_ch3_rest.py with --mode final first.")
        return

    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()
    # DO NOT create CatAbs here. It must be created inside the loop.

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
        dataset_name = input_file_path.name.replace('chapter3_result_final_', 'chapter4_result_').replace('.csv', '')

        if FLAGS.mode == 'aggregation':
            window_sizes = [int(5000 / milliseconds_per_instance),
                            int(30000 / milliseconds_per_instance),
                            int(300000 / milliseconds_per_instance)]
            ACC_X = 'acc_X (m/s^2)'
            for ws in window_sizes:
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
            dataset = FreqAbs.abstract_frequency(dataset, [ACC_X], ws, fs)
            DataViz.plot_dataset(dataset, [f'{ACC_X}_max_freq', f'{ACC_X}_freq_weighted', f'{ACC_X}_pse', 'label'],
                                 ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
                                 dataset_name=dataset_name, method='Frequency')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_frequency.csv'
            dataset.to_csv(output_file)
            print(f"Results for {input_file_path.name} saved to: {output_file}")

        if FLAGS.mode == 'final':
            ws = int(3000 / milliseconds_per_instance)
            fs = 100 / milliseconds_per_instance

            selected_predictor_cols = [c for c in dataset.columns
                                       if 'label' not in c and pd.api.types.is_numeric_dtype(dataset[c]) and not pd.api.types.is_bool_dtype(dataset[c])]
            dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
            dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')

            ACC_X = 'acc_X (m/s^2)'
            GYR_X = 'gyr_X (rad/s)'
            MAG_X = 'mag_X (µT)'
            LOC_HEIGHT = 'loc_Height (m)'
            PCA_1 = 'pca_1'
            DataViz.plot_dataset(dataset, [ACC_X, GYR_X, MAG_X, LOC_HEIGHT, PCA_1, 'label'],
                                 ['like', 'like', 'like', 'like', 'like', 'like'],
                                 ['line', 'line', 'line', 'line', 'line', 'points'],
                                 dataset_name=dataset_name, method='Final')

            # FIX: Instantiate CategoricalAbstraction INSIDE the loop.
            # This creates a fresh, stateless object for each file.
            CatAbs = CategoricalAbstraction()

            label_cols = [col for col in dataset.columns if col.startswith('label')]
            if label_cols:
                dataset = CatAbs.abstract_categorical(dataset, ['label'], ['like'], 0.03,
                                                      ws, 2)
            else:
                print(f"Warning: No label columns found in {input_file_path.name}. Skipping categorical abstraction.")

            periodic_predictor_cols = ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)',
                                       "gyr_X (rad/s)","gyr_Y (rad/s)","gyr_Z (rad/s)",
                                       "mag_X (µT)","mag_Y (µT)","mag_Z (µT)",
                                       "loc_Height (m)","loc_Velocity (m/s)"]
            periodic_measurements_for_freq = [col for col in periodic_predictor_cols
                                              if col in dataset.columns and pd.api.types.is_numeric_dtype(dataset[col])]
            if periodic_measurements_for_freq:
                dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), periodic_measurements_for_freq,
                                                     ws, fs)
            else:
                print(f"Warning: No valid periodic measurements found in {input_file_path.name}. Skipping.")

            window_overlap = 0.9
            skip_points = int((1 - window_overlap) * ws)
            if skip_points == 0:
                skip_points = 1
            dataset = dataset.iloc[::skip_points, :]

            output_file = DATA_PATH / f'{base_output_name}'
            dataset.to_csv(output_file)
            print(f"Final results for {input_file_path.name} saved to: {output_file}")

    print("--- Total processing time: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final, aggregation or freq \
                        'aggregation' studies the effect of several aggeregation methods \
                        'frequency' applies a Fast Fourier transformation to a single variable \
                        'final' is used for the next chapter ", choices=['aggregation', 'frequency', 'final'])

    FLAGS, unparsed = parser.parse_known_args()

    main()