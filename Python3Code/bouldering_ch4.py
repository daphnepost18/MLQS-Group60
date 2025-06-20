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

    ACC_X = 'acc_X (m/s^2)'
    ACC_Y = 'acc_Y (m/s^2)'
    ACC_Z = 'acc_Z (m/s^2)'

    GYR_X = 'gyr_X (rad/s)'
    GYR_Y = 'gyr_Y (rad/s)'
    GYR_Z = 'gyr_Z (rad/s)'

    MAG_X = 'mag_X (µT)'
    MAG_Y = 'mag_Y (µT)'
    MAG_Z = 'mag_Z (µT)'

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
            ws_s = [2, 5, 10]  # window sizes in seconds
            window_sizes = [int(s * 1000 / milliseconds_per_instance) for s in ws_s]

            for ws in window_sizes:
                if ws > 0:
                    dataset = NumAbs.abstract_numerical(dataset, [ACC_X, ACC_Y, ACC_Z, GYR_X, GYR_Y, GYR_Z, MAG_X, MAG_Y, MAG_Z], ws, 'mean')
                    dataset = NumAbs.abstract_numerical(dataset, [ACC_X, ACC_Y, ACC_Z, GYR_X, GYR_Y, GYR_Z, MAG_X, MAG_Y, MAG_Z], ws, 'std')
                    dataset = NumAbs.abstract_numerical(dataset, [ACC_X, ACC_Y, ACC_Z, GYR_X, GYR_Y, GYR_Z, MAG_X, MAG_Y, MAG_Z], ws, 'median')

            DataViz.plot_dataset(dataset, [ACC_X, f'{ACC_X}_temp_mean', f'{ACC_X}_temp_std', f'{ACC_X}_temp_median', 'label'],
                                 ['exact', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'points'],
                                 dataset_name=dataset_name, method='Aggregation')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_aggregation_X.csv'
            dataset.to_csv(output_file)
            print(f"Results for {input_file_path.name} saved to: {output_file}")

            # Visualize the Y component of acceleration too
            DataViz.plot_dataset(dataset, [ACC_Y, f'{ACC_Y}_temp_mean', f'{ACC_Y}_temp_std', f'{ACC_Y}_temp_median', 'label'],
                                 ['exact', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'points'],
                                 dataset_name=dataset_name, method='Aggregation')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_aggregation_Y.csv'
            dataset.to_csv(output_file)
            print(f"Results for {input_file_path.name} saved to: {output_file}")

            # Visualize the Z component of acceleration too
            DataViz.plot_dataset(dataset, [ACC_Z, f'{ACC_Z}_temp_mean', f'{ACC_Z}_temp_std', f'{ACC_Z}_temp_median', 'label'],
                                 ['exact', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'points'],
                                 dataset_name=dataset_name, method='Aggregation')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_aggregation_Z.csv'
            dataset.to_csv(output_file)
            print(f"Results for {input_file_path.name} saved to: {output_file}")

# --------------------- GYROSCOPE DATA ---------------------------
            DataViz.plot_dataset(dataset, [f'{GYR_X}_temp_mean', f'{GYR_X}_temp_std', f'{GYR_X}_temp_median', 'label'],
                                 ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
                                 dataset_name=dataset_name, method='Aggregation')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_aggregation_X.csv'
            dataset.to_csv(output_file)
            print(f"Results for {input_file_path.name} saved to: {output_file}")

            DataViz.plot_dataset(dataset, [f'{GYR_Y}_temp_mean', f'{GYR_Y}_temp_std', f'{GYR_Y}_temp_median', 'label'],
                                 ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
                                 dataset_name=dataset_name, method='Aggregation')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_aggregation_Y.csv'
            dataset.to_csv(output_file)
            print(f"Results for {input_file_path.name} saved to: {output_file}")

            DataViz.plot_dataset(dataset, [f'{GYR_Z}_temp_mean', f'{GYR_Z}_temp_std', f'{GYR_Z}_temp_median', 'label'],
                                 ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
                                 dataset_name=dataset_name, method='Aggregation')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_aggregation_Z.csv'
            dataset.to_csv(output_file)
            print(f"Results for {input_file_path.name} saved to: {output_file}")

# --------------------- MAGNETIC FIELD DATA ---------------------------

            DataViz.plot_dataset(dataset, [MAG_X, f'{MAG_X}_temp_mean', f'{MAG_X}_temp_std', f'{MAG_X}_temp_median', 'label'],
                                 ['exact', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'points'],
                                 dataset_name=dataset_name, method='Aggregation')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_aggregation_X.csv'
            dataset.to_csv(output_file)
            print(f"Results for {input_file_path.name} saved to: {output_file}")

            # Visualize the Y component of acceleration too
            DataViz.plot_dataset(dataset, [MAG_Y, f'{MAG_Y}_temp_mean', f'{MAG_Y}_temp_std', f'{MAG_Y}_temp_median', 'label'],
                                 ['exact', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'points'],
                                 dataset_name=dataset_name, method='Aggregation')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_aggregation_Y.csv'
            dataset.to_csv(output_file)
            print(f"Results for {input_file_path.name} saved to: {output_file}")

            # Visualize the Z component of acceleration too
            DataViz.plot_dataset(dataset, [MAG_Z, f'{MAG_Z}_temp_mean', f'{MAG_Z}_temp_std', f'{MAG_Z}_temp_median', 'label'],
                                 ['exact', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'points'],
                                 dataset_name=dataset_name, method='Aggregation')
            output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_aggregation_Z.csv'
            dataset.to_csv(output_file)
            print(f"Results for {input_file_path.name} saved to: {output_file}")


        if FLAGS.mode == 'frequency':
            fs = 5000 / milliseconds_per_instance
            ws = int(50000 / milliseconds_per_instance)

            if ws > 0:
                dataset = FreqAbs.abstract_frequency(dataset, [ACC_X, ACC_Y, ACC_Z, GYR_X, GYR_Y, GYR_Z, MAG_X, MAG_Y, MAG_Z], ws, fs)

                DataViz.plot_dataset(dataset, [f'{ACC_X}_max_freq', f'{ACC_X}_freq_weighted', f'{ACC_X}_pse', 'label'],
                                     ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
                                     dataset_name=dataset_name, method='Frequency')
                output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_frequency_X.csv'
                dataset.to_csv(output_file)
                print(f"Results for {input_file_path.name} saved to: {output_file}")

                DataViz.plot_dataset(dataset, [f'{ACC_Y}_max_freq', f'{ACC_Y}_freq_weighted', f'{ACC_Y}_pse', 'label'],
                                     ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
                                     dataset_name=dataset_name, method='Frequency')
                output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_frequency_Y.csv'
                dataset.to_csv(output_file)
                print(f"Results for {input_file_path.name} saved to: {output_file}")

                DataViz.plot_dataset(dataset, [f'{ACC_Z}_max_freq', f'{ACC_Z}_freq_weighted', f'{ACC_Z}_pse', 'label'],
                                     ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
                                     dataset_name=dataset_name, method='Frequency')
                output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_frequency_Z.csv'
                dataset.to_csv(output_file)
                print(f"Results for {input_file_path.name} saved to: {output_file}")

 # --------------------- GYROSCOPE DATA ---------------------------

                DataViz.plot_dataset(dataset, [f'{GYR_X}_max_freq', f'{GYR_X}_freq_weighted', f'{GYR_X}_pse', 'label'],
                                     ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
                                     dataset_name=dataset_name, method='Frequency')
                output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_frequency_X.csv'
                dataset.to_csv(output_file)
                print(f"Results for {input_file_path.name} saved to: {output_file}")

                DataViz.plot_dataset(dataset, [f'{GYR_Y}_max_freq', f'{GYR_Y}_freq_weighted', f'{GYR_Y}_pse', 'label'],
                                     ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
                                     dataset_name=dataset_name, method='Frequency')
                output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_frequency_Y.csv'
                dataset.to_csv(output_file)
                print(f"Results for {input_file_path.name} saved to: {output_file}")

                DataViz.plot_dataset(dataset, [f'{GYR_Z}_max_freq', f'{GYR_Z}_freq_weighted', f'{GYR_Z}_pse', 'label'],
                                     ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
                                     dataset_name=dataset_name, method='Frequency')
                output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_frequency_Z.csv'
                dataset.to_csv(output_file)
                print(f"Results for {input_file_path.name} saved to: {output_file}")

# --------------------- MAGNETIC FIELD DATA ---------------------------

                DataViz.plot_dataset(dataset, [f'{MAG_X}_max_freq', f'{MAG_X}_freq_weighted', f'{MAG_X}_pse', 'label'],
                                     ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
                                     dataset_name=dataset_name, method='Frequency')
                output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_frequency_X.csv'
                dataset.to_csv(output_file)
                print(f"Results for {input_file_path.name} saved to: {output_file}")

                DataViz.plot_dataset(dataset, [f'{MAG_Y}_max_freq', f'{MAG_Y}_freq_weighted', f'{MAG_Y}_pse', 'label'],
                                     ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
                                     dataset_name=dataset_name, method='Frequency')
                output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_frequency_Y.csv'
                dataset.to_csv(output_file)
                print(f"Results for {input_file_path.name} saved to: {output_file}")

                DataViz.plot_dataset(dataset, [f'{MAG_Z}_max_freq', f'{MAG_Z}_freq_weighted', f'{MAG_Z}_pse', 'label'],
                                     ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
                                     dataset_name=dataset_name, method='Frequency')
                output_file = DATA_PATH / f'{base_output_name.replace(".csv", "")}_frequency_Z.csv'
                dataset.to_csv(output_file)
                print(f"Results for {input_file_path.name} saved to: {output_file}")

        if FLAGS.mode == 'final':
            ws = int(5000 / milliseconds_per_instance)
            if ws == 0:
                print("Window size is zero, skipping final processing.")
                continue

            fs = 2000 / milliseconds_per_instance

            selected_predictor_cols = [c for c in dataset.columns
                                       if 'label' not in c and pd.api.types.is_numeric_dtype(
                    dataset[c]) and not pd.api.types.is_bool_dtype(dataset[c])]
            dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
            dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')

            # CatAbs = CategoricalAbstraction()
            # label_cols = [col for col in dataset.columns if col.startswith('label')]
            # if label_cols:
            #     dataset = CatAbs.abstract_categorical(dataset, label_cols, ['exact'] * len(label_cols), 0.03, ws, 2)
            #
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