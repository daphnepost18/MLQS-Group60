##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import os
import sys
from datetime import datetime
import pandas as pd
import glob

# Set the root path to the bouldering data directory
ROOT_DATA_PATH = Path('./datasets/bouldering/')
RESULT_PATH = Path('./intermediate_datafiles_bouldering/')

GRANULARITIES = [250, 1000]

# Create the necessary directories if they don't exist
[path.mkdir(exist_ok=True, parents=True) for path in [ROOT_DATA_PATH, RESULT_PATH]]

print('Please wait, this will take a while to run!')

all_fine_grained_datasets_overall = []
all_fine_grained_dataset_names_overall = []
# NEW: Dictionary to store the start time for each dataset, mapping name to timestamp.
folder_timestamps = {}

for dataset_folder_name_raw in os.listdir(ROOT_DATA_PATH):
    DATASET_PATH = ROOT_DATA_PATH / dataset_folder_name_raw

    if not DATASET_PATH.is_dir():
        continue

    path_parts = str(DATASET_PATH).split('/')
    dataset_name_full = path_parts[-1]
    dataset_name = dataset_name_full.split(' ')[0]
    print(f"\nProcessing session '{dataset_name_full}'...")

    try:
        time_part_str = ' '.join(dataset_name_full.split(' ')[1:])
        time_components = time_part_str.split(' ')
        date_str = time_components[0]
        time_str = time_components[1].replace('-', ':')
        BOULDERING_START_TIME_FOR_RELATIVE_DATA = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S')
        print(f"Using folder timestamp as anchor: {BOULDERING_START_TIME_FOR_RELATIVE_DATA}")
        # NEW: Store the parsed timestamp in our dictionary with the short dataset name as the key.
        folder_timestamps[dataset_name] = BOULDERING_START_TIME_FOR_RELATIVE_DATA
    except (ValueError, IndexError) as e:
        print(
            f"Warning: Could not parse timestamp from folder name '{dataset_folder_name_raw}'. Error: {e}. Skipping folder.")
        continue

    datasets_for_current_folder_granularities = []
    DataViz = VisualizeDataset(__file__)

    for ms_per_instance in GRANULARITIES:
        print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {ms_per_instance}.')

        dataset = CreateDataset(DATASET_PATH, ms_per_instance)

        dataset.add_numerical_dataset_with_unit('Accelerometer.csv', "Time (s)",
                                                ["X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"], 'avg', 'acc_',
                                                is_relative_time=True,
                                                recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)
        dataset.add_numerical_dataset_with_unit('Gyroscope.csv', "Time (s)",
                                                ["X (rad/s)", "Y (rad/s)", "Z (rad/s)"], 'avg', 'gyr_',
                                                is_relative_time=True,
                                                recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)
        dataset.add_numerical_dataset_with_unit('Magnetometer.csv', "Time (s)", ["X (µT)", "Y (µT)", "Z (µT)"],
                                                'avg', 'mag_', is_relative_time=True,
                                                recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)
        dataset.add_numerical_dataset_with_unit('Location.csv', "Time (s)", ["Height (m)", "Velocity (m/s)"], 'avg',
                                                'loc_', is_relative_time=True,
                                                recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)

        dataset = dataset.data_table

        possible_labels = ['Easy', 'Medium', 'Hard']
        for label in possible_labels:
            dataset[f'label{label}'] = 0

        if 'Easy' in dataset_name:
            dataset['labelEasy'] = 1
        elif 'Medium' in dataset_name:
            dataset['labelMedium'] = 1
        elif 'Hard' in dataset_name:
            dataset['labelHard'] = 1

        DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'mag_', 'loc_', 'label'],
                             ['like', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'points'],
                             dataset_name=dataset_name)

        numerical_cols = [col for col in dataset.columns if 'label' not in col]
        datasets_for_current_folder_granularities.append(copy.deepcopy(dataset))

        if ms_per_instance == 250:
            all_fine_grained_datasets_overall.append(copy.deepcopy(dataset))
            all_fine_grained_dataset_names_overall.append(f"{dataset_name}")
            RESULT_FNAME_CURRENT = f'chapter2_result_{dataset_name.replace(" ", "_")}.csv'
            dataset.to_csv(RESULT_PATH / RESULT_FNAME_CURRENT)

    if len(datasets_for_current_folder_granularities) == 2:
        util.print_latex_table_statistics_two_datasets(datasets_for_current_folder_granularities[0],
                                                       datasets_for_current_folder_granularities[1])

if len(all_fine_grained_datasets_overall) > 0:
    DataViz = VisualizeDataset(__file__)
    features_to_compare = ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)', 'gyr_X (rad/s)', 'gyr_Y (rad/s)',
                           'gyr_Z (rad/s)', 'mag_X (µT)', 'mag_Y (µT)', 'mag_Z (µT)', 'loc_Height (m)',
                           'loc_Velocity (m/s)']
    DataViz.plot_feature_distributions_across_datasets(all_fine_grained_datasets_overall, features_to_compare,
                                                       all_fine_grained_dataset_names_overall,
                                                       main_title=f"Feature Distributions Across All Bouldering Datasets")
    DataViz.plot_feature_over_time_multi_dataset(all_fine_grained_datasets_overall, features_to_compare,
                                                 all_fine_grained_dataset_names_overall,
                                                 main_title='All Feature Readings Across All Bouldering Sessions (Relative Time)',
                                                 use_relative_time=True)

print('\nThe code has run through successfully!')

# ---------------------------------------------------------------------------
# Combine all generated CSV files into a single master file,
# adjusting timestamps to make them sequential.
# ---------------------------------------------------------------------------
search_pattern = RESULT_PATH / 'chapter2_result_*.csv'
all_csv_files = glob.glob(str(search_pattern))

final_filename_str = 'chapter2_result_combined.csv'
csv_files_to_combine = [f for f in all_csv_files if 'combined' not in f]

# MODIFIED: Replaced the simple alphabetical sort with a chronological sort.
# We use the 'folder_timestamps' dictionary we created earlier as a lookup.
def get_timestamp_for_file(filepath):
    # Extracts the dataset name (e.g., 'Easy1') from a full filepath.
    base_name = Path(filepath).stem
    dataset_name_key = base_name.replace('chapter2_result_', '')
    # Returns the timestamp for that dataset name. Defaults to a very old time if not found.
    return folder_timestamps.get(dataset_name_key, datetime.min)

csv_files_to_combine.sort(key=get_timestamp_for_file)

# A printout to confirm the new chronological order.
print("\nFiles will be combined in the following chronological order:")
for f in csv_files_to_combine:
    print(f"  - {Path(f).name}")
# END OF MODIFIED SECTION

adjusted_dfs = []
last_timestamp = None
sampling_interval = None

for file in csv_files_to_combine:
    print(f"Processing and adjusting timestamps for {file}...")
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)

    if sampling_interval is None and len(df.index) > 1:
        sampling_interval = df.index[1] - df.index[0]

    if last_timestamp is not None:
        current_start_time = df.index.min()
        new_start_time = last_timestamp + sampling_interval
        time_shift = new_start_time - current_start_time
        df.index = df.index + time_shift

    last_timestamp = df.index.max()
    adjusted_dfs.append(df)

if adjusted_dfs:
    combined_df = pd.concat(adjusted_dfs)
    combined_df.sort_index(inplace=True)
    output_path = RESULT_PATH / final_filename_str
    combined_df.to_csv(output_path)

    print('\n--- Statistics for the Combined Sequential Dataset ---')
    util.print_statistics(combined_df)

    # ---------------------------------------------------------------------------
    # Visualize the newly created combined dataset.
    # ---------------------------------------------------------------------------
    DataViz_combined = VisualizeDataset(__file__)
    dataset_name_combined = 'Combined'

    DataViz_combined.plot_dataset_boxplot(combined_df, ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'],
                                                                                dataset_name=dataset_name_combined)
    DataViz_combined.plot_dataset_boxplot(combined_df, ["gyr_X (rad/s)", "gyr_Y (rad/s)", "gyr_Z (rad/s)"],
                                          dataset_name=dataset_name_combined)
    DataViz_combined.plot_dataset_boxplot(combined_df, ["mag_X (µT)", "mag_Y (µT)", "mag_Z (µT)"],
                                          dataset_name=dataset_name_combined)
    DataViz_combined.plot_dataset_boxplot(combined_df, ["loc_Height (m)", "loc_Velocity (m/s)"],
                                          dataset_name=dataset_name_combined)

    DataViz_combined.plot_dataset(combined_df, ['acc_', 'gyr_', 'mag_', 'loc_', 'label'],
                                  ['like', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'points'],
                                  dataset_name=dataset_name_combined)

    numerical_cols_combined = [col for col in combined_df.columns if 'label' not in col]
    DataViz_combined.plot_correlation_heatmap(combined_df, columns=numerical_cols_combined,
                                              title=f"Correlation Heatmap for {dataset_name_combined} Dataset")
else:
    print("\nNo files were found to combine.")