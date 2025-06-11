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

ROOT_DATA_PATH = Path('./datasets/bouldering/')
RESULT_PATH = Path('./intermediate_datafiles_bouldering/')

# Set a granularity (the discrete step size of our time series data). We'll use a course-grained granularity of one
# instance per minute, and a fine-grained one with four instances per second.
GRANULARITIES = [250, 1000]

[path.mkdir(exist_ok=True, parents=True) for path in [ROOT_DATA_PATH, RESULT_PATH]]

print('Please wait, this will take a while to run!')

# Lists to collect fine-grained datasets and their names from ALL participants and ALL folders
all_fine_grained_datasets_overall = []
all_fine_grained_dataset_names_overall = []

for participant_folder_name in os.listdir(ROOT_DATA_PATH):
    BASE_DATA_PATH = ROOT_DATA_PATH / participant_folder_name

    if not BASE_DATA_PATH.is_dir():
        continue  # Skip if it's not a directory (e.g., .DS_Store file)

    participant_name = participant_folder_name

    print(f"\n--- Processing Participant: {participant_name} ---")

    participant_labels_path = BASE_DATA_PATH / 'Labels.csv'
    if not participant_labels_path.is_file():
        print(f"Warning: Labels.csv not found in {participant_labels_path}. Labels will not be added for this participant's sessions.")

    # Lists for datasets within the current participant
    datasets_for_current_participant = []
    all_fine_grained_datasets_for_participant = []
    all_fine_grained_dataset_names_for_participant = []

    # Iterate through each dataset folder within the current participant's directory
    for dataset_folder_name_raw in os.listdir(BASE_DATA_PATH):
        DATASET_PATH = BASE_DATA_PATH / dataset_folder_name_raw

        if not DATASET_PATH.is_dir():
            continue  # Skip if it's not a directory (e.g., .DS_Store file)

        path_parts = str(DATASET_PATH).split('/')
        dataset_name_full = path_parts[-1]
        dataset_name = dataset_name_full.split(' ')[0]

        # Dynamically derive the recording_start_time from the folder name
        time_part_str = ' '.join(dataset_name_full.split(' ')[1:])
        time_part_str = time_part_str.split(' ')[0] + ' ' + time_part_str.split(' ')[1].replace('-', ':')
        BOULDERING_START_TIME_FOR_RELATIVE_DATA = datetime.strptime(time_part_str, '%Y-%m-%d %H:%M:%S')

        datasets_for_current_folder_granularities = []

        for ms_per_instance in GRANULARITIES:
            print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {ms_per_instance}.')

            dataset = CreateDataset(DATASET_PATH, ms_per_instance)

            dataset.add_numerical_dataset_with_unit('Accelerometer.csv', "Time (s)",
                                                    ["X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"],
                                                    'avg', 'acc_', is_relative_time=True,
                                                    recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)
            dataset.add_numerical_dataset_with_unit('Gyroscope.csv', "Time (s)",
                                                    ["X (rad/s)", "Y (rad/s)", "Z (rad/s)"], 'avg',
                                                    'gyr_', is_relative_time=True,
                                                    recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)
            if participant_labels_path.is_file():
                dataset.add_event_dataset_with_unit(participant_labels_path, 'label_start', 'label_end', 'label', 'binary',
                                                    is_relative_time=False,
                                                    recording_start_time=None)
            else:
                print(f"Skipping label addition for session {dataset_name} as Labels.csv was not found at {participant_labels_path}.")

            dataset.add_numerical_dataset_with_unit('Magnetometer.csv', "Time (s)", ["X (µT)", "Y (µT)", "Z (µT)"],
                                                    'avg',
                                                    'mag_', is_relative_time=True,
                                                    recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)
            dataset.add_numerical_dataset_with_unit('Location.csv', "Time (s)", ["Height (m)", "Velocity (m/s)"], 'avg',
                                                    'loc_',
                                                    is_relative_time=True,
                                                    recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)

            dataset = dataset.data_table

            DataViz = VisualizeDataset(__file__)

            DataViz.plot_dataset_boxplot(dataset, ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)'],
                                         participant_name=participant_name, dataset_name=dataset_name)
            DataViz.plot_dataset_boxplot(dataset, ["gyr_X (rad/s)", "gyr_Y (rad/s)", "gyr_Z (rad/s)"],
                                         participant_name=participant_name, dataset_name=dataset_name)
            DataViz.plot_dataset_boxplot(dataset, ["mag_X (µT)", "mag_Y (µT)", "mag_Z (µT)"],
                                         participant_name=participant_name, dataset_name=dataset_name)
            DataViz.plot_dataset_boxplot(dataset, ["loc_Height (m)", "loc_Velocity (m/s)"],
                                         participant_name=participant_name, dataset_name=dataset_name)

            DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'mag_', 'loc_'],
                                 ['like', 'like', 'like', 'like'],
                                 ['line', 'line', 'line', 'line'],
                                 participant_name=participant_name, dataset_name=dataset_name)

            numerical_cols = ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)',
                              'gyr_X (rad/s)', 'gyr_Y (rad/s)', 'gyr_Z (rad/s)',
                              'mag_X (µT)', 'mag_Y (µT)', 'mag_Z (µT)',
                              'loc_Height (m)', 'loc_Velocity (m/s)']
            DataViz.plot_correlation_heatmap(dataset, columns=numerical_cols,
                                             title=f"Correlation Heatmap for {participant_name} - {dataset_name}")

            util.print_statistics(dataset)
            datasets_for_current_folder_granularities.append(copy.deepcopy(dataset))

            # Collect the 250ms granularity dataset for cross-dataset comparison FOR THIS PARTICIPANT
            if ms_per_instance == 250:
                all_fine_grained_datasets_for_participant.append(copy.deepcopy(dataset))
                all_fine_grained_dataset_names_for_participant.append(dataset_name)
                RESULT_FNAME_CURRENT = f'chapter2_result_{participant_name.replace(" ", "_")}_{dataset_name.replace(" ", "_")}_{ms_per_instance}.csv'
                dataset.to_csv(RESULT_PATH / RESULT_FNAME_CURRENT)

        util.print_latex_table_statistics_two_datasets(datasets_for_current_folder_granularities[0],
                                                       datasets_for_current_folder_granularities[1])

    # After processing all datasets for a participant, add them to the overall list
    all_fine_grained_datasets_overall.extend(all_fine_grained_datasets_for_participant)
    all_fine_grained_dataset_names_overall.extend(all_fine_grained_dataset_names_for_participant)

# After ALL participants and ALL their datasets are processed, make the final overall comparison plot
if len(all_fine_grained_datasets_overall) > 0:
    DataViz = VisualizeDataset(__file__)

    features_to_compare =   ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)',
                            'gyr_X (rad/s)', 'gyr_Y (rad/s)', 'gyr_Z (rad/s)',
                            'mag_X (µT)', 'mag_Y (µT)', 'mag_Z (µT)',
                            'loc_Height (m)', 'loc_Velocity (m/s)']

    # The main title will reflect all participants if names are sufficiently varied
    DataViz.plot_feature_distributions_across_datasets(all_fine_grained_datasets_overall,
        features_to_compare,all_fine_grained_dataset_names_overall,
        main_title=f"Feature Distributions Across All Bouldering Datasets (All Participants)")

    DataViz.plot_feature_over_time_multi_dataset(
        all_fine_grained_datasets_overall,
        features_to_compare,
        all_fine_grained_dataset_names_overall,
        main_title='All Feature Readings Across All Bouldering Sessions (Relative Time)',
        use_relative_time=True
    )

print('The code has run through successfully!')