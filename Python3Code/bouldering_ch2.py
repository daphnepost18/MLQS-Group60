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

# Changed to the root directory containing all participant folders
ROOT_DATA_PATH = Path('./datasets/bouldering/')
RESULT_PATH = Path('./intermediate_datafiles_bouldering/')

GRANULARITIES = [60000, 250]

[path.mkdir(exist_ok=True, parents=True) for path in [ROOT_DATA_PATH, RESULT_PATH]]

print('Please wait, this will take a while to run!')

# Lists to collect fine-grained datasets and their names from ALL participants and ALL folders
all_fine_grained_datasets_overall = []
all_fine_grained_dataset_names_overall = []

# New Outer Loop: Iterate through each participant folder
for participant_folder_name in os.listdir(ROOT_DATA_PATH):
    BASE_DATA_PATH = ROOT_DATA_PATH / participant_folder_name

    if not BASE_DATA_PATH.is_dir():
        continue  # Skip if it's not a directory (e.g., .DS_Store file)

    participant_name = participant_folder_name  # e.g., 'participant1', 'participant2'

    print(f"\n--- Processing Participant: {participant_name} ---")

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

        for milliseconds_per_instance in GRANULARITIES:
            print(
                f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

            dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)

            dataset.add_numerical_dataset_with_unit('Accelerometer.csv', "Time (s)",
                                                    ["X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"],
                                                    'avg', 'acc_', is_relative_time=True,
                                                    recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)
            dataset.add_numerical_dataset_with_unit('Gyroscope.csv', "Time (s)",
                                                    ["X (rad/s)", "Y (rad/s)", "Z (rad/s)"], 'avg',
                                                    'gyr_', is_relative_time=True,
                                                    recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)
            dataset.add_event_dataset_with_unit('Labels.csv', 'label_start', 'label_end', 'label', 'binary',
                                                is_relative_time=True,
                                                recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)
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

            util.print_statistics(dataset)
            datasets_for_current_folder_granularities.append(copy.deepcopy(dataset))

            RESULT_FNAME_CURRENT = f'chapter2_result_{participant_name.replace(" ", "_")}_{dataset_name.replace(" ", "_")}_{milliseconds_per_instance}.csv'
            dataset.to_csv(RESULT_PATH / RESULT_FNAME_CURRENT)

            # Collect the 250ms granularity dataset for cross-dataset comparison FOR THIS PARTICIPANT
            if milliseconds_per_instance == 250:
                all_fine_grained_datasets_for_participant.append(copy.deepcopy(dataset))
                all_fine_grained_dataset_names_for_participant.append(dataset_name)

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

print('The code has run through successfully!')