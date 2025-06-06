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
from datetime import datetime # Still needed for datetime.strptime if you derive start time from folder name

BASE_DATA_PATH = Path('./datasets/bouldering/participant1/')
RESULT_PATH = Path('./intermediate_datafiles_bouldering/')

GRANULARITIES = [60000, 250]

[path.mkdir(exist_ok=True, parents=True) for path in [BASE_DATA_PATH, RESULT_PATH]]

print('Please wait, this will take a while to run!')

# Loop through each dataset folder
for dataset_folder_name_raw in os.listdir(BASE_DATA_PATH):
    DATASET_PATH = BASE_DATA_PATH / dataset_folder_name_raw

    if not DATASET_PATH.is_dir():
        continue

    path_parts = str(DATASET_PATH).split('/')
    print(path_parts)
    participant_name = path_parts[-2]
    print(participant_name)
    dataset_name_full = path_parts[-1]
    print(dataset_name_full)
    dataset_name = dataset_name_full.split(' ')[0]
    print(dataset_name)

    # This part dynamically derives the recording_start_time from the folder name.
    # This is *critical* for bouldering data's relative timestamps to plot correctly.
    # Assuming folder name format: 'DatasetName YYYY-MM-DD HH-MM-SS'
    time_part_str = ' '.join(dataset_name_full.split(' ')[1:])
    # Adjust for potential '-' in the time portion (e.g., 19-59-30)
    time_part_str = time_part_str.split(' ')[0] + ' ' + time_part_str.split(' ')[1].replace('-', ':')
    BOULDERING_START_TIME_FOR_RELATIVE_DATA = datetime.strptime(time_part_str, '%Y-%m-%d %H:%M:%S')


    datasets_for_current_folder = []

    for milliseconds_per_instance in GRANULARITIES:
        print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

        dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)

        # Call add_..._with_unit with is_relative_time=True and the derived start time
        # This is where the relative timestamps are anchored to a real date/time.
        dataset.add_numerical_dataset_with_unit('Accelerometer.csv', "Time (s)", ["X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"],
                                                'avg', 'acc_', is_relative_time=True, recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)
        dataset.add_numerical_dataset_with_unit('Gyroscope.csv', "Time (s)", ["X (rad/s)", "Y (rad/s)", "Z (rad/s)"], 'avg',
                                                'gyr_', is_relative_time=True, recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)
        dataset.add_event_dataset_with_unit('Labels.csv', 'label_start', 'label_end', 'label', 'binary', is_relative_time=True, recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)
        dataset.add_numerical_dataset_with_unit('Magnetometer.csv', "Time (s)", ["X (µT)", "Y (µT)", "Z (µT)"], 'avg',
                                                'mag_', is_relative_time=True, recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)
        dataset.add_numerical_dataset_with_unit('Location.csv', "Time (s)", ["Height (m)","Velocity (m/s)"], 'avg', 'loc_',
                                                is_relative_time=True, recording_start_time=BOULDERING_START_TIME_FOR_RELATIVE_DATA)

        dataset = dataset.data_table

        DataViz = VisualizeDataset(__file__)

        DataViz.plot_dataset_boxplot(dataset, ['acc_X (m/s^2)','acc_Y (m/s^2)','acc_Z (m/s^2)'],
                                     participant_name=participant_name, dataset_name=dataset_name)
        DataViz.plot_dataset_boxplot(dataset, ["gyr_X (rad/s)","gyr_Y (rad/s)","gyr_Z (rad/s)"],
                                     participant_name=participant_name, dataset_name=dataset_name)
        DataViz.plot_dataset_boxplot(dataset, ["mag_X (µT)", "mag_Y (µT)", "mag_Z (µT)"],
                                     participant_name=participant_name, dataset_name=dataset_name)
        DataViz.plot_dataset_boxplot(dataset, ["loc_Height (m)","loc_Velocity (m/s)"],
                                     participant_name=participant_name, dataset_name=dataset_name)

        DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'mag_', 'loc_'],
                                      ['like', 'like', 'like', 'like'],
                                      ['line', 'line', 'line', 'line'],
                                      participant_name=participant_name, dataset_name=dataset_name)

        util.print_statistics(dataset)
        datasets_for_current_folder.append(copy.deepcopy(dataset))

        # Dynamically save the processed dataset for each granularity and folder
        # RESULT_FNAME_CURRENT = f'chapter2_result_{dataset_name.replace(" ", "_")}_{milliseconds_per_instance}.csv'
        # dataset.to_csv(RESULT_PATH / RESULT_FNAME_CURRENT)

    util.print_latex_table_statistics_two_datasets(datasets_for_current_folder[0], datasets_for_current_folder[1])

print('The code has run through successfully!')