##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

# Import the relevant classes.
from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import os
import sys

# Chapter 2: Initial exploration of the dataset.

DATASET_PATH = Path('./datasets/bouldering/participant1/Easy1 2025-06-03 19-59-30')
RESULT_PATH = Path('./intermediate_datafiles_bouldering/')
RESULT_FNAME = 'chapter2_result_bouldering.csv'

# Set a granularity (the discrete step size of our time series data). We'll use a course-grained granularity of one
# instance per minute, and a fine-grained one with four instances per second.
GRANULARITIES = [60000, 250]

# We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
[path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]

print('Please wait, this will take a while to run!')

datasets = []

# Dynamically extract participant and dataset names from DATASET_PATH
path_parts = str(DATASET_PATH).split('/')
participant_name = path_parts[-2] # e.g., 'participant1'
dataset_name = path_parts[-1].split(' ')[0] # Takes 'Easy1' from 'Easy1 2025-06-03 19-59-30'

for milliseconds_per_instance in GRANULARITIES:
    print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

    # Create an initial dataset object with the base directory for our data and a granularity
    dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)

    # Add the selected measurements to it.
    dataset.add_numerical_dataset_with_unit('Accelerometer.csv', "Time (s)", ["X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"],
                                            'avg', 'acc_', timestamp_unit='s')
    dataset.add_numerical_dataset_with_unit('Gyroscope.csv', "Time (s)", ["X (rad/s)", "Y (rad/s)", "Z (rad/s)"], 'avg',
                                            'gyr_', timestamp_unit='s')
    dataset.add_event_dataset_with_unit('Labels.csv', 'label_start', 'label_end', 'label', 'binary', timestamp_unit='s')
    dataset.add_numerical_dataset_with_unit('Magnetometer.csv', "Time (s)", ["X (µT)", "Y (µT)", "Z (µT)"], 'avg',
                                            'mag_', timestamp_unit='s')
    dataset.add_numerical_dataset_with_unit('Location.csv', "Time (s)", ["Height (m)","Velocity (m/s)"], 'avg', 'loc_',
                                            timestamp_unit='s')

    # Get the resulting pandas data table
    dataset = dataset.data_table

    # Plot the data
    DataViz = VisualizeDataset(__file__)

    # Boxplot
    DataViz.plot_dataset_boxplot(dataset, ['acc_X (m/s^2)','acc_Y (m/s^2)','acc_Z (m/s^2)'],participant_name=participant_name, dataset_name=dataset_name)
    DataViz.plot_dataset_boxplot(dataset, ["gyr_X (rad/s)","gyr_Y (rad/s)","gyr_Z (rad/s)"],participant_name=participant_name, dataset_name=dataset_name)
    DataViz.plot_dataset_boxplot(dataset, ["mag_X (µT)", "mag_Y (µT)", "mag_Z (µT)"],participant_name=participant_name, dataset_name=dataset_name)
    DataViz.plot_dataset_boxplot(dataset, ["loc_Height (m)","loc_Velocity (m/s)"],participant_name=participant_name, dataset_name=dataset_name)

    # Plot all data
    DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'mag_', 'loc_'],
                                  ['like', 'like', 'like', 'like'],
                                  ['line', 'line', 'line', 'line'], participant_name=participant_name, dataset_name=dataset_name)

    # And print a summary of the dataset.
    util.print_statistics(dataset)
    datasets.append(copy.deepcopy(dataset))

    # If needed, we could save the various versions of the dataset we create in the loop with logical filenames:
    # dataset.to_csv(RESULT_PATH / f'chapter2_result_{milliseconds_per_instance}')


# Make a table like the one shown in the book, comparing the two datasets produced.
util.print_latex_table_statistics_two_datasets(datasets[0], datasets[1])

# Finally, store the last dataset we generated (250 ms).
dataset.to_csv(RESULT_PATH / RESULT_FNAME)

# Lastly, print a statement to know the code went through

print('The code has run through successfully!')