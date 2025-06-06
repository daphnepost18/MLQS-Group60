##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

import pandas as pd
import numpy as np
import re
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plot
import matplotlib.dates as md

class CreateDataset:

    base_dir = ''
    granularity = 0
    data_table = None

    def __init__(self, base_dir, granularity):
        self.base_dir = base_dir
        self.granularity = granularity

    # Create an initial data table with entries from start till end time, with steps
    # of size granularity. Granularity is specified in milliseconds
    def create_timestamps(self, start_time, end_time):
        return pd.date_range(start_time, end_time, freq=str(self.granularity)+'ms')

    def create_dataset(self, start_time, end_time, cols, prefix):
        c = copy.deepcopy(cols)
        if not prefix == '':
            for i in range(0, len(c)):
                c[i] = str(prefix) + str(c[i])
        timestamps = self.create_timestamps(start_time, end_time)

        #Specify the datatype here to prevent an issue
        self.data_table = pd.DataFrame(index=timestamps, columns=c, dtype=object)

    # Add numerical data, we assume timestamps in the form of nanoseconds from the epoch
    def add_numerical_dataset(self, file, timestamp_col, value_cols, aggregation='avg', prefix=''):
        print(f'Reading data from {file}')
        dataset = pd.read_csv(self.base_dir / file, skipinitialspace=True)

        # Convert timestamps to dates
        dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col])

        # Create a table based on the times found in the dataset
        if self.data_table is None:
            self.create_dataset(min(dataset[timestamp_col]), max(dataset[timestamp_col]), value_cols, prefix)
        else:
            for col in value_cols:
                self.data_table[str(prefix) + str(col)] = np.nan

        # Over all rows in the new table
        for i in range(0, len(self.data_table.index)):
            # Select the relevant measurements.
            relevant_rows = dataset[
                (dataset[timestamp_col] >= self.data_table.index[i]) &
                (dataset[timestamp_col] < (self.data_table.index[i] +
                                           timedelta(milliseconds=self.granularity)))
            ]
            for col in value_cols:
                # Take the average value
                if len(relevant_rows) > 0:
                    if aggregation == 'avg':
                        self.data_table.loc[self.data_table.index[i], str(prefix)+str(col)] = np.average(relevant_rows[col])
                    else:
                        raise ValueError(f"Unknown aggregation {aggregation}")
                else:
                    self.data_table.loc[self.data_table.index[i], str(prefix)+str(col)] = np.nan

    # Remove undesired value from the names.
    def clean_name(self, name):
        return re.sub('[^0-9a-zA-Z]+', '', name)

    # Add data in which we have rows that indicate the occurrence of a certain event with a given start and end time.
    # 'aggregation' can be 'sum' or 'binary'.
    def add_event_dataset(self, file, start_timestamp_col, end_timestamp_col, value_col, aggregation='sum'):
        print(f'Reading data from {file}')
        dataset = pd.read_csv(self.base_dir / file)

        # Convert timestamps to datetime.
        dataset[start_timestamp_col] = pd.to_datetime(dataset[start_timestamp_col])
        dataset[end_timestamp_col] = pd.to_datetime(dataset[end_timestamp_col])

        # Clean the event values in the dataset
        dataset[value_col] = dataset[value_col].apply(self.clean_name)
        event_values = dataset[value_col].unique()

        # Add columns for all possible values (or create a new dataset if empty), set the default to 0 occurrences
        if self.data_table is None:
            self.create_dataset(min(dataset[start_timestamp_col]), max(dataset[end_timestamp_col]), event_values, value_col)
        for col in event_values:
            self.data_table[(str(value_col) + str(col))] = 0

        # Now we need to start counting by passing along the rows....
        for i in range(0, len(dataset.index)):
            # identify the time points of the row in our dataset and the value
            start = dataset[start_timestamp_col][i]
            end = dataset[end_timestamp_col][i]
            value = dataset[value_col][i]
            border = (start - timedelta(milliseconds=self.granularity))

            # get the right rows from our data table
            relevant_rows = self.data_table[(start <= (self.data_table.index +timedelta(milliseconds=self.granularity))) & (end > self.data_table.index)]

            # and add 1 to the rows if we take the sum
            if aggregation == 'sum':
                self.data_table.loc[relevant_rows.index, str(value_col) + str(value)] += 1
            # or set to 1 if we just want to know it happened
            elif aggregation == 'binary':
                self.data_table.loc[relevant_rows.index, str(value_col) + str(value)] = 1
            else:
                raise ValueError("Unknown aggregation '" + aggregation + "'")

    # This function returns the column names that have one of the strings expressed by 'ids' in the column name.
    def get_relevant_columns(self, ids):
        relevant_dataset_cols = []
        cols = list(self.data_table.columns)

        for id in ids:
            relevant_dataset_cols.extend([col for col in cols if id in col])

        return relevant_dataset_cols

    # Add numerical data, with an explicit timestamp unit
    def add_numerical_dataset_with_unit(self, file, timestamp_col, value_cols, aggregation='avg', prefix='',
                                        timestamp_unit=None, is_relative_time=False, recording_start_time=None):
        print(f'Reading data from {file}')
        dataset = pd.read_csv(self.base_dir / file, skipinitialspace=True)

        if is_relative_time:
            # Convert timestamp column to float to perform arithmetic operations
            dataset[timestamp_col] = dataset[timestamp_col].astype(float)

            # Shift timestamps so the earliest one in this file becomes 0
            min_ts_in_file = dataset[timestamp_col].min()
            dataset[timestamp_col] = dataset[timestamp_col] - min_ts_in_file

            # If recording_start_time is not provided, use an arbitrary default for the origin
            if recording_start_time is None:
                recording_start_time = datetime(2025, 1, 1, 0, 0, 0)  # Arbitrary start date

            # Convert the 0-based relative seconds to absolute datetimes using the recording_start_time as origin
            dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col], unit='s', origin=recording_start_time)
        else:
            # This is the original logic for absolute timestamps (e.g., crowdsignals)
            if timestamp_unit:
                dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col], unit=timestamp_unit)
            else:
                dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col])

        # --- Existing logic for creating/expanding data_table and aggregation ---

        # Create the initial data_table if it doesn't exist, or expand its time range if needed
        if self.data_table is None:
            self.create_dataset(min(dataset[timestamp_col]), max(dataset[timestamp_col]), value_cols, prefix)
        else:
            # Determine the new overall time range by combining current data_table and new dataset ranges
            current_min_time = self.data_table.index.min()
            current_max_time = self.data_table.index.max()
            new_min_time = min(current_min_time, dataset[timestamp_col].min())
            new_max_time = max(current_max_time, dataset[timestamp_col].max())

            # If the data_table's time range needs to be expanded, reindex it
            if new_min_time < current_min_time or new_max_time > current_max_time:
                extended_timestamps = self.create_timestamps(new_min_time, new_max_time)
                # Reindex and fill new time intervals with NaN, preserve existing columns
                existing_data = self.data_table.copy()
                self.data_table = pd.DataFrame(index=extended_timestamps, columns=existing_data.columns, dtype=object)
                self.data_table.update(existing_data)  # Populate with existing data

            # Add new columns if they don't exist in the expanded data_table
            for col in value_cols:
                full_col_name = str(prefix) + str(col)
                if full_col_name not in self.data_table.columns:
                    self.data_table[full_col_name] = np.nan  # Initialize new column with NaN

        # Over all rows in the new table (self.data_table) - this is the aggregation loop
        for current_interval_start in self.data_table.index:
            current_interval_end = current_interval_start + timedelta(milliseconds=self.granularity)

            relevant_rows = dataset[
                (dataset[timestamp_col] >= current_interval_start) &
                (dataset[timestamp_col] < current_interval_end)
                ]
            for col in value_cols:
                full_col_name = str(prefix) + str(col)
                # Take the average value
                if len(relevant_rows) > 0:
                    if aggregation == 'avg':
                        self.data_table.loc[current_interval_start, full_col_name] = np.average(relevant_rows[col])
                    else:
                        raise ValueError(f"Unknown aggregation {aggregation}")
                # If relevant_rows is empty, the value remains NaN as initialized, which is desired.

    # Add event data, with an explicit timestamp unit
    def add_event_dataset_with_unit(self, file, start_timestamp_col, end_timestamp_col, value_col, aggregation='sum', timestamp_unit=None, is_relative_time=False, recording_start_time=None):
        print(f'Reading data from {file}')
        dataset = pd.read_csv(self.base_dir / file)

        if is_relative_time:
            # Convert timestamp columns to float to perform arithmetic operations
            dataset[start_timestamp_col] = dataset[start_timestamp_col].astype(float)
            dataset[end_timestamp_col] = dataset[end_timestamp_col].astype(float)

            # Determine the overall minimum timestamp across both start and end times in this file
            min_ts_start = dataset[start_timestamp_col].min()
            min_ts_end = dataset[end_timestamp_col].min()
            min_ts_in_file = min(min_ts_start, min_ts_end)

            # Shift timestamps so the earliest one in this file becomes 0
            dataset[start_timestamp_col] = dataset[start_timestamp_col] - min_ts_in_file
            dataset[end_timestamp_col] = dataset[end_timestamp_col] - min_ts_in_file

            # If recording_start_time is not provided, use an arbitrary default for the origin
            if recording_start_time is None:
                recording_start_time = datetime(2025, 1, 1, 0, 0, 0) # Arbitrary start date

            # Convert the 0-based relative seconds to absolute datetimes using the recording_start_time as origin
            dataset[start_timestamp_col] = pd.to_datetime(dataset[start_timestamp_col], unit='s', origin=recording_start_time)
            dataset[end_timestamp_col] = pd.to_datetime(dataset[end_timestamp_col], unit='s', origin=recording_start_time)
        else:
            # This is the original logic for absolute timestamps (e.g., crowdsignals)
            if timestamp_unit:
                dataset[start_timestamp_col] = pd.to_datetime(dataset[start_timestamp_col], unit=timestamp_unit)
                dataset[end_timestamp_col] = pd.to_datetime(dataset[end_timestamp_col], unit=timestamp_unit)
            else:
                dataset[start_timestamp_col] = pd.to_datetime(dataset[start_timestamp_col])
                dataset[end_timestamp_col] = pd.to_datetime(dataset[end_timestamp_col])


        # Clean the event values in the dataset
        dataset[value_col] = dataset[value_col].apply(self.clean_name)
        event_values = dataset[value_col].unique()

        # Add columns for all possible values (or create a new dataset if empty), set the default to 0 occurrences
        if self.data_table is None:
            self.create_dataset(min(dataset[start_timestamp_col]), max(dataset[end_timestamp_col]), event_values, value_col)
        else:
            # Determine the new overall time range by combining current data_table and new dataset ranges
            current_min_time = self.data_table.index.min()
            current_max_time = self.data_table.index.max()
            new_min_time = min(current_min_time, dataset[start_timestamp_col].min())
            new_max_time = max(current_max_time, dataset[end_timestamp_col].max())

            # If the data_table's time range needs to be expanded, reindex it
            if new_min_time < current_min_time or new_max_time > current_max_time:
                extended_timestamps = self.create_timestamps(new_min_time, new_max_time)
                existing_data = self.data_table.copy()
                self.data_table = pd.DataFrame(index=extended_timestamps, columns=existing_data.columns, dtype=object)
                self.data_table.update(existing_data)

            # Add new columns if they don't exist in the expanded data_table and reset to 0 for current aggregation
            for col_val in event_values:
                col_name = str(value_col) + str(col_val)
                if col_name not in self.data_table.columns:
                    self.data_table[col_name] = 0 # Add new column if it doesn't exist
                else:
                    self.data_table[col_name] = 0 # Reset existing column to 0 for counting in this run


        # Now we need to start counting by passing along the rows....
        for index, row_data in dataset.iterrows(): # Using iterrows() for robustness
            # identify the time points of the row in our dataset and the value
            start = row_data[start_timestamp_col]
            end = row_data[end_timestamp_col]
            value = row_data[value_col]

            # get the right rows from our data table
            relevant_rows = self.data_table[(start <= (self.data_table.index +timedelta(milliseconds=self.granularity))) & (end > self.data_table.index)]

            # and add 1 to the rows if we take the sum
            if aggregation == 'sum':
                self.data_table.loc[relevant_rows.index, str(value_col) + str(value)] += 1
            # or set to 1 if we just want to know it happened
            elif aggregation == 'binary':
                self.data_table.loc[relevant_rows.index, str(value_col) + str(value)] = 1
            else:
                raise ValueError("Unknown aggregation '" + aggregation + "'")

