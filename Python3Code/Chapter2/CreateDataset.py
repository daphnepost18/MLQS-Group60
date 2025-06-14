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
from pathlib import Path  # Import Path for robust path handling


class CreateDataset:
    base_dir = ''
    granularity = 0
    data_table = None

    def __init__(self, base_dir, granularity):
        self.base_dir = Path(base_dir)  # Ensure base_dir is a Path object
        self.granularity = granularity

    # Create an initial data table with entries from start till end time, with steps
    # of size granularity. Granularity is specified in milliseconds
    def create_timestamps(self, start_time, end_time):
        return pd.date_range(start_time, end_time, freq=str(self.granularity) + 'ms')

    def create_dataset(self, start_time, end_time, cols, prefix):
        c = copy.deepcopy(cols)
        if not prefix == '':
            for i in range(0, len(c)):
                c[i] = str(prefix) + str(c[i])
        timestamps = self.create_timestamps(start_time, end_time)

        # Specify the datatype here to prevent an issue
        self.data_table = pd.DataFrame(index=timestamps, columns=c, dtype=object)

    # Remove undesired value from the names.
    def clean_name(self, name):
        return re.sub('[^0-9a-zA-Z]+', '', name)

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

        if isinstance(file, pd.DataFrame):
            dataset = file.copy()
            print('Reading numerical data from supplied DataFrame')
        else:
            if isinstance(file, Path):
                file_path = file
            else:
                file_path = self.base_dir / file
            print(f'Reading data from {file_path}')
            dataset = pd.read_csv(file_path, skipinitialspace=True)

        if is_relative_time:
            dataset[timestamp_col] = dataset[timestamp_col].astype(float)
            if recording_start_time is not None:
                origin_time = recording_start_time
            else:
                min_ts_in_file = dataset[timestamp_col].min()
                dataset[timestamp_col] = dataset[timestamp_col] - min_ts_in_file
                origin_time = datetime(2025, 1, 1, 0, 0, 0)
            dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col], unit='s', origin=origin_time)
        else:
            if timestamp_unit:
                dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col], unit=timestamp_unit)
            else:
                dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col])

        if self.data_table is None:
            self.create_dataset(min(dataset[timestamp_col]), max(dataset[timestamp_col]), value_cols, prefix)
        else:
            current_min_time = self.data_table.index.min()
            current_max_time = self.data_table.index.max()
            new_min_time = min(current_min_time, dataset[timestamp_col].min())
            new_max_time = max(current_max_time, dataset[timestamp_col].max())

            if new_min_time < current_min_time or new_max_time > current_max_time:
                extended_timestamps = self.create_timestamps(new_min_time, new_max_time)
                existing_data = self.data_table.copy()
                self.data_table = pd.DataFrame(index=extended_timestamps, columns=existing_data.columns, dtype=object)
                self.data_table.update(existing_data)

            for col in value_cols:
                full_col_name = str(prefix) + str(col)
                if full_col_name not in self.data_table.columns:
                    self.data_table[full_col_name] = np.nan

        for current_interval_start in self.data_table.index:
            current_interval_end = current_interval_start + timedelta(milliseconds=self.granularity)
            relevant_rows = dataset[
                (dataset[timestamp_col] >= current_interval_start) &
                (dataset[timestamp_col] < current_interval_end)
                ]
            for col in value_cols:
                full_col_name = str(prefix) + str(col)
                if len(relevant_rows) > 0:
                    if aggregation == 'avg':
                        self.data_table.loc[current_interval_start, full_col_name] = np.average(relevant_rows[col])
                    else:
                        raise ValueError(f"Unknown aggregation {aggregation}")

    # Add event data, with an explicit timestamp unit
    def add_event_dataset_with_unit(self, file, start_timestamp_col, end_timestamp_col, value_col, aggregation='sum',
                                    timestamp_unit=None, is_relative_time=False, recording_start_time=None):

        if isinstance(file, pd.DataFrame):
            dataset = file.copy()
            print('Reading event data from supplied DataFrame')
        else:
            if isinstance(file, Path):
                file_path = file
            else:
                file_path = self.base_dir / file
            print(f'Reading data from {file_path}')
            dataset = pd.read_csv(file_path)

        if is_relative_time:
            dataset[start_timestamp_col] = dataset[start_timestamp_col].astype(float)
            dataset[end_timestamp_col] = dataset[end_timestamp_col].astype(float)
            if recording_start_time is not None:
                origin_time = recording_start_time
            else:
                min_ts_start = dataset[start_timestamp_col].min()
                min_ts_end = dataset[end_timestamp_col].min()
                min_ts_in_file = min(min_ts_start, min_ts_end)
                dataset[start_timestamp_col] = dataset[start_timestamp_col] - min_ts_in_file
                dataset[end_timestamp_col] = dataset[end_timestamp_col] - min_ts_in_file
                origin_time = datetime(2025, 1, 1, 0, 0, 0)
            dataset[start_timestamp_col] = pd.to_datetime(dataset[start_timestamp_col], unit='s', origin=origin_time)
            dataset[end_timestamp_col] = pd.to_datetime(dataset[end_timestamp_col], unit='s', origin=origin_time)
        else:
            if timestamp_unit:
                dataset[start_timestamp_col] = pd.to_datetime(dataset[start_timestamp_col], unit=timestamp_unit)
                dataset[end_timestamp_col] = pd.to_datetime(dataset[end_timestamp_col], unit=timestamp_unit)
            else:
                dataset[start_timestamp_col] = pd.to_datetime(dataset[start_timestamp_col])
                dataset[end_timestamp_col] = pd.to_datetime(dataset[end_timestamp_col])

        dataset[value_col] = dataset[value_col].apply(self.clean_name)
        event_values = dataset[value_col].unique()

        if self.data_table is None:
            self.create_dataset(min(dataset[start_timestamp_col]), max(dataset[end_timestamp_col]), event_values,
                                value_col)
        else:
            current_min_time = self.data_table.index.min()
            current_max_time = self.data_table.index.max()
            new_min_time = min(current_min_time, dataset[start_timestamp_col].min())
            new_max_time = max(current_max_time, dataset[end_timestamp_col].max())

            if new_min_time < current_min_time or new_max_time > current_max_time:
                extended_timestamps = self.create_timestamps(new_min_time, new_max_time)
                existing_data = self.data_table.copy()
                self.data_table = pd.DataFrame(index=extended_timestamps, columns=existing_data.columns, dtype=object)
                self.data_table.update(existing_data)

            for col_val in event_values:
                col_name = str(value_col) + str(col_val)
                if col_name not in self.data_table.columns:
                    self.data_table[col_name] = 0
                else:
                    self.data_table[col_name] = 0

        for index, row_data in dataset.iterrows():
            start = row_data[start_timestamp_col]
            end = row_data[end_timestamp_col]
            value = row_data[value_col]
            relevant_rows = self.data_table[
                (start <= (self.data_table.index + timedelta(milliseconds=self.granularity))) & (
                            end > self.data_table.index)]

            if aggregation == 'sum':
                self.data_table.loc[relevant_rows.index, str(value_col) + str(value)] += 1
            elif aggregation == 'binary':
                self.data_table.loc[relevant_rows.index, str(value_col) + str(value)] = 1
            else:
                raise ValueError("Unknown aggregation '" + aggregation + "'")