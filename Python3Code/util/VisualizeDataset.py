from util.util import get_chapter

import matplotlib.colors as cl
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram
import itertools
from scipy.optimize import curve_fit
import re
import math
import sys
from pathlib import Path
import dateutil
import matplotlib as mpl

mpl.use('Agg')


class VisualizeDataset:
    point_displays = ['+', 'x']
    line_displays = ['-']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    def __init__(self, module_path='.py'):
        subdir = Path(module_path).name.split('.')[0]
        self.plot_number = 1
        self.figures_dir = Path('figures') / subdir
        self.figures_dir.mkdir(exist_ok=True, parents=True)

    def save(self, plot_obj, format='png', prefix=None):
        prefix = f"{prefix}_" if prefix else ""
        fig_name = f'{prefix}fig{self.plot_number}'
        save_path = self.figures_dir / f'{fig_name}.{format}'
        plot_obj.savefig(save_path)
        print(f'Figure saved to {save_path}')
        self.plot_number += 1

    def plot_dataset_boxplot(self, dataset, cols, participant_name=None, dataset_name=None):
        # FIX: Changed plt.Figure() to plt.figure() to ensure a new figure is created for each plot.
        plt.figure()
        dataset[cols].plot.box()

        title_str = ""
        if participant_name and dataset_name:
            title_str = f"Boxplot for Participant: {participant_name}, Dataset: {dataset_name}"
        elif participant_name:
            title_str = f"Boxplot for Participant: {participant_name}"
        elif dataset_name:
            title_str = f"Boxplot for Dataset: {dataset_name}"

        if title_str:
            plt.title(title_str, fontsize=14)

        file_prefix = None
        if participant_name and dataset_name:
            file_prefix = f"boxplot_{participant_name.replace(' ', '_')}_{dataset_name.replace(' ', '_')}"
        elif participant_name:
            file_prefix = f"boxplot_{participant_name.replace(' ', '_')}"
        elif dataset_name:
            file_prefix = f"boxplot_{dataset_name.replace(' ', '_')}"

        self.save(plt, prefix=file_prefix)
        plt.close('all')

    def plot_dataset(self, data_table, columns, match='like', display='line', participant_name=None, dataset_name=None, method=None):
        data_table.index = pd.to_datetime(data_table.index)
        names = list(data_table.columns)

        if len(columns) > 1:
            f, xar = plt.subplots(len(columns), sharex=True, sharey=False)
        else:
            f, xar = plt.subplots()
            xar = [xar]

        f.subplots_adjust(hspace=0.4)

        # FIX: Changed the format string to '%H:%M:%S' to remove milliseconds.
        xfmt = md.DateFormatter('%H:%M:%S')

        for i in range(0, len(columns)):
            xar[i].xaxis.set_major_formatter(xfmt)
            xar[i].set_prop_cycle(color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])

            if match[i] == 'exact':
                relevant_cols = [columns[i]]
            elif match[i] == 'like':
                relevant_cols = [name for name in names if columns[i] == name[0:len(columns[i])]]
            else:
                raise ValueError("Match should be 'exact' or 'like' for " + str(i) + ".")

            max_values = []
            min_values = []

            for j in range(0, len(relevant_cols)):
                mask = data_table[relevant_cols[j]].replace([np.inf, -np.inf], np.nan).notnull()
                max_values.append(data_table[relevant_cols[j]][mask].max())
                min_values.append(data_table[relevant_cols[j]][mask].min())

                if display[i] == 'points':
                    xar[i].plot(data_table.index[mask], data_table[relevant_cols[j]][mask],
                                self.point_displays[j % len(self.point_displays)])
                else:
                    xar[i].plot(data_table.index[mask], data_table[relevant_cols[j]][mask],
                                self.line_displays[j % len(self.line_displays)])

            xar[i].tick_params(axis='y', labelsize=10)
            xar[i].legend(relevant_cols, fontsize='xx-small', numpoints=1, loc='upper center',
                          bbox_to_anchor=(0.5, 1.3), ncol=len(relevant_cols), fancybox=True, shadow=True)

            if min_values and max_values:
                filtered_min_values = [v for v in min_values if pd.notna(v)]
                filtered_max_values = [v for v in max_values if pd.notna(v)]

                if filtered_min_values and filtered_max_values:
                    y_min = min(filtered_min_values)
                    y_max = max(filtered_max_values)
                    buffer = 0.1 * (y_max - y_min) if (y_max - y_min) != 0 else 0.1
                    xar[i].set_ylim([y_min - buffer, y_max + buffer])
                else:
                    xar[i].set_ylim(
                        [-1, 1])
            else:
                xar[i].set_ylim([-1, 1])

        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.xlabel('time')

        title_str = ""
        if participant_name and dataset_name:
            title_str = f"Data for Participant: {participant_name}, Dataset: {dataset_name}"
        elif participant_name:
            title_str = f"Data for Participant: {participant_name}"
        elif dataset_name:
            title_str = f"Data for Dataset: {dataset_name}"

        if title_str:
            plt.suptitle(title_str, fontsize=16)

        file_prefix = None
        if participant_name and dataset_name and method:
            file_prefix = f"plot_{participant_name.replace(' ', '_')}_{dataset_name.replace(' ', '_')}_{method}"
        elif dataset_name and method:
            file_prefix = f"plot_{dataset_name.replace(' ', '_')}_{method}"
        elif participant_name and dataset_name:
            file_prefix = f"plot_{participant_name.replace(' ', '_')}_{dataset_name.replace(' ', '_')}"
        elif participant_name:
            file_prefix = f"plot_{participant_name.replace(' ', '_')}"
        elif dataset_name:
            file_prefix = f"plot_{dataset_name.replace(' ', '_')}"

        self.save(plt, prefix=file_prefix)
        plt.close('all')

    def plot_xy(self, x, y, method='plot', xlabel=None, ylabel=None, xlim=None, ylim=None, names=None,
                line_styles=None, loc=None, title=None):
        for input in x, y:
            if not hasattr(input[0], '__iter__'):
                raise TypeError('x/y should be given as a list of lists of coordinates')

        plot_method = getattr(plt, method)
        for i, (x_line, y_line) in enumerate(zip(x, y)):

            plot_method(x_line, y_line, line_styles[i]) if line_styles is not None else plt.plot(x_line, y_line)

            if xlabel is not None: plt.xlabel(xlabel)
            if ylabel is not None: plt.ylabel(ylabel)
            if xlim is not None: plt.xlim(xlim)
            if ylim is not None: plt.ylim(ylim)
            if title is not None: plt.title(title)
            if names is not None: plt.legend(names)

        self.save(plt)
        plt.close('all')

    def plot_feature_distributions_across_datasets(self, datasets, feature_cols, dataset_labels, main_title=None):
        num_features = len(feature_cols)

        n_cols = 3
        n_rows = math.ceil(num_features / n_cols)
        if num_features == 1:
            n_cols = 1

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15 * n_cols / 2, 6 * n_rows), sharex=False)
        axs = axs.flatten() if num_features > 1 else [axs]

        for i, feature_col in enumerate(feature_cols):
            ax = axs[i]
            max_val = -np.inf
            min_val = np.inf

            all_feature_data_for_col = []
            for dataset_df in datasets:
                clean_data = dataset_df[feature_col].dropna()
                if not clean_data.empty:
                    all_feature_data_for_col.extend(clean_data.tolist())
                    max_val = max(max_val, clean_data.max())
                    min_val = min(min_val, clean_data.min())

            if all_feature_data_for_col:
                num_bins = 50
                if np.isfinite(min_val) and np.isfinite(max_val) and (max_val - min_val > 0):
                    bins = np.histogram_bin_edges(all_feature_data_for_col, bins=num_bins)
                else:
                    bins = num_bins

                for j, dataset_df in enumerate(datasets):
                    clean_data = dataset_df[feature_col].dropna()
                    if not clean_data.empty:
                        ax.hist(clean_data, bins=bins, density=True, alpha=0.5, label=dataset_labels[j],
                                color=self.colors[j % len(self.colors)])

                ax.set_title(f'Distribution of {feature_col}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend(fontsize='small')
            else:
                ax.set_title(f'No data for {feature_col}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')

        for k in range(num_features, len(axs)):
            fig.delaxes(axs[k])

        if main_title:
            fig.suptitle(main_title, fontsize=18)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        file_prefix = "feature_distributions"
        self.save(fig, prefix=file_prefix)
        plt.close('all')

    def plot_correlation_heatmap(self, dataset, columns=None, title=None):
        if columns:
            df_for_corr = dataset[columns].select_dtypes(include=np.number)
        else:
            df_for_corr = dataset.select_dtypes(include=np.number)

        if df_for_corr.empty:
            print("No numerical columns to plot for correlation heatmap.")
            return

        correlation_matrix = df_for_corr.corr(method='pearson')

        plt.figure(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Pearson Correlation Coefficient')
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90, fontsize=8)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, fontsize=8)
        plt.title(title if title else 'Pearson Correlation Heatmap')
        plt.tight_layout()

        file_prefix = title.replace(" ", "_").lower() if title else "correlation_heatmap"
        self.save(plt, prefix=file_prefix)
        plt.close('all')

    def plot_feature_over_time_multi_dataset(self, datasets, feature_cols, dataset_labels, main_title=None,
                                             use_relative_time=False):
        if not datasets or not feature_cols or not dataset_labels:
            print("Please provide datasets, feature columns, and dataset labels.")
            return
        if len(datasets) != len(dataset_labels):
            print("Number of datasets must match the number of labels.")
            return

        num_features = len(feature_cols)

        n_cols = 3
        n_rows = math.ceil(num_features / n_cols)
        if num_features == 1:
            n_cols = 1

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15 * n_cols / 2, 6 * n_rows), sharex=True)
        axs = axs.flatten() if num_features > 1 else [axs]

        max_overall_relative_seconds = 0.0

        processed_datasets = []
        for dataset_df in datasets:
            df_copy = dataset_df.copy()
            if use_relative_time:
                session_start_time = df_copy.index.min()
                df_copy['relative_time_s'] = (df_copy.index - session_start_time).total_seconds()
            processed_datasets.append(df_copy)

        if use_relative_time:
            for feature_col in feature_cols:
                for dataset_df_processed in processed_datasets:
                    if feature_col in dataset_df_processed.columns:
                        valid_data = dataset_df_processed[feature_col].dropna()
                        if not valid_data.empty:
                            max_time_this_feature = dataset_df_processed.loc[valid_data.index, 'relative_time_s'].max()
                            max_overall_relative_seconds = max(max_overall_relative_seconds, max_time_this_feature)

        for i, feature_col in enumerate(feature_cols):
            ax = axs[i]
            if not use_relative_time:
                xfmt = md.DateFormatter('%H:%M')
                ax.xaxis.set_major_formatter(xfmt)

            min_y, max_y = float('inf'), float('-inf')

            for j, dataset_df_processed in enumerate(processed_datasets):
                if feature_col in dataset_df_processed.columns:
                    valid_data_points = dataset_df_processed[feature_col].dropna()

                    if not valid_data_points.empty:
                        color = self.colors[j % len(self.colors)]

                        if use_relative_time:
                            x_data_for_plot = dataset_df_processed.loc[valid_data_points.index, 'relative_time_s']
                        else:
                            x_data_for_plot = valid_data_points.index

                        ax.plot(x_data_for_plot, valid_data_points, color=color, linewidth=0.8, label=dataset_labels[j])

                        min_y = min(min_y, valid_data_points.min())
                        max_y = max(max_y, valid_data_points.max())
                    else:
                        print(f"No valid data for '{feature_col}' in dataset: {dataset_labels[j]} to plot over time.")
                else:
                    print(
                        f"Feature column '{feature_col}' not found in dataset: {dataset_labels[j]}. Skipping this subplot.")

            ax.set_title(f'{feature_col}')
            ax.set_ylabel('Value')

            if min_y != float('inf') and max_y != float('-inf') and (max_y - min_y) > 0:
                buffer = 0.1 * (max_y - min_y)
                ax.set_ylim([min_y - buffer, max_y + buffer])
            elif min_y != float('inf'):
                buffer = 0.1 * abs(min_y) if min_y != 0 else 0.1
                ax.set_ylim([min_y - buffer, min_y + buffer])
            else:
                ax.set_ylim([-1, 1])

            if any(feature_col in d.columns and not d[feature_col].dropna().empty for d in datasets):
                ax.legend(loc='upper right', fontsize='x-small')
            ax.grid(True, linestyle='--', alpha=0.6)

        for k in range(num_features, len(axs)):
            fig.delaxes(axs[k])

        fig.text(0.5, 0.04, 'Relative Time (s)' if use_relative_time else 'Time', ha='center', va='center', fontsize=12)
        if main_title:
            fig.suptitle(main_title, fontsize=18)

        if use_relative_time:
            for i, ax in enumerate(axs[:num_features]):
                max_time_for_subplot = 0
                for j, dataset_df_processed in enumerate(processed_datasets):
                    feature_col = feature_cols[i]
                    if feature_col in dataset_df_processed.columns:
                        valid_data = dataset_df_processed[feature_col].dropna()
                        if not valid_data.empty:
                            max_time_this_dataset = dataset_df_processed.loc[valid_data.index, 'relative_time_s'].max()
                            max_time_for_subplot = max(max_time_for_subplot, max_time_this_dataset)

                if max_time_for_subplot > 0:
                    ax.set_xlim([0, max_time_for_subplot * 1.05])

        fig.tight_layout(rect=[0, 0.06, 1, 0.95])

        file_prefix = "all_features_over_relative_time_multi_dataset" if use_relative_time else "all_features_over_time_multi_dataset"
        self.save(fig, prefix=file_prefix)

    def plot_binary_outliers(self, data_table, col, outlier_col):
        data_table.loc[:, :] = data_table.dropna(axis=0, subset=[col, outlier_col])
        data_table.loc[:, outlier_col] = data_table[outlier_col].astype('bool')
        f, xar = plt.subplots()
        xfmt = md.DateFormatter('%H:%M')
        xar.xaxis.set_major_formatter(xfmt)
        plt.xlabel('time')
        plt.ylabel('value')
        xar.plot(data_table.index[data_table[outlier_col]], data_table[col][data_table[outlier_col]], 'r+')
        xar.plot(data_table.index[~data_table[outlier_col]], data_table[col][~data_table[outlier_col]], 'b+')
        plt.legend(['outlier ' + col, 'no_outlier_' + col], numpoints=1, fontsize='xx-small', loc='upper center',
                   ncol=2, fancybox=True, shadow=True)
        self.save(plt)
        plt.close('all')

    def plot_imputed_values(self, data_table, names, col, *values):

        xfmt = md.DateFormatter('%H:%M')

        if len(values) > 0:
            f, xar = plt.subplots(len(values) + 1, sharex=True, sharey=False)
        else:
            f, xar = plt.subplots()
            xar = [xar]

        f.subplots_adjust(hspace=0.4)

        xar[0].xaxis.set_major_formatter(xfmt)
        xar[0].plot(data_table.index[data_table[col].notnull()], data_table[col][data_table[col].notnull()], 'b+',
                    markersize='2')
        xar[0].legend([names[0]], fontsize='small', numpoints=1, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=1,
                      fancybox=True, shadow=True)

        for i in range(1, len(values) + 1):
            xar[i].xaxis.set_major_formatter(xfmt)
            xar[i].plot(data_table.index, values[i - 1], 'b+', markersize='2')
            xar[i].legend([names[i]], fontsize='small', numpoints=1, loc='upper center', bbox_to_anchor=(0.5, 1.3),
                          ncol=1, fancybox=True, shadow=True)

        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.xlabel('time')
        self.save(plt)
        plt.close('all')

    def plot_clusters_3d(self, data_table, data_cols, cluster_col, label_cols):

        color_index = 0
        point_displays = ['+', 'x', '*', 'd', 'o', 's', '<', '>']

        clusters = data_table[cluster_col].unique()
        labels = []

        for i in range(0, len(label_cols)):
            labels.extend([name for name in list(data_table.columns) if label_cols[i] == name[0:len(label_cols[i])]])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        handles = []

        for cluster in clusters:
            marker_index = 0
            for label in labels:
                rows = data_table.loc[(data_table[cluster_col] == cluster) & (data_table[label] > 0)]
                if not len(data_cols) == 3:
                    return
                plot_color = self.colors[color_index % len(self.colors)]
                plot_marker = point_displays[marker_index % len(point_displays)]
                pt = ax.scatter(rows[data_cols[0]], rows[data_cols[1]], rows[data_cols[2]], c=plot_color,
                                marker=plot_marker)
                if color_index == 0:
                    handles.append(pt)
                ax.set_xlabel(data_cols[0])
                ax.set_ylabel(data_cols[1])
                ax.set_zlabel(data_cols[2])
                marker_index += 1
            color_index += 1

        plt.legend(handles, labels, fontsize='xx-small', numpoints=1)
        self.save(plt)
        plt.close('all')

    def plot_silhouette(self, data_table, cluster_col, silhouette_col):

        clusters = data_table[cluster_col].unique()

        fig, ax1 = plt.subplots(1, 1)
        ax1.set_xlim([-0.1, 1])
        y_lower = 10
        for i in range(0, len(clusters)):
            rows = data_table.mask(data_table[cluster_col] != clusters[i])
            ith_cluster_silhouette_values = np.array(rows[silhouette_col])
            ith_cluster_silhouette_values.sort()

            size_cluster_i = len(rows.index)
            y_upper = y_lower + size_cluster_i

            color = plt.get_cmap('Spectral')(float(i) / len(clusters))
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=data_table[silhouette_col].mean(), color="red", linestyle="--")

        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        self.save(plt)
        plt.close('all')

    def plot_dendrogram(self, dataset, linkage):
        sys.setrecursionlimit(40000)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('time points')
        plt.ylabel('distance')
        times = dataset.index.strftime('%H:%M:%S')
        dendrogram(linkage, truncate_mode='lastp', p=16, show_leaf_counts=True, leaf_rotation=45., leaf_font_size=8.,
                   show_contracted=True, labels=times)
        self.save(plt)
        plt.close('all')

    def plot_confusion_matrix(self, cm, classes, normalize=False):
        cmap = plt.cm.Blues
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        self.save(plt)
        plt.close('all')

    def plot_numerical_prediction_versus_real(self, train_time, train_y, regr_train_y, test_time, test_y, regr_test_y,
                                              label):
        self.legends = {}

        f, xar = plt.subplots(1, 1)

        xfmt = md.DateFormatter('%H:%M')
        xar.xaxis.set_major_formatter(xfmt)
        xar.set_prop_cycle(color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        plt.plot(train_time, train_y, '-', linewidth=0.5)
        plt.plot(train_time, regr_train_y, '--', linewidth=0.5)

        plt.plot(test_time, test_y, '-', linewidth=0.5)
        plt.plot(test_time, regr_test_y, '--', linewidth=0.5)

        plt.legend(['real values training', 'predicted values training', 'real values test', 'predicted values test'],
                   loc=4)

        max_y_value = max(max(train_y.tolist()), max(regr_train_y.tolist()), max(test_y.tolist()),
                          max(regr_test_y.tolist()))
        min_y_value = min(min(train_y.tolist()), min(regr_train_y.tolist()), min(test_y.tolist()),
                          min(regr_test_y.tolist()))
        range = max_y_value - min_y_value
        y_coord_labels = max(max(train_y.tolist()), max(regr_train_y.tolist()), max(test_y.tolist()),
                             max(regr_test_y.tolist())) + (0.01 * range)

        plt.title('Performance of model for ' + str(label))
        plt.ylabel(label)
        plt.xlabel('time')
        plt.annotate('', xy=(train_time[0], y_coord_labels), xycoords='data', xytext=(train_time[-1], y_coord_labels),
                     textcoords='data', arrowprops={'arrowstyle': '<->'})
        plt.annotate('training set', xy=(train_time[int(float(len(train_time)) / 2)], y_coord_labels * 1.02),
                     color='blue', xycoords='data', ha='center')
        plt.annotate('', xy=(test_time[0], y_coord_labels), xycoords='data', xytext=(test_time[-1], y_coord_labels),
                     textcoords='data', arrowprops={'arrowstyle': '<->'})
        plt.annotate('test set', xy=(test_time[int(float(len(test_time)) / 2)], y_coord_labels * 1.02), color='red',
                     xycoords='data', ha='center')
        self.save(plt)
        plt.close('all')

    def plot_pareto_front(self, dynsys_output):
        fit_1_train = []
        fit_2_train = []
        fit_1_test = []
        fit_2_test = []
        for row in dynsys_output:
            fit_1_train.append(row[1][0])
            fit_2_train.append(row[1][1])

        plt.scatter(fit_1_train, fit_2_train, color='r')
        plt.xlabel('mse on ' + str(dynsys_output[0][0].columns[0]))
        plt.ylabel('mse on ' + str(dynsys_output[0][0].columns[1]))
        self.save(plt)
        plt.close('all')

    def plot_numerical_prediction_versus_real_dynsys_mo(self, train_time, train_y, test_time, test_y, dynsys_output,
                                                        individual, label):
        regr_train_y = dynsys_output[individual][0][label]
        regr_test_y = dynsys_output[individual][2][label]
        train_y = train_y[label]
        test_y = test_y[label]
        self.plot_numerical_prediction_versus_real(train_time, train_y, regr_train_y, test_time, test_y, regr_test_y,
                                                   label)

    def plot_performances(self, algs, feature_subset_names, scores_over_all_algs, ylim, std_mult, y_name):

        width = float(1) / (len(feature_subset_names) + 1)
        ind = np.arange(len(algs))
        for i in range(0, len(feature_subset_names)):
            means = []
            std = []
            for j in range(0, len(algs)):
                means.append(scores_over_all_algs[i][j][2])
                std.append(std_mult * scores_over_all_algs[i][j][3])
            plt.errorbar(ind + i * width, means, yerr=std, fmt=self.colors[i % len(self.colors)] + 'o', markersize='3')
        plt.ylabel(y_name)
        plt.xticks(ind + (float(len(feature_subset_names)) / 2) * width, algs)
        plt.legend(feature_subset_names, loc=4, numpoints=1)
        if not ylim is None:
            plt.ylim(ylim)
        self.save(plt)
        plt.close('all')

    def plot_performances_classification(self, algs, feature_subset_names, scores_over_all_algs):
        self.plot_performances(algs, feature_subset_names, scores_over_all_algs, [0.70, 1.0], 2, 'Accuracy')

    def plot_performances_regression(self, algs, feature_subset_names, scores_over_all_algs):
        self.plot_performances(algs, feature_subset_names, scores_over_all_algs, None, 1, 'Mean Squared Error')