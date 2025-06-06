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

    point_displays = ['+', 'x'] #'*', 'd', 'o', 's', '<', '>']
    line_displays = ['-'] #, '--', ':', '-.']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    # Set some initial attributes to define and create a save location for the images.
    def __init__(self, module_path='.py'):
        subdir = Path(module_path).name.split('.')[0]

        self.plot_number = 1
        self.figures_dir = Path('figures') / subdir
        self.figures_dir.mkdir(exist_ok=True, parents=True)


    def save(self, plot_obj, format='png', prefix=None):  # 'svg'
        # Use the provided prefix or default to an empty string
        prefix = f"{prefix}_" if prefix else ""
        fig_name = f'{prefix}fig{self.plot_number}'

        save_path = self.figures_dir / f'{fig_name}.{format}'
        plot_obj.savefig(save_path)
        print(f'Figure saved to {save_path}')

        self.plot_number += 1

    # Plot the dataset, here columns can specify a specific attribute, but also a generic name that occurs
    # among multiple attributes (e.g. label which occurs as labelWalking, etc). In such a case they are plotted
    # in the same graph. The display should express whether points or a line should be plotted.
    # Match can be 'exact' or 'like'. Display can be 'points' or 'line'.
    def plot_dataset(self, data_table, columns, match='like', display='line', participant_name=None,dataset_name=None):
        names = list(data_table.columns)

        # Create subplots if more columns are specified.
        if len(columns) > 1:
            f, xar = plt.subplots(len(columns), sharex=True, sharey=False)
        else:
            f, xar = plt.subplots()
            xar = [xar]

        f.subplots_adjust(hspace=0.4)

        xfmt = md.DateFormatter('%H:%M')

        # Pass through the columns specified.
        for i in range(0, len(columns)):
            xar[i].xaxis.set_major_formatter(xfmt)
            xar[i].set_prop_cycle(color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])
            # if a column match is specified as 'exact', select the column name(s) with an exact match.
            # If it's specified as 'like', select columns containing the name.

            # We can match exact (i.e. a columns name is an exact name of a columns or 'like' for
            # which we need to find columns names in the dataset that contain the name.
            if match[i] == 'exact':
                relevant_cols = [columns[i]]
            elif match[i] == 'like':
                relevant_cols = [name for name in names if columns[i] == name[0:len(columns[i])]]
            else:
                raise ValueError("Match should be 'exact' or 'like' for " + str(i) + ".")

            max_values = []
            min_values = []

            # Pass through the relevant columns.
            for j in range(0, len(relevant_cols)):
                # Create a mask to ignore the NaN and Inf values when plotting:
                mask = data_table[relevant_cols[j]].replace([np.inf, -np.inf], np.nan).notnull()
                max_values.append(data_table[relevant_cols[j]][mask].max())
                min_values.append(data_table[relevant_cols[j]][mask].min())

                # Display point, or as a line
                if display[i] == 'points':
                    xar[i].plot(data_table.index[mask], data_table[relevant_cols[j]][mask],
                                self.point_displays[j % len(self.point_displays)])
                else:
                    xar[i].plot(data_table.index[mask], data_table[relevant_cols[j]][mask],
                                self.line_displays[j % len(self.line_displays)])

            xar[i].tick_params(axis='y', labelsize=10)
            xar[i].legend(relevant_cols, fontsize='xx-small', numpoints=1, loc='upper center',
                          bbox_to_anchor=(0.5, 1.3), ncol=len(relevant_cols), fancybox=True, shadow=True)

            # Added Robustness to set_ylim (from previous solution)
            if min_values and max_values:
                filtered_min_values = [v for v in min_values if pd.notna(v)]
                filtered_max_values = [v for v in max_values if pd.notna(v)]

                if filtered_min_values and filtered_max_values:
                    y_min = min(filtered_min_values)
                    y_max = max(filtered_max_values)
                    buffer = 0.1 * (y_max - y_min) if (
                                                                  y_max - y_min) != 0 else 0.1  # Ensure buffer is not zero for flat lines
                    xar[i].set_ylim([y_min - buffer, y_max + buffer])
                else:
                    xar[i].set_ylim(
                        [-1, 1])  # Default range if filtered lists are empty (e.g., all data for a subplot is NaN)
            else:
                xar[i].set_ylim([-1, 1])  # Default range if min/max lists are empty

        # Make sure we get a nice figure with only a single x-axis and labels there.
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.xlabel('time')

        # Add the main title to the figure using the new parameters
        title_str = ""
        if participant_name and dataset_name:
            title_str = f"Data for Participant: {participant_name}, Dataset: {dataset_name}"
        elif participant_name:
            title_str = f"Data for Participant: {participant_name}"
        elif dataset_name:
            title_str = f"Data for Dataset: {dataset_name}"

        if title_str:
            plt.suptitle(title_str, fontsize=16)  # Using suptitle for a main title over all subplots

        # Generate a filename prefix based on the provided names
        file_prefix = None
        if participant_name and dataset_name:
            file_prefix = f"plot_{participant_name.replace(' ', '_')}_{dataset_name.replace(' ', '_')}"
        elif participant_name:
            file_prefix = f"plot_{participant_name.replace(' ', '_')}"
        elif dataset_name:
            file_prefix = f"plot_{dataset_name.replace(' ', '_')}"

        self.save(plt, prefix=file_prefix)  # Pass the new prefix to the save method
        

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
        

    def plot_dataset_boxplot(self, dataset, cols, participant_name=None,dataset_name=None):
        plt.Figure(); dataset[cols].plot.box()

        # Set title for the boxplot
        title_str = ""
        if participant_name and dataset_name:
            title_str = f"Boxplot for Participant: {participant_name}, Dataset: {dataset_name}"
        elif participant_name:
            title_str = f"Boxplot for Participant: {participant_name}"
        elif dataset_name:
            title_str = f"Boxplot for Dataset: {dataset_name}"

        if title_str:
            plt.title(title_str, fontsize=14)

        # Generate a filename prefix based on the provided names
        file_prefix = None
        if participant_name and dataset_name:
            file_prefix = f"boxplot_{participant_name.replace(' ', '_')}_{dataset_name.replace(' ', '_')}"
        elif participant_name:
            file_prefix = f"boxplot_{participant_name.replace(' ', '_')}"
        elif dataset_name:
            file_prefix = f"boxplot_{dataset_name.replace(' ', '_')}"

        self.save(plt, prefix=file_prefix)
        

    def plot_feature_distributions_across_datasets(self, datasets, feature_cols, dataset_labels, main_title=None):
        num_features = len(feature_cols)

        # Determine grid for subplots (e.g., 2x2, 2x3, etc.)
        n_rows = math.ceil(num_features / 2.0) if num_features > 1 else 1
        n_cols = 2 if num_features > 1 else 1
        if num_features == 1:  # Single plot case, to avoid issue with axs being a single Axes object
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 5))
            axs = [axs]  # Make it a list so we can iterate consistently
        else:
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 5 * n_rows))
            axs = axs.flatten()  # Flatten 2D array of axes for easy iteration

        # Iterate through each feature column to create a subplot
        for i, feature_col in enumerate(feature_cols):
            ax = axs[i]
            max_val = -np.inf
            min_val = np.inf

            # Collect data for bin calculation and limits
            all_feature_data_for_col = []
            for dataset_df in datasets:
                # Filter out NaN values for robust plotting
                clean_data = dataset_df[feature_col].dropna()
                if not clean_data.empty:
                    all_feature_data_for_col.extend(clean_data.tolist())
                    max_val = max(max_val, clean_data.max())
                    min_val = min(min_val, clean_data.min())

            # Only plot if there's valid data for this feature column across datasets
            if all_feature_data_for_col:
                # Determine common bins across all datasets for this feature for consistent comparison
                num_bins = 50  # You can adjust this number
                # Use numpy.histogram_bin_edges to get consistent bins if min/max are finite
                if np.isfinite(min_val) and np.isfinite(max_val) and (max_val - min_val > 0):
                    bins = np.histogram_bin_edges(all_feature_data_for_col, bins=num_bins)
                else:
                    bins = num_bins  # Fallback to auto-bins if range is zero or infinite

                # Plot histogram for each dataset
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

        # Hide any unused subplots
        for k in range(num_features, len(axs)):
            fig.delaxes(axs[k])

        if main_title:
            fig.suptitle(main_title, fontsize=18)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

        # Generate a filename prefix for this comparison plot
        file_prefix = "feature_distributions"
        self.save(fig, prefix=file_prefix)
        plt.close(fig)

    def plot_correlation_heatmap(self, dataset, columns=None, title=None):
        """
        Plots a heatmap of the Pearson correlation matrix for numerical columns in the dataset.

        Parameters:
        - dataset (pd.DataFrame): The input pandas DataFrame.
        - columns (list, optional): A list of column names to include in the correlation matrix.
                                    If None, all numerical columns will be used.
        - title (str, optional): The title of the plot.
        """
        if columns:
            df_for_corr = dataset[columns].select_dtypes(include=np.number)
        else:
            df_for_corr = dataset.select_dtypes(include=np.number)  # Select only numerical columns

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
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping

        file_prefix = title.replace(" ", "_").lower() if title else "correlation_heatmap"
        self.save(plt, prefix=file_prefix)
        plt.close(plt)

    # This function plots the real and imaginary amplitudes of the frequencies found in the Fourier transformation.
    def plot_fourier_amplitudes(self, freq, ampl_real, ampl_imag):
        plt.xlabel('Freq(Hz)')
        plt.ylabel('amplitude')
        # Plot the real values as a '+' and imaginary in the same way (though with a different color).
        plt.plot(freq, ampl_real, '+', freq, ampl_imag,'+')
        plt.legend(['real', 'imaginary'], numpoints=1)
        self.save(plt)
        plt.close(plt)
        

    # Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    # column and outlier_col the columns with a binary value (outlier or not)
    def plot_binary_outliers(self, data_table, col, outlier_col):
        data_table.loc[:,:] = data_table.dropna(axis=0, subset=[col, outlier_col])
        data_table.loc[:,outlier_col] = data_table[outlier_col].astype('bool')
        f, xar = plt.subplots()
        xfmt = md.DateFormatter('%H:%M')
        xar.xaxis.set_major_formatter(xfmt)
        plt.xlabel('time')
        plt.ylabel('value')
        # Plot data points that are outliers in red, and non outliers in blue.
        xar.plot(data_table.index[data_table[outlier_col]], data_table[col][data_table[outlier_col]], 'r+')
        xar.plot(data_table.index[~data_table[outlier_col]], data_table[col][~data_table[outlier_col]], 'b+')
        plt.legend(['outlier ' + col, 'no_outlier_' + col], numpoints=1, fontsize='xx-small', loc='upper center',  ncol=2, fancybox=True, shadow=True)
        self.save(plt)
        plt.close(plt)
        

    # Plot values that have been imputed using one of our imputation approaches. Here, values expresses the
    # 1 to n datasets that have resulted from value imputation.
    def plot_imputed_values(self, data_table, names, col, *values):

        xfmt = md.DateFormatter('%H:%M')

        # Create proper subplots.
        if len(values) > 0:
            f, xar = plt.subplots(len(values) + 1, sharex=True, sharey=False)
        else:
            f, xar = plt.subplots()
            xar = [xar]

        f.subplots_adjust(hspace=0.4)

        # plot the regular dataset.

        xar[0].xaxis.set_major_formatter(xfmt)
        xar[0].plot(data_table.index[data_table[col].notnull()], data_table[col][data_table[col].notnull()], 'b+', markersize='2')
        xar[0].legend([names[0]], fontsize='small', numpoints=1, loc='upper center',  bbox_to_anchor=(0.5, 1.3), ncol=1, fancybox=True, shadow=True)

        # and plot the others that have resulted from imputation.
        for i in range(1, len(values)+1):
            xar[i].xaxis.set_major_formatter(xfmt)
            xar[i].plot(data_table.index, values[i-1], 'b+', markersize='2')
            xar[i].legend([names[i]], fontsize='small', numpoints=1, loc='upper center',  bbox_to_anchor=(0.5, 1.3), ncol=1, fancybox=True, shadow=True)

        # Diplay is nicely in subplots.
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.xlabel('time')
        self.save(plt)
        plt.close(plt)
        

    # This function plots clusters that result from the application of a clustering algorithm
    # and also shows the class label of points. Clusters are displayed via colors, classes
    # by means of different types of points. We assume that three data columns are clustered
    # that do not include the label. We assume the labels to be represented by 1 or more binary
    # columns.
    def plot_clusters_3d(self, data_table, data_cols, cluster_col, label_cols):

        color_index = 0
        point_displays = ['+', 'x', '*', 'd', 'o', 's', '<', '>']

        # Determine the number of clusters:
        clusters = data_table[cluster_col].unique()
        labels = []

        # Get the possible labels, assuming 1 or more label columns with binary values.
        for i in range(0, len(label_cols)):
            labels.extend([name for name in list(data_table.columns) if label_cols[i] == name[0:len(label_cols[i])]])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        handles = []

        # Plot clusters individually with a certain color.
        for cluster in clusters:
            marker_index = 0
            # And make sure the points of a label receive the right marker type.
            for label in labels:
                rows = data_table.loc[(data_table[cluster_col] == cluster) & (data_table[label] > 0)]
                # Now we come to the assumption that there are three data_cols specified:
                if not len(data_cols) == 3:
                    return
                plot_color = self.colors[color_index%len(self.colors)]
                plot_marker = point_displays[marker_index%len(point_displays)]
                pt = ax.scatter(rows[data_cols[0]], rows[data_cols[1]], rows[data_cols[2]], c=plot_color, marker=plot_marker)
                if color_index == 0:
                    handles.append(pt)
                ax.set_xlabel(data_cols[0])
                ax.set_ylabel(data_cols[1])
                ax.set_zlabel(data_cols[2])
                marker_index += 1
            color_index += 1

        plt.legend(handles, labels, fontsize='xx-small', numpoints=1)
        self.save(plt)
        plt.close(plt)

    # This function plots the silhouettes of the different clusters that have been identified. It plots the
    # silhouette of the individual datapoints per cluster to allow studying the clusters internally as well.
    # For this, a column expressing the silhouette for each datapoint is assumed.
    def plot_silhouette(self, data_table, cluster_col, silhouette_col):
        # Taken from the examples of scikit learn
        #(http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)

        clusters = data_table[cluster_col].unique()

        fig, ax1 = plt.subplots(1, 1)
        ax1.set_xlim([-0.1, 1])
        #ax1.set_ylim([0, len(data_table.index) + (len(clusters) + 1) * 10])
        y_lower = 10
        for i in range(0, len(clusters)):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            rows = data_table.mask(data_table[cluster_col] != clusters[i])
            ith_cluster_silhouette_values = np.array(rows[silhouette_col])
            ith_cluster_silhouette_values.sort()

            size_cluster_i = len(rows.index)
            y_upper = y_lower + size_cluster_i

            color = plt.get_cmap('Spectral')(float(i) / len(clusters))
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=data_table[silhouette_col].mean(), color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        self.save(plt)
        plt.close(plt)

    # Plot a dendorgram for hierarchical clustering. It assumes that the linkage as
    # used in sk learn is passed as an argument as well.
    def plot_dendrogram(self, dataset, linkage):
        sys.setrecursionlimit(40000)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('time points')
        plt.ylabel('distance')
        times = dataset.index.strftime('%H:%M:%S')
        #dendrogram(linkage,truncate_mode='lastp',p=10, show_leaf_counts=True, leaf_rotation=90.,leaf_font_size=12.,show_contracted=True, labels=times)
        dendrogram(linkage,truncate_mode='lastp',p=16, show_leaf_counts=True, leaf_rotation=45.,leaf_font_size=8.,show_contracted=True, labels=times)
        self.save(plt)
        plt.close(plt)

    # Plot the confusion matrix that has been derived in the evaluation metrics. Classes expresses the labels
    # for the matrix. We can normalize or show the raw counts. Of course this applies to classification problems.
    def plot_confusion_matrix(self, cm, classes, normalize=False):
        # Taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        # Select the colormap.
        cmap=plt.cm.Blues
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
        plt.close(plt)

    # This function plots the predictions or an algorithms (both for the training and test set) versus the real values for
    # a regression problem. It assumes only a single value to be predicted over a number of cases. The variables identified
    # with reg_ are the predictions.
    def plot_numerical_prediction_versus_real(self, train_time, train_y, regr_train_y, test_time, test_y, regr_test_y, label):
        self.legends = {}

        # Plot the values, training set cases in blue, test set in red.
        f, xar = plt.subplots(1, 1)

        xfmt = md.DateFormatter('%H:%M')
        xar.xaxis.set_major_formatter(xfmt)
        xar.set_prop_cycle(color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        plt.plot(train_time, train_y, '-', linewidth=0.5)
        plt.plot(train_time, regr_train_y, '--', linewidth=0.5)

        plt.plot(test_time, test_y, '-', linewidth=0.5)
        plt.plot(test_time, regr_test_y, '--', linewidth=0.5)

        plt.legend(['real values training', 'predicted values training', 'real values test', 'predicted values test'], loc=4)


        # And create some fancy stuff in the figure to label the training and test set a bit clearer.
        max_y_value = max(max(train_y.tolist()), max(regr_train_y.tolist()), max(test_y.tolist()), max(regr_test_y.tolist()))
        min_y_value = min(min(train_y.tolist()), min(regr_train_y.tolist()), min(test_y.tolist()), min(regr_test_y.tolist()))
        range = max_y_value - min_y_value
        y_coord_labels = max(max(train_y.tolist()), max(regr_train_y.tolist()), max(test_y.tolist()), max(regr_test_y.tolist()))+(0.01*range)


        plt.title('Performance of model for ' + str(label))
        plt.ylabel(label)
        plt.xlabel('time')
        plt.annotate('', xy=(train_time[0],y_coord_labels), xycoords='data', xytext=(train_time[-1], y_coord_labels), textcoords='data', arrowprops={'arrowstyle': '<->'})
        plt.annotate('training set', xy=(train_time[int(float(len(train_time))/2)], y_coord_labels*1.02), color='blue', xycoords='data', ha='center')
        plt.annotate('', xy=(test_time[0], y_coord_labels), xycoords='data', xytext=(test_time[-1], y_coord_labels), textcoords='data', arrowprops={'arrowstyle': '<->'})
        plt.annotate('test set', xy=(test_time[int(float(len(test_time))/2)], y_coord_labels*1.02), color='red', xycoords='data', ha='center')
        self.save(plt)
        plt.close(plt)

    # Plot the Pareto front for multi objective optimization problems (for the dynamical systems stuff). We consider the
    # raw output of the MO dynamical systems approach, which includes rows with the fitness and predictions for the training
    # and test set. We select the fitness and plot them in a graph. Note that the plot only considers the first two dimensions.
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
        #plt.savefig('{0} Example ({1}).pdf'.format(ea.__class__.__name__, problem.__class__.__name__), format='pdf')
        self.save(plt)
        plt.close(plt)

    # Plot a prediction for a regression model in case it concerns a multi-objective dynamical systems model. Here, we plot
    # the individual specified. Again, the complete output of the MO approach is used as argument.
    def plot_numerical_prediction_versus_real_dynsys_mo(self, train_time, train_y, test_time, test_y, dynsys_output, individual, label):
        regr_train_y = dynsys_output[individual][0][label]
        regr_test_y = dynsys_output[individual][2][label]
        train_y = train_y[label]
        test_y = test_y[label]
        self.plot_numerical_prediction_versus_real(train_time, train_y, regr_train_y, test_time, test_y, regr_test_y, label)

    # Visualizes the performance of different algorithms over different feature sets. Assumes the scores to contain
    # a score on the training set followed by an sd, and the same for the test set.
    def plot_performances(self, algs, feature_subset_names, scores_over_all_algs, ylim, std_mult, y_name):

        width = float(1)/(len(feature_subset_names)+1)
        ind = np.arange(len(algs))
        for i in range(0, len(feature_subset_names)):
            means = []
            std = []
            for j in range(0, len(algs)):
                means.append(scores_over_all_algs[i][j][2])
                std.append(std_mult * scores_over_all_algs[i][j][3])
            plt.errorbar(ind + i * width, means, yerr=std, fmt=self.colors[i%len(self.colors)] + 'o', markersize='3')
        plt.ylabel(y_name)
        plt.xticks(ind+(float(len(feature_subset_names))/2)*width, algs)
        plt.legend(feature_subset_names, loc=4, numpoints=1)
        if not ylim is None:
            plt.ylim(ylim)
        self.save(plt)
        plt.close(plt)

    def plot_performances_classification(self, algs, feature_subset_names, scores_over_all_algs):
        self.plot_performances(algs, feature_subset_names, scores_over_all_algs, [0.70, 1.0], 2, 'Accuracy')

    def plot_performances_regression(self, algs, feature_subset_names, scores_over_all_algs):
        self.plot_performances(algs, feature_subset_names, scores_over_all_algs, None, 1, 'Mean Squared Error')
