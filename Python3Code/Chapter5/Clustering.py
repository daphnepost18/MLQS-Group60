##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 5                                               #
#                                                            #
##############################################################

from sklearn.cluster import KMeans
from Chapter5.DistanceMetrics import InstanceDistanceMetrics
import sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from Chapter5.DistanceMetrics import PersonDistanceMetricsNoOrdering
from Chapter5.DistanceMetrics import PersonDistanceMetricsOrdering
import random
import scipy
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import DistanceMetric
import pyclust

from nltk.cluster.kmeans import KMeansClusterer

# Implementation of the non hierarchical clustering approaches.
class NonHierarchicalClustering:

    # Global parameters for distance functions
    p = 1
    max_lag = 1

    # Identifiers of the various distance and abstraction approaches.
    euclidean = 'euclidean'
    minkowski = 'minkowski'
    manhattan = 'manhattan'
    gower = 'gower'
    abstraction_mean = 'abstraction_mean'
    abstraction_normal = 'abstraction_normal'
    abstraction_p = 'abstraction_p'
    abstraction_euclidean = 'abstract_euclidean'
    abstraction_lag = 'abstract_lag'
    abstraction_dtw = 'abstract_dtw'

    # Define the gowers distance between arrays to be used in k-means and k-medoids.
    def gowers_similarity(self, X, Y=None, Y_norm_squared=None, squared=False):
        X = np.matrix(X)
        distances = np.zeros(shape=(X.shape[0], Y.shape[0]))
        DM = InstanceDistanceMetrics()
        # Pairs up the elements in the dataset
        for x_row in range(0, X.shape[0]):
            data_row1 = pd.DataFrame(X[x_row])
            for y_row in range(0, Y.shape[0]):
                data_row2 = pd.DataFrame(Y[y_row]).transpose()
                # And computer the distance as defined in our distance metrics class.
                distances[x_row, y_row] = DM.gowers_similarity(data_row1, data_row2, self.p)
        return np.array(distances)

    # Use a predefined distance function for the Minkowski distance
    def minkowski_distance(self, X, Y=None, Y_norm_squared=None, squared=False):
        dist = DistanceMetric.get_metric('minkowski', p=self.p)
        return dist.pairwise(X, Y)

    # Use a predefined distance function for the Manhattan distance
    def manhattan_distance(self, X, Y=None, Y_norm_squared=None, squared=False):
        dist = DistanceMetric.get_metric('manhattan')
        return dist.pairwise(X, Y)

    # Use a predefined distance function for the Euclidean distance
    def euclidean_distance(self, X, Y=None, Y_norm_squared=None, squared=False):
        dist = DistanceMetric.get_metric('euclidean')
        return dist.pairwise(X, Y)

    # If we want to compare dataset between persons one approach is to flatten
    # each dataset to a single record/instance. This is done based on the approaches
    # we have defined in the distance metrics file.
    def aggregate_datasets(self, datasets, cols, abstraction_method):
        temp_datasets = []
        DM = PersonDistanceMetricsNoOrdering()

        # Flatten all datasets and add them to the newly formed dataset.
        for i in range(0, len(datasets)):
            temp_dataset = datasets[i][cols]
            temp_datasets.append(temp_dataset)

        if abstraction_method == self.abstraction_normal:
            return DM.create_instances_normal_distribution(temp_datasets)
        else:
            return DM.create_instances_mean(temp_datasets)

    # Perform k-means over an individual dataset.
    def k_means_over_instances(self, dataset, cols, k, distance_metric, max_iters, n_inits, p=1):

        # Take the appropriate columns.
        temp_dataset = dataset[cols]
        # Override the standard distance functions. Store the original first
        # sklearn_euclidian_distances = sklearn.cluster.k_means_.euclidean_distances
        sklearn_euclidian_distances = sklearn.metrics.pairwise.euclidean_distances
        if distance_metric == self.euclidean:
            sklearn.metrics.pairwise.euclidean_distances = self.euclidean_distance
        elif distance_metric == self.minkowski:
            self.p = p
            sklearn.metrics.pairwise.euclidean_distances = self.minkowski_distance
        elif distance_metric == self.manhattan:
            sklearn.metrics.pairwise.euclidean_distances = self.manhattan_distance
        elif distance_metric == self.gower:
            self.ranges = []
            for col in temp_dataset.columns:
                self.ranges.append(temp_dataset[col].max() - temp_dataset[col].min())
            sklearn.metrics.pairwise.euclidean_distances = self.gowers_similarity
        # If we do not recognize the option we use the default distance function, which is much
        # faster....
        # Now apply the k-means algorithm
        kmeans = KMeans(n_clusters=k, max_iter=max_iters, n_init=n_inits, random_state=0).fit(temp_dataset)
        # Add the labels to the dataset
        dataset['cluster'] = kmeans.labels_
        # Compute the solhouette and add it as well.
        silhouette_avg = silhouette_score(temp_dataset, kmeans.labels_)
        silhouette_per_inst = silhouette_samples(temp_dataset, kmeans.labels_)
        dataset['silhouette'] = silhouette_per_inst

        # Reset the module distance function for further usage
        sklearn_euclidian_distances = sklearn_euclidian_distances

        return dataset

    # We have datasets covering multiple persons. We abstract the datatasets using an approach and create
    # clusters of persons.
    def k_means_over_datasets(self, datasets, cols, k, abstraction_method, distance_metric, max_iters, n_inits, p=1):
        # Convert the datasets to instances
        temp_dataset = self.aggregate_datasets(datasets, cols, abstraction_method)

        # And simply apply the instance based algorithm.....
        return self.k_means_over_instances(temp_dataset, temp_dataset.columns, k, distance_metric, max_iters, n_inits, p)

    # For our own k-medoids algorithm we use our own implementation. For this we computer a complete distance matrix
    # between points.
    def compute_distance_matrix_instances(self, dataset, distance_metric):
        # If the distance function is not defined in our distance metrics, we use the standard euclidean distance.
        if not (distance_metric in [self.manhattan, self.minkowski, self.gower, self.euclidean]):
            distances = sklearn.metrics.pairwise.euclidean_distances(X=dataset, Y=dataset)
            return pd.DataFrame(distances, index=range(0, len(dataset.index)), columns=range(0, len(dataset.index)))
        # Create an empty pandas dataframe for our distance matrix
        distances = pd.DataFrame(index=range(0, len(dataset.index)), columns=range(0, len(dataset.index)))
        DM = InstanceDistanceMetrics()

        # Define the ranges of the columns if we use the gower distance.
        ranges = []
        if distance_metric == self.gower:
            for col in dataset.columns:
                self.ranges.append(dataset[col].max() - dataset[col].min())

        # And compute the distances for each pair. Note that we assume the distances to be symmetric.
        for i in range(0, len(dataset.index)):
            for j in range(i, len(dataset.index)):
                if distance_metric == self.manhattan:
                    distances.iloc[i,j] = self.manhattan_distance(dataset.iloc[i:i+1,:], dataset.iloc[j:j+1,:])
                elif distance_metric == self.minkowski:
                    distances.iloc[i,j] = self.manhattan_distance(dataset.iloc[i:i+1,:], dataset.iloc[j:j+1,:], self.p)
                elif distance_metric == self.gower:
                    distances.iloc[i,j] = self.gowers_similarity(dataset.iloc[i:i+1,:], dataset.iloc[j:j+1,:])
                elif distance_metric == self.euclidean:
                    distances.iloc[i,j] = self.euclidean_distance(dataset.iloc[i:i+1,:], dataset.iloc[j:j+1,:])
                distances.iloc[j,i] = distances.iloc[i,j]
        return distances

    # We need to implement k-medoids ourselves to accommodate all distance metrics
    def k_medoids_over_instances(self, dataset, cols, k, distance_metric, max_iters, n_inits=5, p=1):
        # If we set it to default we use the pyclust package...
        temp_dataset = dataset[cols]
        if distance_metric == 'default':
            km = pyclust.KMedoids(n_clusters=k, n_trials=n_inits)
            km.fit(temp_dataset.values)
            cluster_assignment = km.labels_

        else:
            print("It workds")
            self.p = p
            cluster_assignment = []
            best_silhouette = -1

            # Compute all distances
            D = self.compute_distance_matrix_instances(temp_dataset, distance_metric)

            for it in range(0, n_inits):
                # First select k random points as centers:
                centers = random.sample(range(0, len(dataset.index)), k)
                prev_centers = []
                points_to_cluster = []

                n_iter = 0
                while (n_iter < max_iters) and not (centers == prev_centers):
                    n_iter += 1
                    prev_centers = centers
                    # Assign points to clusters.
                    points_to_centroid = D[centers].idxmin(axis=1)

                    new_centers = []
                    for i in range(0, k):
                    # And find the new center that minimized the sum of the differences.
                      
                        best_center = D.loc[points_to_centroid == centers[i]].sum().idxmin(axis=1)
                        new_centers.append(best_center)
                    centers = new_centers

                # Convert centroids to cluster numbers:

                points_to_centroid = D[centers].idxmin(axis=1)
                current_cluster_assignment = []
                for i in range(0, len(dataset.index)):
                    current_cluster_assignment.append(centers.index(points_to_centroid.iloc[i]))

                silhouette_avg = silhouette_score(temp_dataset, np.array(current_cluster_assignment))
                if silhouette_avg > best_silhouette:
                    cluster_assignment = current_cluster_assignment
                    best_silhouette = silhouette_avg

        # And add the clusters and silhouette scores to the dataset.
        dataset['cluster'] = cluster_assignment
        silhouette_avg = silhouette_score(temp_dataset, np.array(cluster_assignment))
        silhouette_per_inst = silhouette_samples(temp_dataset, np.array(cluster_assignment))
        dataset['silhouette'] = silhouette_per_inst

        return dataset

    # For k-medoids we use all possible distance metrics between datasets as well. For this we
    # again need to define a distance matrix between the datasets.
    def compute_distance_matrix_datasets(self, datasets, distance_metric):
        distances = pd.DataFrame(index=range(0, len(datasets)), columns=range(0, len(datasets)))
        DMNoOrdering = PersonDistanceMetricsNoOrdering()
        DMOrdering = PersonDistanceMetricsOrdering()

        # And compute the distances for each pair. Note that we assume the distances to be symmetric.
        for i in range(0, len(datasets)):
            for j in range(i, len(datasets)):
                if distance_metric == self.abstraction_p:
                    distances.iloc[i,j] = DMNoOrdering.p_distance(datasets[i], datasets[j])
                elif distance_metric == self.abstraction_euclidean:
                    distances.iloc[i,j] = DMOrdering.euclidean_distance(datasets[i], datasets[j])
                elif distance_metric == self.abstraction_lag:
                    distances.iloc[i,j] = DMOrdering.lag_correlation(datasets[i], datasets[j], self.max_lag)
                elif distance_metric == self.abstraction_dtw:
                    distances.iloc[i,j] = DMOrdering.dynamic_time_warping(datasets[i], datasets[j])
                distances.iloc[j,i] = distances.iloc[i,j]
        return distances

    # Note: distance metric only important in combination with certain abstraction methods as we allow for more
    # in k-medoids.
    def k_medoids_over_datasets(self, datasets, cols, k, abstraction_method, distance_metric, max_iters, n_inits=5, p=1, max_lag=5):
        self.p = p
        self.max_lag = max_lag

        # If we compare datasets by flattening them, we can simply flatten the dataset and apply the instance based
        # variant.
        if abstraction_method in [self.abstraction_mean, self.abstraction_normal]:
            # Convert the datasets to instances
            temp_dataset = self.aggregate_datasets(datasets, cols, abstraction_method)

            # And simply apply the instance based algorithm in case of
            return self.k_medoids_over_instances(temp_dataset, temp_dataset.columns, k, distance_metric, max_iters, n_inits=n_inits, p=p)

        # For the case over datasets we do not have a quality metric, therefore we just look at a single initialization for now (!)

        # First select k random points as centers:
        centers = random.sample(range(0, len(datasets)), k)
        prev_centers = []
        points_to_cluster = []
        # Compute all distances
        D = self.compute_distance_matrix_datasets(datasets, abstraction_method)

        n_iter = 0
        while (n_iter < max_iters) and not (centers == prev_centers):
            n_iter += 1
            prev_centers = centers
            # Assign points to clusters.
            points_to_centroid = D[centers].idxmin(axis=1)

            new_centers = []
            for i in range(0, k):
                # And find the new center that minimized the sum of the differences.
                best_center = D.loc[points_to_centroid == centers[i], points_to_centroid == centers[i]].sum().idxmin(axis=1)
                new_centers.append(best_center)
            centers = new_centers

        # Convert centroids to cluster numbers:

        points_to_centroid = D[centers].idxmin(axis=1)
        cluster_assignment = []
        for i in range(0, len(datasets)):
            cluster_assignment.append(centers.index(points_to_centroid.iloc[i,:]))

        dataset = pd.DataFrame(index=range(0, len(datasets)))
        dataset['cluster'] = cluster_assignment

        # Silhouette cannot be used here as it used a distance between instances, not datasets.

        return dataset

# In this class, we do not implement the Gover distance between instance, all others are included.
# Furthermore, we only implement the agglomerative approach.
class HierarchicalClustering:
    link = None

    def agglomerative_over_instances(self, dataset, cols, max_clusters, distance_metric, use_prev_linkage=False,
                                     link_function='single'):
        # --- MODIFICATION START ---
        # 1. Create a clean subset of the data for clustering
        #    This is crucial to ensure 'linkage' or 'AgglomerativeClustering' doesn't receive NaNs.
        #    We then capture the index of these *cleaned* rows.
        temp_dataset_for_clustering = dataset[cols].dropna()
        original_indices_of_clustered_data = temp_dataset_for_clustering.index

        # Handle cases where there's no data or not enough data after dropping NaNs
        if temp_dataset_for_clustering.empty:
            print(
                f"Warning: No valid data points for agglomerative clustering after dropping NaNs in columns: {cols}. Returning original dataset with NaNs for cluster/silhouette.")
            dataset['cluster'] = np.nan
            dataset['silhouette'] = np.nan
            return dataset, None

        # Also check if enough samples for silhouette/clustering (n_clusters > 1 and < num_samples)
        if max_clusters <= 1 or max_clusters >= len(temp_dataset_for_clustering):
            print(
                f"Warning: max_clusters={max_clusters} is invalid for agglomerative clustering with {len(temp_dataset_for_clustering)} samples. k must be > 1 and < number of samples. Returning original dataset with NaNs for cluster/silhouette.")
            dataset['cluster'] = np.nan
            dataset['silhouette'] = np.nan
            return dataset, None

        # --- MODIFICATION END ---

        df_nh = NonHierarchicalClustering()  # Reference to your NonHierarchicalClustering class

        if (not use_prev_linkage) or (self.link is None):
            # Perform the clustering process according to the specified distance metric.
            if distance_metric == df_nh.manhattan:
                # Use temp_dataset_for_clustering.values here
                self.link = linkage(temp_dataset_for_clustering.values, method=link_function, metric='cityblock')
            else:
                # Use temp_dataset_for_clustering.values here
                self.link = linkage(temp_dataset_for_clustering.values, method=link_function, metric='euclidean')

        # And assign the clusters given the set maximum.
        cluster_assignment_raw = fcluster(self.link, max_clusters, criterion='maxclust')

        # --- MODIFICATION START ---
        # Map cluster_assignment back to the original dataset's index
        # Create a Series using the cluster assignments and the *original indices of the clustered data*
        cluster_assignment_series = pd.Series(cluster_assignment_raw, index=original_indices_of_clustered_data)
        # Reindex this Series to the full original dataset's index. This will fill NaN for rows
        # that were not included in clustering (e.g., due to NaNs or prior filtering).
        dataset['cluster'] = cluster_assignment_series.reindex(dataset.index)

        # Compute the silhouette score and map it back
        # Use temp_dataset_for_clustering and cluster_assignment_raw for silhouette calculation
        silhouette_per_inst_raw = silhouette_samples(temp_dataset_for_clustering, np.array(cluster_assignment_raw))
        silhouette_per_inst_series = pd.Series(silhouette_per_inst_raw, index=original_indices_of_clustered_data)
        dataset['silhouette'] = silhouette_per_inst_series.reindex(dataset.index)
        # --- MODIFICATION END ---

        # The original code also calculated silhouette_avg, which is not assigned to dataset.
        silhouette_avg = silhouette_score(temp_dataset_for_clustering,
                                          np.array(cluster_assignment_raw))  # Use cleaned data for calculation

        return dataset, self.link

    # Perform agglomerative clustering over the datasets by flattening them into a single dataset.
    def agglomerative_over_datasets(self, datasets, cols, max_clusters, abstraction_method, distance_metric,
                                    use_prev_linkage=False, link_function='single'):
        # Convert the datasets to instances
        df_nh = NonHierarchicalClustering()
        temp_dataset = df_nh.aggregate_datasets(datasets, cols, abstraction_method)

        # And simply apply the instance based algorithm...
        return self.agglomerative_over_instances(temp_dataset, temp_dataset.columns, max_clusters, distance_metric,
                                                 use_prev_linkage=use_prev_linkage, link_function=link_function)
