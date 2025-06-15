##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7 - Adapted for Bouldering Dataset              #
#                                                            #
##############################################################

import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time

start = time.time()

from sklearn.model_selection import train_test_split

from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from util import util
from util.VisualizeDataset import VisualizeDataset

DATA_PATH = Path('./intermediate_datafiles_bouldering/')
DATASET_FNAME = 'chapter5_result_participant1_combined.csv'
RESULT_FNAME = 'chapter7_classification_result_combined.csv'
EXPORT_TREE_PATH = Path('./figures/bouldering_ch7_combined/')

# Next, we declare the parameters we'll use in the algorithms.
N_FORWARD_SELECTION = 50

try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
except IOError as e:
    print(f'File not found: {DATA_PATH / DATASET_FNAME}')
    print('Try to run the previous scripts first to generate the combined dataset!')
    raise e

dataset.index = pd.to_datetime(dataset.index)

# Let us create our visualization class again.
DataViz = VisualizeDataset(__file__)

# We create a single column with the categorical attribute representing our class.
prepare = PrepareDatasetForLearning()

# The function will find 'labelEasy', 'labelMedium', etc., and create a 'class' column.
train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.7,
                                                                               filter=True, temporal=False)

print('Training set length is: ', len(train_X.index))
print('Test set length is: ', len(test_X.index))

basic_features = ['acc_X (m/s^2)', 'acc_Y (m/s^2)', 'acc_Z (m/s^2)',
                  'gyr_X (rad/s)', 'gyr_Y (rad/s)', 'gyr_Z (rad/s)',
                  'mag_X (µT)', 'mag_Y (µT)', 'mag_Z (µT)',
                  'loc_Height (m)', 'loc_Velocity (m/s)']
pca_features = ['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7']
time_features = [name for name in train_X.columns if '_temp_' in name]
freq_features = [name for name in train_X.columns if (('_freq' in name) or ('_pse' in name))]
cluster_features = ['cluster']

# Filter to only include columns that actually exist in the training data
basic_features = [f for f in basic_features if f in train_X.columns]
pca_features = [f for f in pca_features if f in train_X.columns]
cluster_features = [f for f in cluster_features if f in train_X.columns]

print('#basic features: ', len(basic_features))
print('#PCA features: ', len(pca_features))
print('#time features: ', len(time_features))
print('#frequency features: ', len(freq_features))
print('#cluster features: ', len(cluster_features))

features_after_chapter_3 = list(set(basic_features + pca_features))
features_after_chapter_4 = list(set(features_after_chapter_3 + time_features + freq_features))
features_after_chapter_5 = list(set(features_after_chapter_4 + cluster_features))
# ---------------------------------------------------------------------------------

# First, let us consider the performance over a selection of features:

print("\nRunning Forward Feature Selection...")
fs = FeatureSelectionClassification()

# This step can take a while. It will identify the best features from your dataset.
features, ordered_features, ordered_scores = fs.forward_selection(N_FORWARD_SELECTION,
                                                                  train_X[features_after_chapter_5],
                                                                  test_X[features_after_chapter_5],
                                                                  train_y,
                                                                  test_y,
                                                                  gridsearch=False)

DataViz.plot_xy(x=[range(1, N_FORWARD_SELECTION + 1)], y=[ordered_scores],
                xlabel='number of features', ylabel='accuracy')

selected_features = ordered_features[:10]
print(f"\nTop 10 selected features: {selected_features}")

# # Let us first study the impact of regularization and model complexity.

learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()

print("\nStudying NN regularization...")
reg_parameters = [0.0001, 0.001, 0.01, 0.1, 1, 10]
performance_training = []
performance_test = []
N_REPEATS_NN = 3

for reg_param in reg_parameters:
    performance_tr = 0
    performance_te = 0
    for i in range(0, N_REPEATS_NN):
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
            train_X, train_y,
            test_X, hidden_layer_sizes=(250,), alpha=reg_param, max_iter=500,
            gridsearch=False
        )
        performance_tr += eval.accuracy(train_y, class_train_y)
        performance_te += eval.accuracy(test_y, class_test_y)
    performance_training.append(performance_tr / N_REPEATS_NN)
    performance_test.append(performance_te / N_REPEATS_NN)

DataViz.plot_xy(x=[reg_parameters, reg_parameters], y=[performance_training, performance_test], method='semilogx',
                xlabel='regularization parameter value', ylabel='accuracy',
                names=['training', 'test'], line_styles=['r-', 'b:'])

print("\nStudying Decision Tree complexity...")
leaf_settings = [1, 2, 5, 10]
performance_training = []
performance_test = []

for no_points_leaf in leaf_settings:
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
        train_X[selected_features], train_y, test_X[selected_features], min_samples_leaf=no_points_leaf,
        gridsearch=False, print_model_details=False)

    performance_training.append(eval.accuracy(train_y, class_train_y))
    performance_test.append(eval.accuracy(test_y, class_test_y))

DataViz.plot_xy(x=[leaf_settings, leaf_settings], y=[performance_training, performance_test],
                xlabel='minimum number of points per leaf', ylabel='accuracy',
                names=['training', 'test'], line_styles=['r-', 'b:'])

# Perform grid searches over the most important parameters by means of cross validation.

possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5,
                         selected_features]
feature_names = ['Initial Set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected Features']
N_KCV_REPEATS = 5

print('\nPreprocessing took', time.time() - start, 'seconds.')
print("--- Starting main classification loop ---")

scores_over_all_algs = []

for i in range(0, len(possible_feature_sets)):
    # Make sure the feature set is not empty
    if not possible_feature_sets[i]:
        print(f"Skipping feature set '{feature_names[i]}' as it contains no features.")
        continue
    selected_train_X = train_X[possible_feature_sets[i]]
    selected_test_X = test_X[possible_feature_sets[i]]

    # First we run our non-deterministic classifiers a number of times to average their score.
    performance_tr_nn = 0
    performance_tr_rf = 0
    performance_tr_svm = 0
    performance_te_nn = 0
    performance_te_rf = 0
    performance_te_svm = 0

    for repeat in range(0, N_KCV_REPEATS):
        print(f"\nRun {repeat + 1}/{N_KCV_REPEATS} for feature set: '{feature_names[i]}'")

        print("Training NeuralNetwork...")
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
            selected_train_X, train_y, selected_test_X, gridsearch=True
        )
        performance_tr_nn += eval.accuracy(train_y, class_train_y)
        performance_te_nn += eval.accuracy(test_y, class_test_y)

        print("Training RandomForest...")
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
            selected_train_X, train_y, selected_test_X, gridsearch=True
        )
        performance_tr_rf += eval.accuracy(train_y, class_train_y)
        performance_te_rf += eval.accuracy(test_y, class_test_y)

        print("Training SVM...")
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.support_vector_machine_with_kernel(
            selected_train_X, train_y, selected_test_X, gridsearch=True
        )
        performance_tr_svm += eval.accuracy(train_y, class_train_y)
        performance_te_svm += eval.accuracy(test_y, class_test_y)

    overall_performance_tr_nn = performance_tr_nn / N_KCV_REPEATS
    overall_performance_te_nn = performance_te_nn / N_KCV_REPEATS
    overall_performance_tr_rf = performance_tr_rf / N_KCV_REPEATS
    overall_performance_te_rf = performance_te_rf / N_KCV_REPEATS
    overall_performance_tr_svm = performance_tr_svm / N_KCV_REPEATS
    overall_performance_te_svm = performance_te_svm / N_KCV_REPEATS

    # And we run our deterministic classifiers:
    print("\nTraining Deterministic Classifiers...")
    print(f"Featureset: {feature_names[i]}")

    print("Training K-Nearest Neighbor...")
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.k_nearest_neighbor(
        selected_train_X, train_y, selected_test_X, gridsearch=True
    )
    performance_tr_knn = eval.accuracy(train_y, class_train_y)
    performance_te_knn = eval.accuracy(test_y, class_test_y)

    print("Training Decision Tree...")
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
        selected_train_X, train_y, selected_test_X, gridsearch=True
    )
    performance_tr_dt = eval.accuracy(train_y, class_train_y)
    performance_te_dt = eval.accuracy(test_y, class_test_y)

    print("Training Naive Bayes...")
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(
        selected_train_X, train_y, selected_test_X
    )
    performance_tr_nb = eval.accuracy(train_y, class_train_y)
    performance_te_nb = eval.accuracy(test_y, class_test_y)

    scores_with_sd = util.print_table_row_performances(feature_names[i], len(selected_train_X.index),
                                                       len(selected_test_X.index), [
                                                           (overall_performance_tr_nn, overall_performance_te_nn),
                                                           (overall_performance_tr_rf, overall_performance_te_rf),
                                                           (overall_performance_tr_svm, overall_performance_te_svm),
                                                           (performance_tr_knn, performance_te_knn),
                                                           (performance_tr_dt, performance_te_dt),
                                                           (performance_tr_nb, performance_te_nb)])
    scores_over_all_algs.append(scores_with_sd)

DataViz.plot_performances_classification(['NN', 'RF', 'SVM', 'KNN', 'DT', 'NB'], feature_names, scores_over_all_algs)

# And we study two promising ones in more detail.
print("\nDetailed analysis of Decision Tree and Random Forest with selected features...")

# Make sure selected_features is not empty before proceeding
if not selected_features:
    print("No features were selected, skipping detailed analysis.")
else:
    print("\nDecision Tree Details:")
    class_train_y_dt, class_test_y_dt, class_train_prob_y_dt, class_test_prob_y_dt = learner.decision_tree(
        train_X[selected_features], train_y, test_X[selected_features],
        gridsearch=True,
        print_model_details=True, export_tree_path=EXPORT_TREE_PATH)
    print("\nRandom Forest Details:")
    class_train_y_rf, class_test_y_rf, class_train_prob_y_rf, class_test_prob_y_rf = learner.random_forest(
        train_X[selected_features], train_y, test_X[selected_features],
        gridsearch=True, print_model_details=True)

    # Using the Random Forest for the confusion matrix as it's often a strong performer
    test_cm = eval.confusion_matrix(test_y, class_test_y_rf, class_train_prob_y_rf.columns)

    DataViz.plot_confusion_matrix(test_cm, class_train_prob_y_rf.columns, normalize=False)