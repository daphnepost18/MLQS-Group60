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
import argparse

from sklearn.model_selection import train_test_split

from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from util import util
from util.VisualizeDataset import VisualizeDataset


def main():
    start_time = time.time()
    DATA_PATH = Path('./intermediate_datafiles_bouldering/')

    USE_ALL_FILES = (FLAGS.source == 'all')

    if USE_ALL_FILES:
        print("Mode: Processing all individual chapter 5 final result files...")
        all_chapter5_files = list(DATA_PATH.glob('chapter5_result_*.csv'))
        input_files = [f for f in all_chapter5_files if 'combined' not in f.name]
    else:
        print("Mode: Processing only the combined chapter 5 final result file...")
        combined_file = DATA_PATH / 'chapter5_result_combined.csv'
        if combined_file.exists():
            input_files = [combined_file]
        else:
            input_files = []

    if not input_files:
        if USE_ALL_FILES:
            print(
                "No individual Chapter 5 final result files found. Please run bouldering_ch5.py with '--mode final' first.")
        else:
            print(
                f"Combined file not found at '{combined_file}'. Please run bouldering_ch5.py with '--mode final' and '--source combined'.")
        return

    N_FORWARD_SELECTION = 30

    for input_file_path in input_files:
        print(f"\n\n======================================================")
        print(f"--- Processing file: {input_file_path.name} ---")
        print(f"======================================================\n")

        try:
            dataset = pd.read_csv(input_file_path, index_col=0)
            dataset.index = pd.to_datetime(dataset.index)
        except IOError as e:
            print(f'File not found: {input_file_path.name}. Skipping.')
            continue

        dataset_name = input_file_path.stem.replace('chapter5_result_', '')
        EXPORT_TREE_PATH = Path(f'./figures/bouldering_ch7_classification/')
        EXPORT_TREE_PATH.mkdir(exist_ok=True, parents=True)

        DataViz = VisualizeDataset(__file__)
        prepare = PrepareDatasetForLearning()

        train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.8,
                                                                                       filter=True, temporal=False)

        print('Training set length is: ', len(train_X.index))
        print('Test set length is: ', len(test_X.index))

        basic_features = [c for c in train_X.columns if
                          c.startswith('acc_') or c.startswith('gyr_') or c.startswith('mag_') or c.startswith('loc_')]
        pca_features = [c for c in train_X.columns if c.startswith('pca_')]
        time_features = [c for c in train_X.columns if '_temp_' in c]
        freq_features = [c for c in train_X.columns if '_freq' in c or '_pse' in c]
        cluster_features = [c for c in train_X.columns if c.startswith('cluster')]

        print(f'#basic features: {len(basic_features)}')
        print(f'#PCA features: {len(pca_features)}')
        print(f'#time features: {len(time_features)}')
        print(f'#frequency features: {len(freq_features)}')
        print(f'#cluster features: {len(cluster_features)}')

        features_after_chapter_3 = list(set(basic_features + pca_features))
        features_after_chapter_4 = list(set(features_after_chapter_3 + time_features + freq_features))
        features_after_chapter_5 = list(set(features_after_chapter_4 + cluster_features))

        print("\nRunning Forward Feature Selection...")
        fs = FeatureSelectionClassification()

        features, ordered_features, ordered_scores = fs.forward_selection(
            N_FORWARD_SELECTION, train_X[features_after_chapter_5], test_X[features_after_chapter_5],
            train_y, test_y, gridsearch=False
        )

        DataViz.plot_xy(x=[range(1, N_FORWARD_SELECTION + 1)], y=[ordered_scores],
                        xlabel='number of features', ylabel='accuracy', dataset_name=dataset_name,
                        methodch3='f_selection')

        selected_features = ordered_features[:10]
        print(f"\nTop 10 selected features for {dataset_name}: {selected_features}")

        learner = ClassificationAlgorithms()
        eval = ClassificationEvaluation()

        print("\nStudying impact of regularization and model complexity...")

        reg_parameters = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        performance_training_nn = []
        performance_test_nn = []
        N_REPEATS_NN = 3

        for reg_param in reg_parameters:
            performance_tr = 0
            performance_te = 0
            for i in range(N_REPEATS_NN):
                class_train_y, class_test_y, _, _ = learner.feedforward_neural_network(
                    train_X, train_y,
                    test_X, hidden_layer_sizes=(250,), alpha=reg_param, max_iter=500,
                    gridsearch=False
                )
                performance_tr += eval.accuracy(train_y, class_train_y)
                performance_te += eval.accuracy(test_y, class_test_y)
            performance_training_nn.append(performance_tr / N_REPEATS_NN)
            performance_test_nn.append(performance_te / N_REPEATS_NN)

        DataViz.plot_xy(x=[reg_parameters, reg_parameters], y=[performance_training_nn, performance_test_nn],
                        method='semilogx', xlabel='regularization parameter value', ylabel='accuracy',
                        names=['training', 'test'], line_styles=['r-', 'b:'], dataset_name=dataset_name,
                        methodch3='regularization')

        leaf_settings = [1, 2, 5, 10]
        performance_training_dt = []
        performance_test_dt = []

        for no_points_leaf in leaf_settings:
            class_train_y, class_test_y, _, _ = learner.decision_tree(
                train_X[selected_features], train_y, test_X[selected_features], min_samples_leaf=no_points_leaf,
                gridsearch=False, print_model_details=False)
            performance_training_dt.append(eval.accuracy(train_y, class_train_y))
            performance_test_dt.append(eval.accuracy(test_y, class_test_y))

        DataViz.plot_xy(x=[leaf_settings, leaf_settings], y=[performance_training_dt, performance_test_dt],
                        xlabel='minimum number of points per leaf', ylabel='accuracy',
                        names=['training', 'test'], line_styles=['r-', 'b:'], dataset_name=dataset_name,
                        methodch3='leaf_settings')

        possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4,
                                 features_after_chapter_5, selected_features]
        feature_names = ['Initial Set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected Features']
        N_KCV_REPEATS = 5

        print(f"\n--- Starting main classification loop for {dataset_name} ---")
        scores_over_all_algs = []

        for i in range(len(possible_feature_sets)):
            current_features = possible_feature_sets[i]
            if not current_features or not all(f in train_X.columns for f in current_features):
                print(f"Skipping feature set '{feature_names[i]}' as it contains missing or no features.")
                scores_over_all_algs.append([(0, 0)] * 6)  # 6 algorithms
                continue

            selected_train_X = train_X[current_features]
            selected_test_X = test_X[current_features]

            performance_tr_nn, performance_te_nn = (0, 0)
            performance_tr_rf, performance_te_rf = (0, 0)
            performance_tr_svm, performance_te_svm = (0, 0)
            last_pred_nn, last_prob_nn = (None, None)
            last_pred_rf, last_prob_rf = (None, None)
            last_pred_svm, last_prob_svm = (None, None)

            for repeat in range(N_KCV_REPEATS):
                print(f"\nRun {repeat + 1}/{N_KCV_REPEATS} for feature set: '{feature_names[i]}'")
                print("Training NeuralNetwork...")
                c_train_y, c_test_y, c_train_p, _ = learner.feedforward_neural_network(selected_train_X, train_y,
                                                                                       selected_test_X, gridsearch=True)
                performance_tr_nn += eval.accuracy(train_y, c_train_y)
                performance_te_nn += eval.accuracy(test_y, c_test_y)
                if repeat == N_KCV_REPEATS - 1: last_pred_nn, last_prob_nn = c_test_y, c_train_p
                print("Training RandomForest...")
                c_train_y, c_test_y, c_train_p, _ = learner.random_forest(selected_train_X, train_y, selected_test_X,
                                                                          gridsearch=True)
                performance_tr_rf += eval.accuracy(train_y, c_train_y)
                performance_te_rf += eval.accuracy(test_y, c_test_y)
                if repeat == N_KCV_REPEATS - 1: last_pred_rf, last_prob_rf = c_test_y, c_train_p
                print("Training SVM...")
                c_train_y, c_test_y, c_train_p, _ = learner.support_vector_machine_with_kernel(selected_train_X,
                                                                                               train_y, selected_test_X,
                                                                                               gridsearch=True)
                performance_tr_svm += eval.accuracy(train_y, c_train_y)
                performance_te_svm += eval.accuracy(test_y, c_test_y)
                if repeat == N_KCV_REPEATS - 1: last_pred_svm, last_prob_svm = c_test_y, c_train_p

                # This section is modified to capture the trained models and plot feature importances.
                performance_tr_nn, performance_te_nn = (0, 0)
                performance_tr_rf, performance_te_rf = (0, 0)
                performance_tr_svm, performance_te_svm = (0, 0)
                last_pred_nn, last_prob_nn = (None, None)
                last_pred_rf, last_prob_rf = (None, None)
                last_pred_svm, last_prob_svm = (None, None)

                # NEW: variables to store the last trained models for feature importance plotting
                last_model_rf = None
                last_model_dt = None

                for repeat in range(N_KCV_REPEATS):
                    print(f"\nRun {repeat + 1}/{N_KCV_REPEATS} for feature set: '{feature_names[i]}'")
                    print("Training NeuralNetwork...")
                    c_train_y, c_test_y, c_train_p, _ = learner.feedforward_neural_network(selected_train_X, train_y,
                                                                                           selected_test_X,
                                                                                           gridsearch=True)
                    performance_tr_nn += eval.accuracy(train_y, c_train_y)
                    performance_te_nn += eval.accuracy(test_y, c_test_y)
                    if repeat == N_KCV_REPEATS - 1: last_pred_nn, last_prob_nn = c_test_y, c_train_p

                    print("Training RandomForest...")
                    # MODIFIED: Capture the returned model object
                    c_train_y, c_test_y, c_train_p, rf_model = learner.random_forest(selected_train_X, train_y,
                                                                                     selected_test_X,
                                                                                     gridsearch=True)
                    performance_tr_rf += eval.accuracy(train_y, c_train_y)
                    performance_te_rf += eval.accuracy(test_y, c_test_y)
                    # MODIFIED: Save the model from the last repeat
                    if repeat == N_KCV_REPEATS - 1:
                        last_pred_rf, last_prob_rf, last_model_rf = c_test_y, c_train_p, rf_model

                    print("Training SVM...")
                    c_train_y, c_test_y, c_train_p, _ = learner.support_vector_machine_with_kernel(selected_train_X,
                                                                                                   train_y,
                                                                                                   selected_test_X,
                                                                                                   gridsearch=True)
                    performance_tr_svm += eval.accuracy(train_y, c_train_y)
                    performance_te_svm += eval.accuracy(test_y, c_test_y)
                    if repeat == N_KCV_REPEATS - 1: last_pred_svm, last_prob_svm = c_test_y, c_train_p

                performance_tr_nn /= N_KCV_REPEATS
                performance_te_nn /= N_KCV_REPEATS
                performance_tr_rf /= N_KCV_REPEATS
                performance_te_rf /= N_KCV_REPEATS
                performance_tr_svm /= N_KCV_REPEATS
                performance_te_svm /= N_KCV_REPEATS

                print("\nGenerating confusion matrices for non-deterministic models...")
                cm_nn = eval.confusion_matrix(test_y, last_pred_nn, last_prob_nn.columns)
                DataViz.plot_confusion_matrix(cm_nn, last_prob_nn.columns, normalize=False, dataset_name=dataset_name,
                                              method=f'{feature_names[i]}_NN')
                cm_rf = eval.confusion_matrix(test_y, last_pred_rf, last_prob_rf.columns)
                DataViz.plot_confusion_matrix(cm_rf, last_prob_rf.columns, normalize=False, dataset_name=dataset_name,
                                              method=f'{feature_names[i]}_RF')
                cm_svm = eval.confusion_matrix(test_y, last_pred_svm, last_prob_svm.columns)
                DataViz.plot_confusion_matrix(cm_svm, last_prob_svm.columns, normalize=False, dataset_name=dataset_name,
                                              method=f'{feature_names[i]}_SVM')

                print("\nTraining Deterministic Classifiers...")
                print(f"Featureset: {feature_names[i]}")
                print("Training K-Nearest Neighbor...")
                c_train_y_knn, c_test_y_knn, c_train_p_knn, _ = learner.k_nearest_neighbor(selected_train_X, train_y,
                                                                                           selected_test_X,
                                                                                           gridsearch=True)
                performance_tr_knn = eval.accuracy(train_y, c_train_y_knn)
                performance_te_knn = eval.accuracy(test_y, c_test_y_knn)
                cm_knn = eval.confusion_matrix(test_y, c_test_y_knn, c_train_p_knn.columns)
                DataViz.plot_confusion_matrix(cm_knn, c_train_p_knn.columns, normalize=False, dataset_name=dataset_name,
                                              method=f'{feature_names[i]}_KNN')

                print("Training Decision Tree...")
                # MODIFIED: Capture the returned model object
                c_train_y_dt, c_test_y_dt, c_train_p_dt, dt_model = learner.decision_tree(selected_train_X, train_y,
                                                                                          selected_test_X,
                                                                                          gridsearch=True)
                performance_tr_dt = eval.accuracy(train_y, c_train_y_dt)
                performance_te_dt = eval.accuracy(test_y, c_test_y_dt)
                last_model_dt = dt_model  # Save the decision tree model
                cm_dt = eval.confusion_matrix(test_y, c_test_y_dt, c_train_p_dt.columns)
                DataViz.plot_confusion_matrix(cm_dt, c_train_p_dt.columns, normalize=False, dataset_name=dataset_name,
                                              method=f'{feature_names[i]}_DT')

                print("Training Naive Bayes...")
                c_train_y_nb, c_test_y_nb, c_train_p_nb, _ = learner.naive_bayes(selected_train_X, train_y,
                                                                                 selected_test_X)
                performance_tr_nb = eval.accuracy(train_y, c_train_y_nb)
                performance_te_nb = eval.accuracy(test_y, c_test_y_nb)
                cm_nb = eval.confusion_matrix(test_y, c_test_y_nb, c_train_p_nb.columns)
                DataViz.plot_confusion_matrix(cm_nb, c_train_p_nb.columns, normalize=False, dataset_name=dataset_name,
                                              method=f'{feature_names[i]}_NB')

                if feature_names[i] == 'Selected Features':
                    if last_model_rf:
                        DataViz.plot_feature_importances(last_model_rf.feature_importances_, current_features,
                                                         dataset_name, 'RandomForest')
                    if last_model_dt:
                        DataViz.plot_feature_importances(last_model_dt.feature_importances_, current_features,
                                                         dataset_name, 'DecisionTree')

                scores_with_sd = util.print_table_row_performances(
                    feature_names[i], len(selected_train_X.index), len(selected_test_X.index),
                    [(performance_tr_nn, performance_te_nn), (performance_tr_rf, performance_te_rf),
                     (performance_tr_svm, performance_te_svm),
                     (performance_tr_knn, performance_te_knn), (performance_tr_dt, performance_te_dt),
                     (performance_tr_nb, performance_te_nb)]
                )
                scores_over_all_algs.append(scores_with_sd)

        DataViz.plot_performances_classification(['NN', 'RF', 'SVM', 'KNN', 'DT', 'NB'], feature_names,
                                                 scores_over_all_algs, dataset_name)

    print(f"\nTotal processing time: {time.time() - start_time} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='combined',
                        help="Specify source: 'all' for individual files, or 'combined' for the single combined file.",
                        choices=['all', 'combined'])
    FLAGS, unparsed = parser.parse_known_args()
    main()