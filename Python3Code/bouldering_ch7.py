##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
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
from Chapter7.LearningAlgorithms import RegressionAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.Evaluation import RegressionEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from Chapter7.FeatureSelection import FeatureSelectionRegression
from util import util
from util.VisualizeDataset import VisualizeDataset

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./intermediate_datafiles_bouldering/') # Adjusted data path
EXPORT_TREE_BASE_PATH = Path('./figures/bouldering_ch7_classification/') # Base path for saving trees

# Next, we declare the parameters we'll use in the algorithms.
N_FORWARD_SELECTION = 50

# Get all chapter 5 result files as input
input_files = list(DATA_PATH.glob('chapter5_result_*.csv'))

if not input_files:
    print("No Chapter 5 result files found. Please run bouldering_ch5.py with --mode final first.")
    exit() # Exit if no files found

# Initialize learners, evaluators, and feature selectors once outside the loop
prepare = PrepareDatasetForLearning()
fs = FeatureSelectionClassification()
learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()

print('Preprocessing took', time.time()-start, 'seconds.')

for input_file_path in input_files:
    print(f"\n--- Processing file: {input_file_path.name} ---")
    try:
        dataset = pd.read_csv(input_file_path, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print(f'File not found: {input_file_path.name}. Skipping.')
        continue # Skip to the next file

    # Initialize a visualization object for this dataset.
    DataViz = VisualizeDataset(__file__)
    DataViz.plot_number = 1 # Reset plot number for each new file

    # Extract relevant part of the filename for use as plot prefix
    file_base_identifier = input_file_path.stem.replace('chapter5_result_', '')

    # Create a dynamic export path for decision trees for this specific file
    current_export_tree_path = EXPORT_TREE_BASE_PATH / file_base_identifier
    current_export_tree_path.mkdir(exist_ok=True, parents=True) # Ensure directory exists

    # Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.

    # We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
    # for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
    # cases where we do not know the label.

    # Ensure 'label' column exists before splitting, as some files might not have it if no labels were present in ch2
    if 'label' not in dataset.columns:
        print(f"Skipping {input_file_path.name}: 'label' column not found.")
        continue

    train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.7, filter=True, temporal=False)

    print('Training set length is: ', len(train_X.index))
    print('Test set length is: ', len(test_X.index))

    # Select subsets of the features that we will consider:
    # Adapted to Bouldering specific column names and patterns
    basic_features = ['acc_X (m/s^2)','acc_Y (m/s^2)','acc_Z (m/s^2)','gyr_X (rad/s)','gyr_Y (rad/s)','gyr_Z (rad/s)',
                      'mag_X (µT)','mag_Y (µT)','mag_Z (µT)','loc_Height (m)','loc_Velocity (m/s)']

    # Filter out columns that might not exist in the current dataset (e.g., if PCA or time/freq was skipped)
    basic_features = [f for f in basic_features if f in dataset.columns]


    # Dynamically find PCA features and ensure they are numeric
    pca_features = [name for name in dataset.columns if ('pca_' in name and pd.api.types.is_numeric_dtype(dataset[name]))]
    # Dynamically find temporal features
    time_features = [name for name in dataset.columns if '_temp_' in name and pd.api.types.is_numeric_dtype(dataset[name])]
    # Dynamically find frequency features
    freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name)) and pd.api.types.is_numeric_dtype(dataset[name])]

    cluster_features = ['cluster']
    cluster_features = [f for f in cluster_features if f in dataset.columns and pd.api.types.is_numeric_dtype(dataset[f])]


    print(f'#basic features: {len(basic_features)}')
    print(f'#PCA features: {len(pca_features)}')
    print(f'#time features: {len(time_features)}')
    print(f'#frequency features: {len(freq_features)}')
    print(f'#cluster features: {len(cluster_features)}')

    # Ensure all feature sets are created from existing and numeric columns
    features_after_chapter_3 = list(set().union(basic_features, pca_features))
    features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
    features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features, cluster_features))

    # Remove any non-numeric or boolean columns that might have slipped through
    all_available_numeric_features = [c for c in dataset.columns if pd.api.types.is_numeric_dtype(dataset[c]) and not pd.api.types.is_bool_dtype(dataset[c]) and 'label' not in c]

    # Filter feature sets against actual available numeric columns
    features_after_chapter_3 = [f for f in features_after_chapter_3 if f in all_available_numeric_features]
    features_after_chapter_4 = [f for f in features_after_chapter_4 if f in all_available_numeric_features]
    features_after_chapter_5 = [f for f in features_after_chapter_5 if f in all_available_numeric_features]


    # # First, let us consider the performance over a selection of features:
    # Ensure there are enough features for forward selection
    if len(features_after_chapter_5) == 0:
        print(f"Skipping feature selection for {input_file_path.name}: No valid features available after Chapter 5.")
        # Create dummy variables to prevent errors in subsequent steps if needed
        selected_features_for_this_file = []
        possible_feature_sets_for_this_file = []
        feature_names_for_this_file = []
    else:
        # Limit N_FORWARD_SELECTION to the actual number of available features
        num_features_to_select = min(N_FORWARD_SELECTION, len(features_after_chapter_5))
        if num_features_to_select > 0:
            features, ordered_features, ordered_scores = fs.forward_selection(num_features_to_select,
                                                                          train_X[features_after_chapter_5],
                                                                          test_X[features_after_chapter_5],
                                                                          train_y,
                                                                          test_y,
                                                                          gridsearch=False)
            # Make selected_features dynamic based on forward selection
            selected_features_for_this_file = features # Use the features actually selected by forward selection
            DataViz.plot_xy(x=[range(1, num_features_to_select + 1)], y=[ordered_scores],
                            xlabel='number of features', ylabel='accuracy',
                            prefix=f"{file_base_identifier}_forward_selection") # Unique prefix
        else:
            print(f"Not enough features for forward selection in {input_file_path.name}.")
            selected_features_for_this_file = [] # No features selected if count is 0

    # Ensure selected_features_for_this_file is not empty before proceeding to use it
    if not selected_features_for_this_file:
        # Fallback: if forward selection yielded no features, use a minimal set or skip.
        # For bouldering, basic_features are usually available.
        selected_features_for_this_file = [f for f in basic_features if f in train_X.columns]
        if not selected_features_for_this_file:
            print(f"Skipping {input_file_path.name}: No basic features available for analysis.")
            continue # Skip to the next file if no features can be used


    # # # Let us first study the impact of regularization and model complexity: does regularization prevent overfitting?

    reg_parameters = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    performance_training = []
    performance_test = []
    ## Due to runtime constraints we run the experiment 3 times, yet if you want even more robust data one should increase the repetitions.
    N_REPEATS_NN = 3


    for reg_param in reg_parameters:
        performance_tr = 0
        performance_te = 0
        # Ensure selected_features_for_this_file are in train_X and test_X
        current_train_X_nn = train_X[selected_features_for_this_file]
        current_test_X_nn = test_X[selected_features_for_this_file]

        if current_train_X_nn.empty or current_test_X_nn.empty:
            print(f"Skipping NN regularization for {input_file_path.name}: Empty feature set after selection.")
            performance_training.append(np.nan)
            performance_test.append(np.nan)
            continue


        for i in range(0, N_REPEATS_NN):

            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
                current_train_X_nn, train_y,
                current_test_X_nn, hidden_layer_sizes=(250, ), alpha=reg_param, max_iter=500,
                gridsearch=False
            )

            performance_tr += eval.accuracy(train_y, class_train_y)
            performance_te += eval.accuracy(test_y, class_test_y)
        performance_training.append(performance_tr/N_REPEATS_NN)
        performance_test.append(performance_te/N_REPEATS_NN)
    DataViz.plot_xy(x=[reg_parameters, reg_parameters], y=[performance_training, performance_test], method='semilogx',
                    xlabel='regularization parameter value', ylabel='accuracy', ylim=[0.95, 1.01],
                    names=['training', 'test'], line_styles=['r-', 'b:'],
                    prefix=f"{file_base_identifier}_nn_regularization") # Unique prefix

    #Second, let us consider the influence of certain parameter settings for the tree model. (very related to the
    #regularization) and study the impact on performance.

    leaf_settings = [1,2,5,10]
    performance_training = []
    performance_test = []

    # Ensure selected_features_for_this_file are in train_X and test_X
    current_train_X_dt = train_X[selected_features_for_this_file]
    current_test_X_dt = test_X[selected_features_for_this_file]

    if current_train_X_dt.empty or current_test_X_dt.empty:
        print(f"Skipping DT regularization for {input_file_path.name}: Empty feature set after selection.")
        # Skip this section or fill with NaNs
    else:
        for no_points_leaf in leaf_settings:

            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
                current_train_X_dt, train_y, current_test_X_dt, min_samples_leaf=no_points_leaf,
                gridsearch=False, print_model_details=False)

            performance_training.append(eval.accuracy(train_y, class_train_y))
            performance_test.append(eval.accuracy(test_y, class_test_y))

        DataViz.plot_xy(x=[leaf_settings, leaf_settings], y=[performance_training, performance_test],
                        xlabel='minimum number of points per leaf', ylabel='accuracy',
                        names=['training', 'test'], line_styles=['r-', 'b:'],
                        prefix=f"{file_base_identifier}_dt_leaf_settings") # Unique prefix

    # So yes, it is important :) Therefore we perform grid searches over the most important parameters, and do so by means
    # of cross validation upon the training set.

    # Filter feature sets to ensure they are not empty and contain existing columns
    possible_feature_sets_filtered = []
    feature_names_filtered = []

    # Prepare possible_feature_sets to use the filtered bouldering features
    temp_possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features_for_this_file]
    temp_feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']

    for i, feature_set in enumerate(temp_possible_feature_sets):
        # Ensure the feature set only contains columns present in train_X and are numeric
        valid_feature_set = [f for f in feature_set if f in train_X.columns and pd.api.types.is_numeric_dtype(train_X[f])]
        if valid_feature_set:
            possible_feature_sets_filtered.append(valid_feature_set)
            feature_names_filtered.append(temp_feature_names[i])
        else:
            print(f"Warning: Feature set '{temp_feature_names[i]}' is empty or contains no valid numeric columns for {input_file_path.name}. Skipping this set.")


    N_KCV_REPEATS = 1 # Reduced for faster execution, original was 5. Adjust as needed.


    scores_over_all_algs = []

    for i in range(0, len(possible_feature_sets_filtered)):
        print(f"\n--- Running classifiers for feature set: {feature_names_filtered[i]} ---")
        selected_train_X = train_X[possible_feature_sets_filtered[i]]
        selected_test_X = test_X[possible_feature_sets_filtered[i]]

        if selected_train_X.empty or selected_test_X.empty:
            print(f"Skipping classifiers for {feature_names_filtered[i]}: Empty feature set.")
            scores_over_all_algs.append(util.create_table_row_performances_empty()) # Use a helper for empty scores
            continue

        performance_tr_nn = 0
        performance_tr_rf = 0
        performance_tr_svm = 0
        performance_te_nn = 0
        performance_te_rf = 0
        performance_te_svm = 0

        for repeat in range(0, N_KCV_REPEATS):
            print(f"Training NeuralNetwork run {repeat+1} / {N_KCV_REPEATS} for feature set: {feature_names_filtered[i]}...")
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
                selected_train_X, train_y, selected_test_X, gridsearch=True
            )
            performance_tr_nn += eval.accuracy(train_y, class_train_y)
            performance_te_nn += eval.accuracy(test_y, class_test_y)

            print(f"Training RandomForest run {repeat+1} / {N_KCV_REPEATS} for feature set: {feature_names_filtered[i]}...")
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
                selected_train_X, train_y, selected_test_X, gridsearch=True
            )
            performance_tr_rf += eval.accuracy(train_y, class_train_y)
            performance_te_rf += eval.accuracy(test_y, class_test_y)

            print(f"Training SVM run {repeat+1} / {N_KCV_REPEATS} for feature set: {feature_names_filtered[i]}...")
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.support_vector_machine_with_kernel(
                selected_train_X, train_y, selected_test_X, gridsearch=True
            )
            performance_tr_svm += eval.accuracy(train_y, class_train_y)
            performance_te_svm += eval.accuracy(test_y, class_test_y)


        overall_performance_tr_nn = performance_tr_nn/N_KCV_REPEATS
        overall_performance_te_nn = performance_te_nn/N_KCV_REPEATS
        overall_performance_tr_rf = performance_tr_rf/N_KCV_REPEATS
        overall_performance_te_rf = performance_te_rf/N_KCV_REPEATS
        overall_performance_tr_svm = performance_tr_svm/N_KCV_REPEATS
        overall_performance_te_svm = performance_te_svm/N_KCV_REPEATS

    #     #And we run our deterministic classifiers:
        print("Determenistic Classifiers:")

        print(f"Training Nearest Neighbor run 1 / 1, featureset: {feature_names_filtered[i]}")
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.k_nearest_neighbor(
            selected_train_X, train_y, selected_test_X, gridsearch=True
        )
        performance_tr_knn = eval.accuracy(train_y, class_train_y)
        performance_te_knn = eval.accuracy(test_y, class_test_y)

        print(f"Training Descision Tree run 1 / 1 featureset: {feature_names_filtered[i]}")
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
            selected_train_X, train_y, selected_test_X, gridsearch=True
        )

        performance_tr_dt = eval.accuracy(train_y, class_train_y)
        performance_te_dt = eval.accuracy(test_y, class_test_y)

        print(f"Training Naive Bayes run 1/1 featureset: {feature_names_filtered[i]}")
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(
            selected_train_X, train_y, selected_test_X
        )

        performance_tr_nb = eval.accuracy(train_y, class_train_y)
        performance_te_nb = eval.accuracy(test_y, class_test_y)

        scores_with_sd = util.print_table_row_performances(feature_names_filtered[i], len(selected_train_X.index), len(selected_test_X.index), [
                                                                                                    (overall_performance_tr_nn, overall_performance_te_nn),
                                                                                                    (overall_performance_tr_rf, overall_performance_te_rf),
                                                                                                    (overall_performance_tr_svm, overall_performance_te_svm),
                                                                                                    (performance_tr_knn, performance_te_knn),
                                                                                                    (performance_tr_dt, performance_te_dt),
                                                                                                    (performance_tr_nb, performance_te_nb)])
        scores_over_all_algs.append(scores_with_sd)

    # Plot performances for current dataset
    if scores_over_all_algs:
        DataViz.plot_performances_classification(['NN', 'RF','SVM', 'KNN', 'DT', 'NB'], feature_names_filtered, scores_over_all_algs,
                                                 prefix=f"{file_base_identifier}_overall_performance") # Unique prefix


    # # And we study two promising ones in more detail. First, let us consider the decision tree, which works best with the
    # # selected features.
    if selected_features_for_this_file: # Only run if features are available
        print(f"\n--- Detailed analysis for {input_file_path.name} ---")
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
            train_X[selected_features_for_this_file], train_y, test_X[selected_features_for_this_file],
            gridsearch=True, print_model_details=True, export_tree_path=current_export_tree_path)

        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
            train_X[selected_features_for_this_file], train_y, test_X[selected_features_for_this_file],
            gridsearch=True, print_model_details=True)

        test_cm = eval.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)

        DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False,
                                      prefix=f"{file_base_identifier}_confusion_matrix") # Unique prefix
    else:
        print(f"Skipping detailed analysis for {input_file_path.name}: No selected features.")


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final', # No longer used to control main logic flow
                        help="This script processes all Chapter 5 files. Mode argument is deprecated for this script.",
                        choices=['final']) # Restrict to final as main modes are now loops

    FLAGS, unparsed = parser.parse_known_args()

    # The main function now processes all files automatically, so the mode argument is less central.
    # We keep it for compatibility with original calls, but it doesn't branch logic within main() as before.
    main()