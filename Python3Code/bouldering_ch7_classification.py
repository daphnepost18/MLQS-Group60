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
import argparse # Needed for command-line arguments

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
DATA_PATH = Path('./intermediate_datafiles_bouldering/')
DATASET_FNAME = 'chapter5_result.csv'
RESULT_FNAME = 'chapter7_classification_result.csv'
EXPORT_TREE_BASE_PATH = Path('./figures/crowdsignals_ch7_classification/')


N_FORWARD_SELECTION = 50

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def process_single_bouldering_dataset(dataset, file_base_identifier, export_tree_base_path):

    # Original start time variable, now for individual dataset processing
    start_time_dataset_processing = time.time()

    # Let us create our visualization class again.
    DataViz = VisualizeDataset(__file__)
    DataViz.plot_number = 1 # Reset plot number for each new file processed

    # Create a dynamic export path for decision trees for this specific file
    current_export_tree_path = export_tree_base_path / file_base_identifier
    current_export_tree_path.mkdir(exist_ok=True, parents=True) # Ensure directory exists

    # Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.
    # We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
    # for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
    # cases where we do not know the label.

    prepare = PrepareDatasetForLearning()

    # Check for the presence of any of the new label columns (e.g., 'labelEasy', 'labelMedium', etc.).
    label_cols = [col for col in dataset.columns if col.startswith('label')]
    if not label_cols:
        print(f"Skipping processing for this dataset ('{file_base_identifier}'): No label columns found. Cannot perform classification.")
        return

    train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.7, filter=True, temporal=False)

    print('Training set length is: ', len(train_X.index))
    print('Test set length is: ', len(test_X.index))

    # Select subsets of the features that we will consider:
    # Adapted to Bouldering specific column names and patterns
    basic_features = ['acc_X (m/s^2)','acc_Y (m/s^2)','acc_Z (m/s^2)',
                      'gyr_X (rad/s)','gyr_Y (rad/s)','gyr_Z (rad/s)',
                      'mag_X (µT)','mag_Y (µT)','mag_Z (µT)',
                      'loc_Height (m)','loc_Velocity (m/s)'] # These are original names from Ch2

    # Filter feature lists to include only columns that exist in the *current* dataset
    # and are numeric (excluding boolean types which cause issues with math operations)
    all_numeric_cols_in_dataset = [c for c in dataset.columns
                                   if pd.api.types.is_numeric_dtype(dataset[c])
                                   and not pd.api.types.is_bool_dtype(dataset[c])
                                   and 'label' not in c]

    # Re-filter basic_features to ensure they exist in this specific dataset
    basic_features = [f for f in basic_features if f in all_numeric_cols_in_dataset]


    # Dynamically find PCA features (pca_X)
    pca_features = [name for name in all_numeric_cols_in_dataset if 'pca_' in name]
    # Dynamically find temporal features (ending with _temp_mean, _temp_std etc.)
    time_features = [name for name in all_numeric_cols_in_dataset if '_temp_' in name]
    # Dynamically find frequency features (containing _freq or _pse)
    freq_features = [name for name in all_numeric_cols_in_dataset if (('_freq' in name) or ('_pse' in name))]
    # Dynamically find cluster features (usually just 'cluster')
    cluster_features = [name for name in all_numeric_cols_in_dataset if name == 'cluster']


    print('#basic features: ', len(basic_features))
    print('#PCA features: ', len(pca_features))
    print('#time features: ', len(time_features))
    print('#frequency features: ', len(freq_features))
    print('#cluster features: ', len(cluster_features))

    # Construct the combined feature sets ensuring only numeric and non-boolean columns are used
    features_after_chapter_3 = list(set().union(basic_features, pca_features))
    features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
    features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features, cluster_features))

    # Ensure final feature sets only contain features present in the training/test splits
    features_after_chapter_3 = [f for f in features_after_chapter_3 if f in train_X.columns]
    features_after_chapter_4 = [f for f in features_after_chapter_4 if f in train_X.columns]
    features_after_chapter_5 = [f for f in features_after_chapter_5 if f in train_X.columns]


    # # First, let us consider the performance over a selection of features:

    fs = FeatureSelectionClassification()

    # Ensure there are enough features for forward selection
    if len(features_after_chapter_5) == 0:
        print(f"Skipping feature selection for '{file_base_identifier}': No valid features available after Chapter 5.")
        selected_features = [] # No features selected
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

            # Make selected_features dynamic based on forward selection for the current file
            selected_features = features
            DataViz.plot_xy(x=[range(1, num_features_to_select + 1)], y=[ordered_scores],
                            xlabel='number of features', ylabel='accuracy',
                            prefix=f"{file_base_identifier}_forward_selection") # Unique plot prefix
        else:
            print(f"Not enough features for forward selection in '{file_base_identifier}'. (Requires >0 features).")
            selected_features = [] # No features selected if count is 0

    # Ensure selected_features is not empty before proceeding
    if not selected_features:
        print(f"Skipping further analysis for '{file_base_identifier}': No selected features to train models.")
        return # Exit this function for the current dataset

    # # # Let us first study the impact of regularization and model complexity: does regularization prevent overfitting?

    learner = ClassificationAlgorithms()
    eval = ClassificationEvaluation()
    start_time_nn_reg = time.time() # Timer for this section

    reg_parameters = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    performance_training = []
    performance_test = []

    N_REPEATS_NN = 3 # Due to runtime constraints we run the experiment 3 times, yet if you want even more robust data one should increase the repetitions.

    # Filter train_X/test_X for selected features.
    current_train_X_nn = train_X[selected_features]
    current_test_X_nn = test_X[selected_features]

    if current_train_X_nn.empty or current_test_X_nn.empty:
        print(f"Skipping NN regularization for '{file_base_identifier}': Empty feature set after selection.")
        # Fill with NaNs to keep lists consistent for plotting, if plot needs to be created
        performance_training = [np.nan] * len(reg_parameters)
        performance_test = [np.nan] * len(reg_parameters)
    else:
        for reg_param in reg_parameters:
            performance_tr = 0
            performance_te = 0
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
                    prefix=f"{file_base_identifier}_nn_regularization") # Unique plot prefix
    print(f"NN regularization took {time.time() - start_time_nn_reg} seconds.")


    #Second, let us consider the influence of certain parameter settings for the tree model. (very related to the
    #regularization) and study the impact on performance.

    leaf_settings = [1,2,5,10]
    performance_training = []
    performance_test = []

    # Filter train_X/test_X for selected features.
    current_train_X_dt = train_X[selected_features]
    current_test_X_dt = test_X[selected_features]

    if current_train_X_dt.empty or current_test_X_dt.empty:
        print(f"Skipping DT regularization for '{file_base_identifier}': Empty feature set after selection.")
        # Fill with NaNs to keep lists consistent for plotting, if plot needs to be created
        performance_training = [np.nan] * len(leaf_settings)
        performance_test = [np.nan] * len(leaf_settings)
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
                        prefix=f"{file_base_identifier}_dt_leaf_settings") # Unique plot prefix


    # So yes, it is important :) Therefore we perform grid searches over the most important parameters, and do so by means
    # of cross validation upon the training set.

    # Prepare possible_feature_sets using the filtered bouldering features
    possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
    feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']

    # Filter out empty feature sets from possible_feature_sets
    # and re-align feature_names
    final_possible_feature_sets = []
    final_feature_names = []
    for i, f_set in enumerate(possible_feature_sets):
        # Ensure the feature set is not empty and all features are in train_X (which implies they are numeric)
        cleaned_f_set = [f for f in f_set if f in train_X.columns]
        if cleaned_f_set:
            final_possible_feature_sets.append(cleaned_f_set)
            final_feature_names.append(feature_names[i])
        else:
            print(f"Warning: Feature set '{feature_names[i]}' is empty or contains no valid columns in train_X for '{file_base_identifier}'. Skipping.")


    N_KCV_REPEATS = 1 # Reduced for faster execution in loop, original was 5. Adjust as needed.

    scores_over_all_algs = []

    # --- MODIFICATION START ---
    # Helper to create a dummy empty score row consistent with util.print_table_row_performances output format
    def create_empty_score_row():
        # (overall_performance_tr_nn, overall_performance_te_nn), etc.
        # Each element is (mean, std) or just a single value
        # For an empty set, we can represent with (np.nan, np.nan) or similar
        return (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan), \
               (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan)
    # --- MODIFICATION END ---

    for i in range(0, len(final_possible_feature_sets)):
        print(f"\n--- Running classifiers for feature set: {final_feature_names[i]} ---")
        selected_train_X = train_X[final_possible_feature_sets[i]]
        selected_test_X = test_X[final_possible_feature_sets[i]]

        if selected_train_X.empty or selected_test_X.empty:
            print(f"Skipping classifiers for {final_feature_names[i]}: Empty feature set.")
            scores_over_all_algs.append(create_empty_score_row()) # Use the helper for empty scores
            continue


        performance_tr_nn = 0
        performance_tr_rf = 0
        performance_tr_svm = 0
        performance_te_nn = 0
        performance_te_rf = 0
        performance_te_svm = 0

        for repeat in range(0, N_KCV_REPEATS):
            print(f"Training NeuralNetwork run {repeat+1} / {N_KCV_REPEATS} for feature set: {final_feature_names[i]}...")
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
                selected_train_X, train_y, selected_test_X, gridsearch=True
            )
            performance_tr_nn += eval.accuracy(train_y, class_train_y)
            performance_te_nn += eval.accuracy(test_y, class_test_y)

            print(f"Training RandomForest run {repeat+1} / {N_KCV_REPEATS} for feature set: {final_feature_names[i]}...")
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
                selected_train_X, train_y, selected_test_X, gridsearch=True
            )
            performance_tr_rf += eval.accuracy(train_y, class_train_y)
            performance_te_rf += eval.accuracy(test_y, class_test_y)

            print(f"Training SVM run {repeat+1} / {N_KCV_REPEATS} for feature set: {final_feature_names[i]}...")
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

        print(f"Training Nearest Neighbor run 1 / 1, featureset: {final_feature_names[i]}")
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.k_nearest_neighbor(
            selected_train_X, train_y, selected_test_X, gridsearch=True
        )
        performance_tr_knn = eval.accuracy(train_y, class_train_y)
        performance_te_knn = eval.accuracy(test_y, class_test_y)

        print(f"Training Descision Tree run 1 / 1 featureset: {final_feature_names[i]}")
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
            selected_train_X, train_y, selected_test_X, gridsearch=True
        )

        performance_tr_dt = eval.accuracy(train_y, class_train_y)
        performance_te_dt = eval.accuracy(test_y, class_test_y)

        print(f"Training Naive Bayes run 1/1 featureset: {final_feature_names[i]}")
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(
            selected_train_X, train_y, selected_test_X
        )

        performance_tr_nb = eval.accuracy(train_y, class_train_y)
        performance_te_nb = eval.accuracy(test_y, class_test_y)

        scores_with_sd = util.print_table_row_performances(final_feature_names[i], len(selected_train_X.index), len(selected_test_X.index), [
                                                                                                    (overall_performance_tr_nn, overall_performance_te_nn),
                                                                                                    (overall_performance_tr_rf, overall_performance_te_rf),
                                                                                                    (overall_performance_tr_svm, overall_performance_te_svm),
                                                                                                    (performance_tr_knn, performance_te_knn),
                                                                                                    (performance_tr_dt, performance_te_dt),
                                                                                                    (performance_tr_nb, performance_te_nb)])
        scores_over_all_algs.append(scores_with_sd)

    # Plot performances for current dataset
    if scores_over_all_algs and any(s is not None for s in scores_over_all_algs): # Check if any valid scores were added
        valid_algs = ['NN', 'RF','SVM', 'KNN', 'DT', 'NB'] # These are hardcoded in the original util function
        DataViz.plot_performances_classification(valid_algs, final_feature_names, scores_over_all_algs,
                                                 prefix=f"{file_base_identifier}_overall_performance") # Unique plot prefix
    else:
        print(f"Skipping overall performance plot for '{file_base_identifier}': No valid scores to plot.")


    # # And we study two promising ones in more detail. First, let us consider the decision tree, which works best with the
    # # selected features.
    if selected_features: # Only run if features are available (using the selected_features from forward selection)
        print(f"\n--- Detailed analysis for '{file_base_identifier}' ---")
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
            train_X[selected_features], train_y, test_X[selected_features],
            gridsearch=True, print_model_details=True, export_tree_path=current_export_tree_path)

        # Note: Original code uses decision_tree output for confusion matrix, then runs random_forest.
        # This means the confusion matrix is for the decision tree.
        class_train_y_rf, class_test_y_rf, class_train_prob_y_rf, class_test_prob_y_rf = learner.random_forest(
            train_X[selected_features], train_y, test_X[selected_features],
            gridsearch=True, print_model_details=True)

        # Ensure class_train_prob_y has columns before passing to confusion_matrix
        if hasattr(class_train_prob_y, 'columns') and len(class_train_prob_y.columns) > 0:
            test_cm = eval.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)
            DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False,
                                          prefix=f"{file_base_identifier}_confusion_matrix") # Unique plot prefix
        else:
            print(f"Skipping confusion matrix plot for '{file_base_identifier}': No class probability columns found (likely due to single class in data or model issues).")

    else:
        print(f"Skipping detailed analysis for '{file_base_identifier}': No selected features to train detailed models.")

    print(f"--- Processing for '{file_base_identifier}' took {time.time() - start_time_dataset_processing} seconds ---")


# Main execution block
def main(): # Re-defined main, as it was missing or incomplete in previous outputs.
    print_flags()

    start_time_overall = time.time()

    # Get all chapter 5 result files as input
    input_files = list(DATA_PATH.glob('chapter5_result_*.csv'))

    if not input_files:
        print("No Chapter 5 result files found. Please run bouldering_ch5.py with --mode final first.")
        return

    # Loop through each input file and process it
    for input_file_path in input_files:
        try:
            # Load dataset within the loop
            dataset = pd.read_csv(input_file_path, index_col=0)
            dataset.index = pd.to_datetime(dataset.index)

            # Call the helper function to process this single dataset
            process_single_bouldering_dataset(dataset, input_file_path.stem.replace('chapter5_result_', ''), EXPORT_TREE_BASE_PATH)

        except Exception as e:
            print(f"Error processing {input_file_path.name}: {e}")
            continue

    print(f"\n--- Total script execution time: {time.time() - start_time_overall} seconds ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final (deprecated, script processes all files) \
                        'final' is used for the next chapter", # Original help text, adapted for looping
                        choices=['final']) # Choices restricted as main logic now loops through files

    FLAGS, unparsed = parser.parse_known_args()

    # Call the main function to start the script execution
    main()