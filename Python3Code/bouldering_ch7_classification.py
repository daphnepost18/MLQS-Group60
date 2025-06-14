import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from util.VisualizeDataset import VisualizeDataset
from util import util

def extract_difficulty_from_filename(filename):
    """Extracts 'Easy', 'Medium', or 'Hard' from a filename."""
    match = re.search(r'_(Easy|Medium|Hard)', filename)
    if match:
        return match.group(1)
    return None


def aggregate_session_features(df):
    """
    Aggregates time-series data from a session into a single feature row.
    """
    # Select only numeric columns for aggregation
    numeric_df = df.select_dtypes(include=np.number)

    # Exclude any label or class columns that might be present
    numeric_df = numeric_df.loc[:, ~numeric_df.columns.str.startswith('label')]
    if 'class' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['class'])

    # Perform aggregation. This results in a DataFrame.
    agg_df = numeric_df.agg(['mean', 'std', 'min', 'max'])

    # Check if aggregation result is empty
    if agg_df.empty:
        return pd.DataFrame()

    # Unstack the DataFrame to create a MultiIndex Series. This is the key step.
    s = agg_df.unstack()

    # Create new, flattened column names by joining the levels of the MultiIndex.
    s.index = s.index.map(lambda x: f"{x[0]}_{x[1]}")

    # Convert the final Series to a single-row DataFrame and return.
    return s.to_frame().T


def main():
    DATA_PATH = Path('./intermediate_datafiles_bouldering/')

    # --- 1. Select Training and Test Files ---

    all_files = list(DATA_PATH.glob('chapter5_result_*.csv'))

    # Hardcode the test files as requested
    test_filenames = [
        'chapter5_result_participant1_Easy2_250.csv',
        'chapter5_result_participant1_Medium3_250.csv',
        'chapter5_result_participant1_Hard1_250.csv'  # Typo from previous version corrected
    ]
    test_files = [p for p in all_files if p.name in test_filenames]

    # Select training files, excluding the test files
    train_files_all = [f for f in all_files if f not in test_files]

    # Select the specified number of files for each difficulty for training
    train_files_selected = []
    difficulties = {'Easy': 3, 'Medium': 4, 'Hard': 4}
    for diff, count in difficulties.items():
        diff_files = [f for f in train_files_all if extract_difficulty_from_filename(f.name) == diff]
        train_files_selected.extend(diff_files[:count])

    print(f"Found {len(train_files_selected)} training files and {len(test_files)} test files.")
    if len(train_files_selected) == 0 or len(test_files) != 3:
        print("Could not find enough files for training/testing. Please check file paths and names.")
        return

    # --- 2. Create Aggregated Training and Test Sets ---

    train_rows = []
    train_labels = []
    for file_path in train_files_selected:
        try:
            session_df = pd.read_csv(file_path, index_col=0)
            if not session_df.empty:
                agg_row = aggregate_session_features(session_df)
                if not agg_row.empty:
                    train_rows.append(agg_row)
                    train_labels.append(extract_difficulty_from_filename(file_path.name))
        except Exception as e:
            print(f"Could not process file {file_path.name}: {e}")

    test_rows = []
    test_labels = []
    for file_path in test_files:
        try:
            session_df = pd.read_csv(file_path, index_col=0)
            if not session_df.empty:
                agg_row = aggregate_session_features(session_df)
                if not agg_row.empty:
                    test_rows.append(agg_row)
                    test_labels.append(extract_difficulty_from_filename(file_path.name))
        except Exception as e:
            print(f"Could not process file {file_path.name}: {e}")

    if not train_rows or not test_rows:
        print("Failed to create training/test sets. The datasets may be empty after filtering.")
        return

    train_X = pd.concat(train_rows, ignore_index=True)
    train_y = pd.Series(train_labels, name="difficulty")

    test_X = pd.concat(test_rows, ignore_index=True)
    test_y = pd.Series(test_labels, name="difficulty")

    # Align columns - crucial if some files have different columns
    train_cols = train_X.columns
    test_cols = test_X.columns
    shared_cols = list(set(train_cols) & set(test_cols))

    train_X = train_X[shared_cols]
    test_X = test_X[shared_cols]

    # Impute NaN values that may result from std of a single point, etc.
    train_X = train_X.fillna(train_X.mean())
    test_X = test_X.fillna(train_X.mean())  # Use training mean for test set

    print("\n--- Created Aggregated Datasets ---")
    print(f"Training set shape: {train_X.shape}")
    print(f"Test set shape: {test_X.shape}")
    print(f"Training labels distribution:\n{train_y.value_counts()}")

    # --- 3. Scale Features ---
    scaler = StandardScaler()
    train_X_scaled = pd.DataFrame(scaler.fit_transform(train_X), columns=train_X.columns)
    test_X_scaled = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)

    # --- 4. Train and Evaluate Models ---

    learner = ClassificationAlgorithms()
    eval = ClassificationEvaluation()
    DataViz = VisualizeDataset(__file__)

    N_KCV_REPEATS = 3

    print("\n--- Comparing Classification Algorithms ---")

    # Non-deterministic classifiers
    perf_tr_nn, perf_te_nn = 0, 0
    perf_tr_rf, perf_te_rf = 0, 0
    for i in range(N_KCV_REPEATS):
        print(f"Training Non-Deterministic Run {i + 1}/{N_KCV_REPEATS}...")
        class_train_y, class_test_y, _, _ = learner.feedforward_neural_network(
            train_X_scaled, train_y, test_X_scaled, gridsearch=True)
        perf_tr_nn += eval.accuracy(train_y, class_train_y)
        perf_te_nn += eval.accuracy(test_y, class_test_y)

        class_train_y, class_test_y, _, _ = learner.random_forest(
            train_X_scaled, train_y, test_X_scaled, gridsearch=True)
        perf_tr_rf += eval.accuracy(train_y, class_train_y)
        perf_te_rf += eval.accuracy(test_y, class_test_y)

    # Deterministic classifiers
    print("Training Deterministic Classifiers...")
    class_train_y, class_test_y, _, _ = learner.support_vector_machine_with_kernel(
        train_X_scaled, train_y, test_X_scaled, gridsearch=True)
    perf_tr_svm = eval.accuracy(train_y, class_train_y)
    perf_te_svm = eval.accuracy(test_y, class_test_y)

    class_train_y, class_test_y, _, _ = learner.k_nearest_neighbor(
        train_X_scaled, train_y, test_X_scaled, gridsearch=True)
    perf_tr_knn = eval.accuracy(train_y, class_train_y)
    perf_te_knn = eval.accuracy(test_y, class_test_y)

    class_train_y, class_test_y, _, _ = learner.decision_tree(
        train_X_scaled, train_y, test_X_scaled, gridsearch=True)
    perf_tr_dt = eval.accuracy(train_y, class_train_y)
    perf_te_dt = eval.accuracy(test_y, class_test_y)

    class_train_y, class_test_y, class_prob_tr_nb, class_prob_te_nb = learner.naive_bayes(
        train_X_scaled, train_y, test_X_scaled)
    perf_tr_nb = eval.accuracy(train_y, class_train_y)
    perf_te_nb = eval.accuracy(test_y, class_test_y)

    scores = util.print_table_row_performances(
        'Aggregated Features', len(train_X), len(test_X),
        [
            (perf_tr_nn / N_KCV_REPEATS, perf_te_nn / N_KCV_REPEATS),
            (perf_tr_rf / N_KCV_REPEATS, perf_te_rf / N_KCV_REPEATS),
            (perf_tr_svm, perf_te_svm),
            (perf_tr_knn, perf_te_knn),
            (perf_tr_dt, perf_te_dt),
            (perf_tr_nb, perf_te_nb)
        ]
    )

    DataViz.plot_performances_classification(
        ['NN', 'RF', 'SVM', 'KNN', 'DT', 'NB'],
        ['Aggregated Features'],
        [scores]
    )

    # --- 5. Detailed Look at Best Model ---

    print("\n--- Detailed Analysis of Best Performing Model ---")

    # We'll use the results from the Naive Bayes model as it's often a good baseline
    # and provides the probabilities needed for the confusion matrix.
    test_cm = eval.confusion_matrix(test_y, class_test_y, class_prob_te_nb.columns)
    DataViz.plot_confusion_matrix(test_cm, class_prob_te_nb.columns, normalize=False)


if __name__ == '__main__':
    main()