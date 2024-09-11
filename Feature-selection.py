import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve, average_precision_score, accuracy_score, precision_score, recall_score, f1_score


# Function to load and merge datasets
def load_and_merge_datasets(feature_types, dir):
    """
    Load and merge positive and negative datasets based on feature types.

    Parameters:
    - feature_types: List of feature extraction methods, e.g., ['CKSAAP-2-3', 'DPC-2'].
    - dir: Directory where the feature files are located.

    Returns:
    - data: A DataFrame containing merged positive and negative samples.
    - stress_type: The type of stress based on the directory name.
    """
    positive = pd.DataFrame()
    negative = pd.DataFrame()
    stress_type = dir.split('/')[-1]
    for feature_type in feature_types:
        pos_file = os.path.join(dir, f'{stress_type}-{feature_type}.csv')
        neg_file = os.path.join(dir, f'non-{stress_type}-{feature_type}.csv')
        pos = pd.read_csv(pos_file)
        neg = pd.read_csv(neg_file)
        pos.columns = [f'{feature_type}_{col}' for col in pos.columns]
        neg.columns = [f'{feature_type}_{col}' for col in neg.columns]
        positive = pd.concat([positive, pos], axis=1)
        negative = pd.concat([negative, neg], axis=1)
    
    positive.insert(positive.shape[1], positive.shape[1], 1)
    negative.insert(negative.shape[1], negative.shape[1], 0)
    data = pd.concat([positive, negative], axis=0)
    
    return data, stress_type

# Function to prepare data
def prepare_data(data, test_size=0.2, scale=False, random_seed=42):
    """
    Prepare training and testing data.

    Parameters:
    - data: DataFrame containing features and labels.
    - test_size: Proportion of the dataset to include in the test split.
    - scale: Whether to standardize the features.
    - random_seed: Random seed for reproducibility.

    Returns:
    - X_train: Features for training.
    - X_test: Features for testing.
    - y_train: Labels for training.
    - y_test: Labels for testing.
    """
    num_features = data.shape[1] - 1
    X = np.array(data.iloc[:, 0:num_features])
    y = np.array(data.iloc[:, num_features])
    
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, stratify=y)
    return X_train, X_test, y_train, y_test

# Function to perform RFECV
def perform_rfecv(X_train, y_train, C=1, balance=True, min_features_to_select=1, step=1, n_jobs=1, random_seed=42):
    """
    Perform Recursive Feature Elimination with Cross-Validation (RFECV).

    Parameters:
    - X_train: Features for training.
    - y_train: Labels for training.
    - C: Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
    - balance: Whether to use balanced class weights in LinearSVC.
    - min_features_to_select: Minimum number of features to select in RFECV.
    - step: Number of features to remove at each iteration.
    - n_jobs: Number of jobs to run in parallel.
    - random_seed: Random seed for reproducibility.

    Returns:
    - rfecv: The fitted RFECV object.
    """
    linear_svc = LinearSVC(class_weight='balanced' if balance else None, C=C if not balance else 1 , max_iter=1000000, random_state=random_seed)
    rfecv = RFECV(estimator=linear_svc, step=step, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed),
                  scoring=make_scorer(roc_auc_score, needs_threshold=True), min_features_to_select=min_features_to_select, n_jobs=n_jobs, verbose=1)
    rfecv.fit(X_train, y_train)
    return rfecv

# Function to evaluate the model
def evaluate_model(model, X_train, X_test, y_train, y_test, rfecv, full_path):
    """
    Evaluate the model using the RFECV selected features and save the results.

    Parameters:
    - model: The model to evaluate (e.g., SVC, LinearSVC).
    - X_train: Features for training.
    - X_test: Features for testing.
    - y_train: Labels for training.
    - y_test: Labels for testing.
    - rfecv: The fitted RFECV object.
    - full_path: Directory to save the evaluation results and plots.

    Returns:
    - train_results: Evaluation metrics for the training set.
    - test_results: Evaluation metrics for the test set.
    - fpr: False positive rate for ROC curve.
    - tpr: True positive rate for ROC curve.
    - test_roc_auc: ROC AUC for the test set.
    """
    optimal_n_features = rfecv.n_features_
    print(f'Optimal number of features: {optimal_n_features}')
    
    X_train_rfe = rfecv.transform(X_train)
    X_test_rfe = rfecv.transform(X_test)
    
    model.fit(X_train_rfe, y_train)
    y_train_pred = model.predict(X_train_rfe)
    y_train_pred_prob = model.predict_proba(X_train_rfe)[:, 1]
    y_test_pred = model.predict(X_test_rfe)
    y_test_pred_prob = model.predict_proba(X_test_rfe)[:, 1]
    
    metrics = lambda y_true, y_pred, y_pred_prob: {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': make_scorer(roc_auc_score, needs_threshold=True)._score_func(y_true, y_pred_prob),
        'prc_auc': average_precision_score(y_true, y_pred_prob)
    }
    
    train_results = metrics(y_train, y_train_pred, y_train_pred_prob)
    test_results = metrics(y_test, y_test_pred, y_test_pred_prob)
    
    results_df = pd.DataFrame([train_results, test_results], index=['train', 'test'])
    results_df.to_csv(os.path.join(full_path, 'evaluation_metrics.csv'))
    print(f"Evaluation metrics have been written to {os.path.join(full_path, 'evaluation_metrics.csv')}")
    
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob)
    return fpr, tpr, test_results['roc_auc']

# Function to plot and save ROC curve
def plot_and_save_roc_curve(fpr, tpr, roc_auc, full_path):
    """
    Plot and save the ROC curve.

    Parameters:
    - fpr: False positive rate for ROC curve.
    - tpr: True positive rate for ROC curve.
    - roc_auc: ROC AUC score.
    - full_path: Directory to save the ROC curve plot.
    """
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(full_path, 'roc_curve_optimal.png'))
    plt.close()

# Function to save feature selection data
def save_feature_selection_data(rfecv, full_path):
    """
    Save the feature number vs AUROC data to a JSON file.

    Parameters:
    - rfecv: The fitted RFECV object.
    - full_path: Directory to save the JSON file.
    """
    feature_vs_auroc = {
        'feature_count': list(range(rfecv.min_features_to_select, len(rfecv.cv_results_['mean_test_score']) + rfecv.min_features_to_select)),
        'train_auroc': list(rfecv.cv_results_['mean_test_score']),
    }
    with open(os.path.join(full_path, 'feature_vs_auroc.json'), 'w') as f:
        json.dump(feature_vs_auroc, f, indent=4)
    print(f"Feature count vs AUROC data have been written to {os.path.join(full_path, 'feature_vs_auroc.json')}")
    return feature_vs_auroc

# Function to plot and save feature vs AUC graph
def plot_and_save_feature_vs_auc(feature_vs_auroc, rfecv, full_path):
    """
    Plot and save the relationship between the number of features and AUROC.

    Parameters:
    - feature_vs_auc: Dictionary containing feature count and AUROC data.
    - rfecv: The fitted RFECV object.
    - full_path: Directory to save the plot.
    """
    optimal_n_features = rfecv.n_features_
    roc_auc_train = max(feature_vs_auroc['train_auroc'])
    
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross-validation score")
    plt.plot(feature_vs_auroc['feature_count'], feature_vs_auroc['train_auroc'])
    plt.axvline(x=optimal_n_features, color='navy', lw=2, linestyle='--')
    plt.scatter([optimal_n_features], [roc_auc_train], color='red')
    plt.text(optimal_n_features, roc_auc_train, f'({optimal_n_features}, {roc_auc_train:.3f})', 
             verticalalignment='bottom', horizontalalignment='right')
    plt.title("Number of Features vs. Cross-validation Score (AUROC)")
    plt.savefig(os.path.join(full_path, 'feature_vs_auroc.png'))
    plt.close()

# Main function
def feature_selection_and_evaluation(feature_types, dir, model, C=1, min_features_to_select=1, test_size=0.2, 
                                     scale=False, balance=True, step=1, n_jobs=1, random_seed=42):
    data, stress_type = load_and_merge_datasets(feature_types, dir)
    X_train, X_test, y_train, y_test = prepare_data(data, test_size=test_size, scale=scale, random_seed=random_seed)
    
    main_folder_name = f'{stress_type}_rfe'
    if not os.path.exists(main_folder_name):
        os.makedirs(main_folder_name)

    sub_folder_name = f'{stress_type}-{"_".join(feature_types)}-rfe'
    full_path = os.path.join(main_folder_name, sub_folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    rfecv = perform_rfecv(X_train, y_train, C=C, balance=balance, min_features_to_select=min_features_to_select, 
                          step=step, n_jobs=n_jobs, random_seed=random_seed)
    
    fpr, tpr, roc_auc = evaluate_model(model, X_train, X_test, y_train, y_test, rfecv, full_path)
    
    plot_and_save_roc_curve(fpr, tpr, roc_auc, full_path)
    
    feature_vs_auroc = save_feature_selection_data(rfecv, full_path)
    plot_and_save_feature_vs_auc(feature_vs_auroc, rfecv, full_path)
    
    # Get the names of the selected features
    selected_features = data.columns[:-1][rfecv.support_].tolist()
    # Output selected feature names to a JSON file
    with open(os.path.join(full_path, 'selected_features.json'), 'w') as f:
        json.dump(selected_features, f, indent=4)
    print(f"Selected features have been written to {os.path.join(full_path, 'selected_features.json')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Feature Selection and Model Evaluation")
    parser.add_argument('--feature_types', nargs='+', required=True, help="List of feature extraction methods, e.g., ['CKSAAP-2-3', 'DPC-2']")
    parser.add_argument('--dir', required=True, help="Directory of feature files")
    parser.add_argument('--min_features_to_select', type=int, default=1, help="Minimum number of features to select in RFECV")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the test set, default is 0.2")
    parser.add_argument('--scale', action='store_true', help="Whether to standardize the features")
    parser.add_argument('--balance', action='store_true', help="Whether to use balanced class weights in LinearSVC")
    parser.add_argument('--C', type=float, default=1.0, help="Penalty parameter for LinearSVC")
    parser.add_argument('--step', type=int, default=1, help="Step size in RFECV")
    parser.add_argument('--n_jobs', type=int, default=1, help="Number of jobs to run in parallel for RFECV")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Model to evaluate on the test set
    model = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=args.random_seed)
    
    feature_selection_and_evaluation(
        feature_types=args.feature_types,
        dir=args.dir,
        model=model,
        min_features_to_select=args.min_features_to_select,
        test_size=args.test_size,
        scale=args.scale,
        balance=args.balance,
        C=args.C,
        step=args.step,
        n_jobs=args.n_jobs,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    if len(os.sys.argv) > 1:
        main()
    else:
        RANDOM_SEED = 42
        # Default settings for direct execution
        # 'CKSAAP-2-3', 'TPC-2', 'DDE', 'DPC-2'
        feature_types = ['DDE']
        dir = 'features/salt'
        model = SVC(C=10, kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_SEED)
        
        feature_selection_and_evaluation(
            feature_types=feature_types,
            dir=dir,
            model=model,
            min_features_to_select=1,
            test_size=0.2,
            scale=False,
            balance=True,
            C=1,
            step=1,
            n_jobs=-1,
            random_seed=RANDOM_SEED
        )

