import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


RANDOM_SEED = 42
# 定义机器学习算法
ALGORITHMS = {
    'SVM': SVC(probability=True, random_state=RANDOM_SEED),
    'XGB': xgb.XGBClassifier(objective='binary:logistic', random_state=RANDOM_SEED),
    'RF': RandomForestClassifier(random_state=RANDOM_SEED),
    'GBDT': GradientBoostingClassifier(random_state=RANDOM_SEED),
    'ADB': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=RANDOM_SEED), random_state=RANDOM_SEED),
    'BAG': BaggingClassifier(base_estimator=SVC(probability=True, random_state=RANDOM_SEED), random_state=RANDOM_SEED)
}

# 定义超参数搜索空间
PARAM_GRIDS = {
    'SVM': {
        'C': [0.1, 1, 10, 20, 30, 100],
        'kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
        'class_weight': ['balanced', None]
    },
    'XGB': {
        'max_depth': [3, 6, 9],
        'eta': [0.01, 0.1, 0.3],
        'min_child_weight': [1, 3, 5],
        'scale_pos_weight': [1, 3]
    },
    'RF': {
        'n_estimators': [100, 200, 500],
        'criterion':['gini', 'entropy', 'log_loss'],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None]
    },
    'GBDT': {
        'loss': ['log_loss', 'deviance', 'exponential'],
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.3],
        'criterion': ['friedman_mse', 'squared_error'],
        'max_depth': [3, 5, 7],
        'max_features': ['auto', 'sqrt', 'log2', None]
    },
    'ADB': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'base_estimator__criterion': ['gini', 'entropy', 'log_loss'],
        'base_estimator__max_features': ['auto', 'sqrt', 'log2', None],
        'base_estimator__max_depth': [3, 6, 9],
        'base_estimator__class_weight': ['balanced', None]
    },
    'BAG': {
        'n_estimators': [10, 50, 100],
        'bootstrap_features': [True, False], 
        'base_estimator__C': [0.1, 1, 10, 20, 30],
        'base_estimator__kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
        'base_estimator__class_weight': ['balanced', None]
    }
}


# 定义评价方法
SCORING = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='binary'),
    'recall': make_scorer(recall_score, average='binary'),
    'f1_score': make_scorer(f1_score, average='binary'),
    'roc_auc': make_scorer(roc_auc_score, needs_threshold=True)
}

# 定义日志文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filename='log.log',
    filemode='a'
)
logger = logging.getLogger(__name__)


# 处理数据文件
def process_files(positive_file, negative_file):
    positive = pd.read_csv(positive_file)
    negative = pd.read_csv(negative_file)
    
    positive.insert(positive.shape[1], positive.shape[1], 1)
    negative.insert(negative.shape[1], negative.shape[1], 0)

    data = pd.concat([positive, negative], axis=0)

    num_features = data.shape[1] - 1
    X = np.array(data.iloc[:, 0:num_features])
    y = np.array(data.iloc[:, num_features])
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    # 对特征使用不同的模型与评价标准进行超参数搜索
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for algo_name, model in ALGORITHMS.items():
        param_grid = PARAM_GRIDS.get(algo_name, {})
        print(param_grid)
        grid_search = GridSearchCV(model, param_grid, scoring=SCORING, cv=cv, refit='roc_auc', return_train_score=True, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # 保存每个超参数组合的结果
        for i in range(len(grid_search.cv_results_['params'])):
            result = {
                'algorithm': algo_name,
                'params': grid_search.cv_results_['params'][i],
                'mean_test_accuracy': grid_search.cv_results_['mean_test_accuracy'][i],
                'mean_test_precision': grid_search.cv_results_['mean_test_precision'][i],
                'mean_test_recall': grid_search.cv_results_['mean_test_recall'][i],
                'mean_test_f1_score': grid_search.cv_results_['mean_test_f1_score'][i],
                'mean_test_roc_auc': grid_search.cv_results_['mean_test_roc_auc'][i]
            }
            results.append(result)
            logger.info(result)
    return pd.DataFrame(results)

# 遍历目录并处理文件
def process_directory(directory_path):
    all_results = {}
    categories = ['cold', 'drought', 'heat', 'salt']
    for category in categories:
        category_results = []
        category_path = os.path.join(directory_path, category)
        
        # 获取正负样本文件名列表
        positive_files = [f for f in os.listdir(category_path) if f.startswith(category + '-')]
        negative_files = [f for f in os.listdir(category_path) if f.startswith('non-' + category + '-')]

        # 创建一个字典来匹配正样本和负样本文件
        file_pairs = {}
        for pos_file in positive_files:
            method = '-'.join(pos_file.split('-')[1:])
            file_pairs[method] = {'positive': pos_file}
        
        for neg_file in negative_files:
            method = '-'.join(neg_file.split('-')[2:])
            if method in file_pairs:
                file_pairs[method]['negative'] = neg_file

        # 处理每对文件
        for method, files in file_pairs.items():
            if 'positive' in files and 'negative' in files:
                positive_file_path = os.path.join(category_path, files['positive'])
                negative_file_path = os.path.join(category_path, files['negative'])
                logger.info(f"Processing files: {positive_file_path} and {negative_file_path}")

                result_df = process_files(positive_file_path, negative_file_path)
                result_df['method'] = method
                category_results.append(result_df)
        
        all_results[category] = pd.concat(category_results, ignore_index=True)
    return all_results

# 保存结果到 Excel文件
def save_results_to_excel(results, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in results.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == "__main__":
    directory_path = 'features-selected'
    output_path = 'hyperparameter_search_results_4.xlsx'  # 保存结果的文件路径

    results = process_directory(directory_path)
    save_results_to_excel(results, output_path)
