import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import logging

# 定义机器学习算法
ALGORITHMS = {
    'SVM': SVC(kernel='rbf', C=1, gamma='auto', random_state=42),
    'XGB': xgb.XGBClassifier(max_depth=6, eta=0.3, min_child_weight=1, objective='binary:logistic'),
    'RF': RandomForestClassifier(n_estimators=100, random_state=200, max_features='sqrt'),
    'GBDT': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
    'ADB': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=6, random_state=42), n_estimators=100, random_state=42),
    'BAG': BaggingClassifier(base_estimator=SVC(), n_estimators=100, random_state=0)
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

    # 对特征使用不同的模型与评价标准
    results = {}
    for algo_name, model in ALGORITHMS.items():
        results[algo_name] = {}
        score = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
        results[algo_name] = score
        logger.info(f'{algo_name}: {score}')
    return results

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

                result = process_files(positive_file_path, negative_file_path)
                for algo, metrics in result.items():
                    metrics['method'] = method
                    category_results.append({'algorithm': algo, **metrics})
        
        all_results[category] = pd.DataFrame(category_results)
    return all_results

# 保存结果到 Excel文件
def save_results_to_excel(results, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in results.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == "__main__":
    directory_path = 'features'
    output_path = 'results.xlsx'  # 保存结果的文件路径

    results = process_directory(directory_path)
    save_results_to_excel(results, output_path)
