import os
import logging
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

RANDOM_SEED = 42


# 定义日志文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filename='log1.log',
    filemode='a'
)
logger = logging.getLogger(__name__)


# 处理数据文件并评估模型
def process_files(positive_file, negative_file, model):
    positive = pd.read_csv(positive_file)
    negative = pd.read_csv(negative_file)
    
    positive.insert(positive.shape[1], positive.shape[1], 1)
    negative.insert(negative.shape[1], negative.shape[1], 0)

    data = pd.concat([positive, negative], axis=0)

    num_features = data.shape[1] - 1
    X = np.array(data.iloc[:, 0:num_features])
    y = np.array(data.iloc[:, num_features])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    # 使用传入的模型进行训练与评价
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # 计算各项评价指标
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 'N/A'
    }
    
    logger.info(results)
    return pd.DataFrame([results])


# 遍历目录并处理文件
def process_directory(directory_path, model):
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

                result_df = process_files(positive_file_path, negative_file_path, model)
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
    output_path = 'model_evaluation_results.xlsx'  # 保存结果的文件路径

    # 传入一个SVM模型
    model = SVC(C=30, kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_SEED)

    results = process_directory(directory_path, model)
    save_results_to_excel(results, output_path)

