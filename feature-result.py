import os
import json
import pandas as pd

# 定义文件路径
base_dir = "Feature-selection"
features_dir = os.path.join(base_dir, "features")
rfe_dir = os.path.join(base_dir, "rfe")

# 胁迫类型列表
stresses = ['cold', 'heat', 'drought', 'salt']

# 处理每种胁迫类型
for stress in stresses:
    # 找到对应的特征选择结果文件夹
    rfe_result_dir = os.path.join(rfe_dir, f"{stress}_rfe")
    
    # 遍历特征选择结果文件夹，处理每个结果文件
    for root, dirs, files in os.walk(rfe_result_dir):
        for file in files:
            if file == "selected_features.json":
                # 读取特征选择结果
                with open(os.path.join(root, file), 'r') as f:
                    selected_features = json.load(f)

                # 获取特征选择的文件夹名，例如 cold-CKSAAP-2-3_DDE-rfe
                feature_combination = os.path.basename(root)

                # 获取特征提取方法名，例如 CKSAAP-2-3_DDE
                feature_methods = feature_combination.replace(f"{stress}-", "").replace("-rfe", "")
                feature_types = feature_methods.split("_")

                # 初始化空的 DataFrame，用于拼接特征
                positive = pd.DataFrame()
                negative = pd.DataFrame()

                for feature_type in feature_types:
                    # 定义正负样本的文件名
                    positive_file = f"{stress}-{feature_type}.csv"
                    negative_file = f"non-{stress}-{feature_type}.csv"

                    # 读取正样本和负样本特征文件
                    positive_df = pd.read_csv(os.path.join(features_dir, stress, positive_file))
                    negative_df = pd.read_csv(os.path.join(features_dir, stress, negative_file))

                    # 更新列名以包含特征提取方法
                    positive_df.columns = [f'{feature_type}_{col}' for col in positive_df.columns]
                    negative_df.columns = [f'{feature_type}_{col}' for col in negative_df.columns]

                    # 拼接到完整的 DataFrame 中
                    positive = pd.concat([positive, positive_df], axis=1)
                    negative = pd.concat([negative, negative_df], axis=1)

                # 根据特征选择结果进行列筛选
                positive_selected_df = positive[selected_features]
                negative_selected_df = negative[selected_features]

                # 重置列名为从0开始的数字编号
                positive_selected_df.columns = range(positive_selected_df.shape[1])
                negative_selected_df.columns = range(negative_selected_df.shape[1])

                # 保存筛选后的正负样本特征到新的文件
                positive_selected_df.to_csv(os.path.join(root, f"{stress}-{feature_methods}_selected.csv"), index=False)
                negative_selected_df.to_csv(os.path.join(root, f"non-{stress}-{feature_methods}_selected.csv"), index=False)

