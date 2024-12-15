import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score

def assemble_prompt(row):
    """
    参数：
    - row: pandas.Series，包含一行 CSV 数据

    返回：
    - str，组装后的 prompt
    """
    base_prompt = row['prompt']  

    with open('政策反讽评论_prompt.md', 'r', encoding='utf-8') as file:
        prompt_template = file.read()

    assembled_prompt = prompt_template + base_prompt # 有可能替换为 .replace()方法 具体看 prompt怎么写
    return assembled_prompt

def evaluate_model(csv_file_path, policy_client):
    data = pd.read_csv(csv_file_path)
    
    if 'prompt' not in data.columns or 'label' not in data.columns:
        raise ValueError("CSV 文件必须包含 'prompt' 和 'label' 两列")

    true_labels = data['label']

    # 检查标签是否为二分类问题
    unique_labels = true_labels.unique()
    if len(unique_labels) > 2:
        raise ValueError("仅支持二分类问题，标签包含多于两个类别")
    
    # 模型预测
    predictions = []
    for _, row in data.iterrows():
        prompt = assemble_prompt(row)
        prediction = policy_client(prompt)  
        predictions.append(int(prediction))  # 确保得是整数类型

    true_labels = true_labels.astype(int)
    predictions = pd.Series(predictions)

    # 计算评估指标
    accuracy = accuracy_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    # 打印评估结果
    print("模型评估结果：")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# 主函数
if __name__ == "__main__":
    # 假设 Policy_client 类在另一个文件中定义，例如 policy_client_module.py
    from policy_client_module import Policy_client  # 替换为实际文件名和类名

    # 实例化 Policy_client 对象
    base_url = ""  # 使用vllm 部署之后：http://127.0.0.1:8000/v1/chat/completions
    policy_client_instance = Policy_client(base_url=base_url)  # 替换为实际的初始化方式

    csv_file_path = "data.csv"  # 政策反讽评论数据

    # 调用评估函数
    evaluate_model(csv_file_path, policy_client_instance)