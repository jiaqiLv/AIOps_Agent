import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

def calculate_residual_correlation(csv_path, col_a, col_b, col_c):
    """
    计算C解释A的残差和C解释B的残差的相关系数
    
    Parameters:
    -----------
    csv_path : str
        CSV文件路径
    col_a, col_b, col_c : str
        三列的列名
        
    Returns:
    --------
    corr : float
        残差相关系数
    p_val : float
        p值
    """
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 提取指定列
    data = df[[col_a, col_b, col_c]].dropna()
    
    # 1. 拟合从 C 到 A 的影响
    reg_a = LinearRegression().fit(data[[col_c]], data[col_a])
    res_a = data[col_a] - reg_a.predict(data[[col_c]])
    
    # 2. 拟合从 C 到 B 的影响
    reg_b = LinearRegression().fit(data[[col_c]], data[col_b])
    res_b = data[col_b] - reg_b.predict(data[[col_c]])
    
    # 3. 计算残差的相关性
    corr, p_val = pearsonr(res_a, res_b)
    
    print(f"--- 残差诊断结果 ---")
    print(f"残差相关系数 (r): {corr:.4f}")
    print(f"P-value: {p_val:.4e}")
    print(f"样本数量: {len(data)}")
    
    # 输出回归系数信息
    print(f"\n--- 回归系数信息 ---")
    print(f"C -> A 回归系数: {reg_a.coef_[0]:.4f}")
    print(f"C -> B 回归系数: {reg_b.coef_[0]:.4f}")
    
    return corr, p_val, data

# 使用示例
if __name__ == "__main__":
    # CSV文件路径
    csv_file = r"../data/processed/zh_dataset/0922/data_processed.csv"
    
    # 指定三列（根据CSV文件的实际列名修改）
    col_A = '202_IO_Utilization'           # 第一列
    col_B = 'ai_request_avg_duration'          # 第二列
    col_C = '202_Network_Traffic_out'   # 第三列
    
    # 计算残差相关系数
    corr, p_val, used_data = calculate_residual_correlation(csv_file, col_A, col_B, col_C)
    
    print(f"\n最终结果: C解释A和B的残差相关系数为 {corr:.4f} (p={p_val:.4e})")