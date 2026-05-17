#!/usr/bin/env python3
"""
条件熵因果方向推断脚本

基于条件熵和最近邻的因果方向推断方法，用于确定两个变量之间的因果方向。
原理：正确的因果方向中，给定原因，结果的条件分布更简单（熵更小）

使用方法:
    python scripts/conditional_entropy_causal_direction.py --data data/processed/zh_dataset/1116/data_processed.csv --col1 column1 --col2 column2

输出:
    - 因果方向 (X->Y 或 Y->X)
    - 详细的条件熵计算结果
    - 可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import os
from typing import Tuple, Dict, Any
from sklearn.neighbors import NearestNeighbors

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConditionalEntropyCausalDirection:
    """
    基于条件熵的因果方向推断器
    """

    def __init__(self, data: pd.DataFrame, col1: str, col2: str, k: int = 5):
        """
        初始化因果方向推断器

        Args:
            data: 输入数据
            col1: 第一列名称
            col2: 第二列名称
            k: 最近邻数量
        """
        self.data = data
        self.col1 = col1
        self.col2 = col2
        self.k = k
        self.results = {}

        # 验证列是否存在
        if col1 not in data.columns:
            raise ValueError(f"列 '{col1}' 不存在于数据中")
        if col2 not in data.columns:
            raise ValueError(f"列 '{col2}' 不存在于数据中")

        logger.info(f"初始化条件熵因果方向推断器: {col1} vs {col2}, k={k}")

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理数据：标准化，移除NaN值

        Returns:
            标准化后的X和Y数组
        """
        # 提取两列数据
        x = self.data[self.col1].values
        y = self.data[self.col2].values

        # 移除NaN值
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x = x[valid_mask]
        y = y[valid_mask]

        logger.info(f"原始数据点数: {len(self.data)}, 有效数据点数: {len(x)}")

        if len(x) < self.k + 1:
            raise ValueError(f"有效数据点数太少 ({len(x)}), 至少需要 {self.k + 1} 个点")

        # 标准化数据
        x_norm = (x - np.mean(x)) / np.std(x)
        y_norm = (y - np.mean(y)) / np.std(y)

        logger.info(f"数据标准化完成: {self.col1} 范围 [{x_norm.min():.3f}, {x_norm.max():.3f}], "
                   f"{self.col2} 范围 [{y_norm.min():.3f}, {y_norm.max():.3f}]")

        return x_norm, y_norm

    def conditional_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        估计条件熵 H(Y|X)
        
        原理：正确的因果方向中，给定原因，结果的条件分布更简单（熵更小）
        
        Args:
            x: 自变量数组
            y: 因变量数组
            
        Returns:
            条件熵值
        """
        n = len(x)
        
        # 在X空间找最近邻
        nn = NearestNeighbors(n_neighbors=self.k+1)
        nn.fit(x.reshape(-1, 1))
        distances, indices = nn.kneighbors(x.reshape(-1, 1))
        
        # 计算Y在这些邻域中的方差
        y_variances = []
        neighbor_y_values = []
        
        for i in range(n):
            neighbor_indices = indices[i, 1:]  # 排除自身
            y_neighbors = y[neighbor_indices]
            y_variances.append(np.var(y_neighbors))
            neighbor_y_values.append(y_neighbors.tolist())
        
        # 估计条件熵（与log方差相关）
        avg_log_var = np.mean(np.log(np.array(y_variances) + 1e-10))
        
        # 存储中间结果
        self.results['neighbor_variances'] = y_variances
        self.results['neighbor_y_values'] = neighbor_y_values
        self.results['x_values'] = x.tolist()
        self.results['y_values'] = y.tolist()
        
        return avg_log_var
    
    def determine_causal_direction_entropy(self) -> Dict[str, Any]:
        """
        基于条件熵和最近邻的因果推断
        
        Returns:
            包含推断结果的字典
        """
        logger.info("开始条件熵因果方向推断...")

        # 预处理数据
        x, y = self.preprocess_data()

        # 计算两个方向的条件熵
        logger.info(f"计算 H({self.col2}|{self.col1})...")
        ce_xy = self.conditional_entropy(x, y)  # H(Y|X)
        
        # 清空中间结果以准备第二个方向
        temp_results = {}
        if 'neighbor_variances' in self.results:
            temp_results['neighbor_variances_xy'] = self.results['neighbor_variances']
            temp_results['neighbor_y_values_xy'] = self.results['neighbor_y_values']
        
        self.results.clear()
        self.results.update(temp_results)
        
        logger.info(f"计算 H({self.col1}|{self.col2})...")
        ce_yx = self.conditional_entropy(y, x)  # H(X|Y)
        
        # 比较条件熵
        entropy_diff = abs(ce_xy - ce_yx)
        
        # 较小的条件熵对应的方向更可能是因果方向
        if entropy_diff < 0.01:  # 阈值，表示差异太小
            direction = "undetermined"
            confidence = "low"
            reason = f"条件熵差异太小 ({entropy_diff:.3f} < 0.01)"
        elif ce_xy < ce_yx:
            direction = f"{self.col1} -> {self.col2}"
            confidence = "high" if entropy_diff > 0.1 else "medium"
            reason = f"H({self.col2}|{self.col1}) = {ce_xy:.3f} < H({self.col1}|{self.col2}) = {ce_yx:.3f}"
        else:
            direction = f"{self.col2} -> {self.col1}"
            confidence = "high" if entropy_diff > 0.1 else "medium"
            reason = f"H({self.col1}|{self.col2}) = {ce_yx:.3f} < H({self.col2}|{self.col1}) = {ce_xy:.3f}"

        # 整理结果
        final_results = {
            'direction': direction,
            'confidence': confidence,
            'reason': reason,
            'ce_xy': ce_xy,
            'ce_yx': ce_yx,
            'entropy_difference': entropy_diff,
            'sample_size': len(x),
            'col1': self.col1,
            'col2': self.col2,
            'k': self.k
        }
        
        # 合并所有中间结果
        final_results.update(self.results)
        self.results = final_results

        logger.info(f"条件熵因果方向推断完成: {direction}")
        logger.info(f"置信度: {confidence}, 原因: {reason}")

        return self.results

    def visualize_results(self, output_dir: str = None) -> None:
        """
        可视化结果

        Args:
            output_dir: 输出目录
        """
        if not self.results:
            logger.warning("没有结果可以可视化")
            return

        x, y = self.preprocess_data()

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'条件熵因果方向分析: {self.col1} vs {self.col2}', fontsize=16)

        # 子图1: 散点图
        axes[0, 0].scatter(x, y, alpha=0.6, s=20)
        axes[0, 0].set_xlabel(f'{self.col1} (标准化)')
        axes[0, 0].set_ylabel(f'{self.col2} (标准化)')
        axes[0, 0].set_title('数据散点图')
        axes[0, 0].grid(True, alpha=0.3)

        # 子图2: 邻域方差 X->Y
        if 'neighbor_variances' in self.results:
            axes[0, 1].plot(x, self.results['neighbor_variances'], 'b-', alpha=0.7)
            axes[0, 1].set_xlabel(f'{self.col1}')
            axes[0, 1].set_ylabel(f'Var({self.col2}|{self.col1}邻域)')
            axes[0, 1].set_title(f'{self.col1} -> {self.col2} 邻域方差')
            axes[0, 1].grid(True, alpha=0.3)

        # 子图3: 邻域方差 Y->X
        x_norm = (self.data[self.col2].values - np.mean(self.data[self.col2].values)) / np.std(self.data[self.col2].values)
        y_norm = (self.data[self.col1].values - np.mean(self.data[self.col1].values)) / np.std(self.data[self.col1].values)
        valid_mask = ~(np.isnan(x_norm) | np.isnan(y_norm))
        x_norm = x_norm[valid_mask]
        y_norm = y_norm[valid_mask]
        
        # 计算Y->X方向的邻域方差
        nn = NearestNeighbors(n_neighbors=self.k+1)
        nn.fit(y_norm.reshape(-1, 1))
        distances, indices = nn.kneighbors(y_norm.reshape(-1, 1))
        
        yx_variances = []
        for i in range(len(y_norm)):
            neighbor_indices = indices[i, 1:]
            x_neighbors = x_norm[neighbor_indices]
            yx_variances.append(np.var(x_neighbors))
        
        axes[1, 0].plot(y_norm, yx_variances, 'r-', alpha=0.7)
        axes[1, 0].set_xlabel(f'{self.col2}')
        axes[1, 0].set_ylabel(f'Var({self.col1}|{self.col2}邻域)')
        axes[1, 0].set_title(f'{self.col2} -> {self.col1} 邻域方差')
        axes[1, 0].grid(True, alpha=0.3)

        # 子图4: 条件熵比较
        entropy_values = [self.results['ce_xy'], self.results['ce_yx']]
        labels = [f'H({self.col2}|{self.col1})', f'H({self.col1}|{self.col2})']
        colors = ['lightblue', 'lightcoral']

        bars = axes[1, 1].bar(labels, entropy_values, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('条件熵值')
        axes[1, 1].set_title('条件熵比较')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, value in zip(bars, entropy_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        # 保存图表
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt_path = os.path.join(output_dir, 'conditional_entropy_causal_direction.png')
            plt.savefig(plt_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化图表已保存到: {plt_path}")

        plt.show()

    def save_results(self, output_dir: str) -> None:
        """
        保存结果到文件

        Args:
            output_dir: 输出目录
        """
        if not self.results:
            logger.warning("没有结果可以保存")
            return

        os.makedirs(output_dir, exist_ok=True)

        # 保存主要结果为CSV
        main_results = {
            'direction': self.results['direction'],
            'confidence': self.results['confidence'],
            'reason': self.results['reason'],
            'ce_xy': self.results['ce_xy'],
            'ce_yx': self.results['ce_yx'],
            'entropy_difference': self.results['entropy_difference'],
            'sample_size': self.results['sample_size'],
            'col1': self.results['col1'],
            'col2': self.results['col2'],
            'k': self.results['k']
        }
        
        results_df = pd.DataFrame([main_results])
        results_path = os.path.join(output_dir, 'conditional_entropy_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"主要结果已保存到: {results_path}")

        # 保存详细的邻域方差结果
        if 'neighbor_variances' in self.results:
            detailed_df = pd.DataFrame({
                'x_value': self.results['x_values'],
                'y_value': self.results['y_values'],
                'neighbor_variance_xy': self.results['neighbor_variances'],
            })
            detailed_path = os.path.join(output_dir, 'neighbor_variances.csv')
            detailed_df.to_csv(detailed_path, index=False)
            logger.info(f"详细邻域方差已保存到: {detailed_path}")


def determine_causal_direction_entropy(X, Y, k=5):
    """
    基于条件熵和最近邻的因果推断
    原理：正确的因果方向中，给定原因，结果的条件分布更简单（熵更小）
    """
    from sklearn.neighbors import NearestNeighbors
    
    # 标准化
    X_norm = (X - np.mean(X)) / np.std(X)
    Y_norm = (Y - np.mean(Y)) / np.std(Y)
    
    def conditional_entropy(x, y):
        """估计条件熵 H(Y|X)"""
        n = len(x)
        
        # 在X空间找最近邻
        nn = NearestNeighbors(n_neighbors=k+1)
        nn.fit(x.reshape(-1, 1))
        distances, indices = nn.kneighbors(x.reshape(-1, 1))
        
        # 计算Y在这些邻域中的方差
        y_variances = []
        for i in range(n):
            neighbor_indices = indices[i, 1:]  # 排除自身
            y_neighbors = y[neighbor_indices]
            y_variances.append(np.var(y_neighbors))
        
        # 估计条件熵（与log方差相关）
        avg_log_var = np.mean(np.log(np.array(y_variances) + 1e-10))
        return avg_log_var
    
    # 计算两个方向的条件熵
    ce_xy = conditional_entropy(X_norm, Y_norm)  # H(Y|X)
    ce_yx = conditional_entropy(Y_norm, X_norm)  # H(X|Y)
    
    # 较小的条件熵对应的方向更可能是因果方向
    if ce_xy < ce_yx:
        return 'X->Y', (ce_xy, ce_yx)
    else:
        return 'Y->X', (ce_xy, ce_yx)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于条件熵的因果方向推断')
    parser.add_argument('--data', help='输入CSV文件路径', default='../data/raw/zh_dataset/0105/data.csv')
    parser.add_argument('--col1', help='第一列名称', default='nvidia_smi_ecc_errors_uncorrected_volatile_total_10.104.128.205:9835')
    parser.add_argument('--col2', help='第二列名称', default='nv_inference_request_failure_10.104.128.205:8002')
    parser.add_argument('--k', type=int, default=100, help='最近邻数量 (默认: 5)')
    parser.add_argument('--output', default='results', help='输出目录 (默认: results)')
    parser.add_argument('--visualize', action='store_true', help='显示可视化图表')
    parser.add_argument('--verbose', action='store_true', help='详细输出')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # 读取数据
        logger.info(f"读取数据文件: {args.data}")
        data = pd.read_csv(args.data)
        logger.info(f"数据形状: {data.shape}")
        logger.info(f"可用列: {list(data.columns)}")

        # 创建推断器
        analyzer = ConditionalEntropyCausalDirection(data, args.col1, args.col2, args.k)

        # 执行推断
        results = analyzer.determine_causal_direction_entropy()

        # 输出结果
        print("\n" + "="*60)
        print("条件熵因果方向推断结果")
        print("="*60)
        print(f"数据文件: {args.data}")
        print(f"分析变量: {args.col1} vs {args.col2}")
        print(f"最近邻数量: {args.k}")
        print(f"样本数量: {results['sample_size']}")
        print(f"条件熵 H({args.col2}|{args.col1}): {results['ce_xy']:.6f}")
        print(f"条件熵 H({args.col1}|{args.col2}): {results['ce_yx']:.6f}")
        print(f"熵值差: {results['entropy_difference']:.6f}")
        print("-"*60)
        print(f"因果方向: {results['direction']}")
        print(f"置信度: {results['confidence']}")
        print(f"推断原因: {results['reason']}")
        print("="*60)

        # 保存结果
        analyzer.save_results(args.output)

        # 可视化
        if args.visualize:
            analyzer.visualize_results(args.output)

    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise


if __name__ == "__main__":
    main()