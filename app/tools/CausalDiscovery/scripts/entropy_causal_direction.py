#!/usr/bin/env python3
"""
信息熵因果方向推断脚本

基于信息熵的因果方向推断方法，用于确定两个变量之间的因果方向。
使用差分近似和熵值比较来推断因果关系。

使用方法:
    python scripts/entropy_causal_direction.py --data data/processed/zh_dataset/1116/data_processed.csv --col1 column1 --col2 column2

输出:
    - 因果方向 (X->Y 或 Y->X)
    - 详细的熵值计算结果
    - 可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import os
from typing import Tuple, Dict, Any
from sklearn.preprocessing import MinMaxScaler

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EntropyCausalDirection:
    """
    基于信息熵的因果方向推断器
    """
    
    def __init__(self, data: pd.DataFrame, col1: str, col2: str):
        """
        初始化因果方向推断器
        
        Args:
            data: 输入数据
            col1: 第一列名称
            col2: 第二列名称
        """
        self.data = data
        self.col1 = col1
        self.col2 = col2
        self.results = {}
        
        # 验证列是否存在
        if col1 not in data.columns:
            raise ValueError(f"列 '{col1}' 不存在于数据中")
        if col2 not in data.columns:
            raise ValueError(f"列 '{col2}' 不存在于数据中")
            
        logger.info(f"初始化因果方向推断器: {col1} vs {col2}")
    
    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理数据：标准化到[0,1]区间，移除NaN值，移除零值对

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

        logger.info(f"原始数据点数: {len(self.data)}, 移除NaN后数据点数: {len(x)}")

        # 移除x或y中值为0的成对观测值
        zero_mask = (x != 0) & (y != 0)
        x = x[zero_mask]
        y = y[zero_mask]

        logger.info(f"移除零值对后数据点数: {len(x)}")
        logger.info(f"移除的零值对数量: {np.sum(~zero_mask)}")

        if len(x) < 10:
            raise ValueError(f"有效数据点数太少 ({len(x)}), 至少需要10个点")

        # 标准化到[0,1]区间
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))

        x_scaled = scaler_x.fit_transform(x.reshape(-1, 1)).flatten()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        logger.info(f"数据标准化完成: {self.col1} 范围 [{x_scaled.min():.3f}, {x_scaled.max():.3f}], "
                   f"{self.col2} 范围 [{y_scaled.min():.3f}, {y_scaled.max():.3f}]")

        return x_scaled, y_scaled
    
    def compute_derivatives(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算导数的差分近似
        
        Args:
            x: 自变量数组
            y: 因变量数组
            
        Returns:
            导数数组和对应的x值
        """
        # 按x排序
        sort_indices = np.argsort(x)
        x_sorted = x[sort_indices]
        y_sorted = y[sort_indices]
        
        # 计算差分
        dx = np.diff(x_sorted)
        dy = np.diff(y_sorted)
        
        # 避免除零错误
        valid_derivatives = dx != 0
        x_derivative_points = x_sorted[:-1][valid_derivatives]
        derivatives = dy[valid_derivatives] / dx[valid_derivatives]
        
        logger.info(f"计算导数: {len(derivatives)} 个有效导数值")
        
        return x_derivative_points, derivatives
    
    def compute_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算C(X->Y) = 1/(N-1) * sum(log|f'(x_i)|)
        
        Args:
            x: 自变量数组
            y: 因变量数组
            
        Returns:
            熵值
        """
        x_points, derivatives = self.compute_derivatives(x, y)
        
        if len(derivatives) == 0:
            raise ValueError("无法计算导数，可能数据有问题")
        
        # 计算熵
        entropy_values = np.log(np.abs(derivatives) + 1e-10)  # 添加小常数避免log(0)
        entropy = np.mean(entropy_values)
        
        logger.info(f"导数范围: [{derivatives.min():.3f}, {derivatives.max():.3f}], "
                   f"熵值: {entropy:.3f}")
        
        # 存储中间结果
        self.results['entropy_values'] = entropy_values
        self.results['derivatives'] = derivatives
        self.results['x_points'] = x_points
        
        return entropy
    
    def infer_direction(self) -> Dict[str, Any]:
        """
        推断因果方向
        
        Returns:
            包含推断结果的字典
        """
        logger.info("开始因果方向推断...")
        
        # 预处理数据
        x, y = self.preprocess_data()
        
        # 计算两个方向的熵
        logger.info("计算 X -> Y 方向的熵...")
        c_xy = self.compute_entropy(x, y)
        
        logger.info("计算 Y -> X 方向的熵...")
        c_yx = self.compute_entropy(y, x)
        
        # 比较熵值
        entropy_diff = abs(c_xy - c_yx)
        
        if entropy_diff < 0.01:  # 阈值，表示差异太小
            direction = "undetermined"
            confidence = "low"
            reason = f"熵值差异太小 ({entropy_diff:.3f} < 0.01)"
        elif c_xy < c_yx:
            direction = f"{self.col1} -> {self.col2}"
            confidence = "high" if entropy_diff > 0.1 else "medium"
            reason = f"C({self.col1}->{self.col2}) = {c_xy:.3f} < C({self.col2}->{self.col1}) = {c_yx:.3f}"
        else:
            direction = f"{self.col2} -> {self.col1}"
            confidence = "high" if entropy_diff > 0.1 else "medium"
            reason = f"C({self.col2}->{self.col1}) = {c_yx:.3f} < C({self.col1}->{self.col2}) = {c_xy:.3f}"
        
        # 计算移除的零值对数量（需要重新预处理以获取统计信息）
        original_x = self.data[self.col1].values
        original_y = self.data[self.col2].values
        nan_mask = ~(np.isnan(original_x) | np.isnan(original_y))
        zero_mask = (original_x[nan_mask] != 0) & (original_y[nan_mask] != 0)
        zero_pairs_removed = np.sum(~zero_mask)

        # 整理结果
        self.results.update({
            'direction': direction,
            'confidence': confidence,
            'reason': reason,
            'c_xy': c_xy,
            'c_yx': c_yx,
            'entropy_difference': entropy_diff,
            'sample_size': len(x),
            'zero_pairs_removed': zero_pairs_removed,
            'col1': self.col1,
            'col2': self.col2
        })
        
        logger.info(f"因果方向推断完成: {direction}")
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
        fig.suptitle(f'信息熵因果方向分析: {self.col1} vs {self.col2}', fontsize=16)
        
        # 子图1: 散点图
        axes[0, 0].scatter(x, y, alpha=0.6, s=20)
        axes[0, 0].set_xlabel(f'{self.col1} (标准化)')
        axes[0, 0].set_ylabel(f'{self.col2} (标准化)')
        axes[0, 0].set_title('数据散点图')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 子图2: X->Y 导数
        if 'derivatives' in self.results:
            x_points, derivatives = self.compute_derivatives(x, y)
            axes[0, 1].plot(x_points, derivatives, 'b-', alpha=0.7)
            axes[0, 1].set_xlabel(f'{self.col1}')
            axes[0, 1].set_ylabel(f'd({self.col2})/d({self.col1})')
            axes[0, 1].set_title(f'{self.col1} -> {self.col2} 导数')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 子图3: Y->X 导数
        x_points_yx, derivatives_yx = self.compute_derivatives(y, x)
        axes[1, 0].plot(x_points_yx, derivatives_yx, 'r-', alpha=0.7)
        axes[1, 0].set_xlabel(f'{self.col2}')
        axes[1, 0].set_ylabel(f'd({self.col1})/d({self.col2})')
        axes[1, 0].set_title(f'{self.col2} -> {self.col1} 导数')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 子图4: 熵值比较
        entropy_values = [self.results['c_xy'], self.results['c_yx']]
        labels = [f'C({self.col1}->{self.col2})', f'C({self.col2}->{self.col1})']
        colors = ['lightblue', 'lightcoral']
        
        bars = axes[1, 1].bar(labels, entropy_values, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('熵值')
        axes[1, 1].set_title('熵值比较')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, entropy_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt_path = os.path.join(output_dir, 'entropy_causal_direction.png')
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
            'c_xy': self.results['c_xy'],
            'c_yx': self.results['c_yx'],
            'entropy_difference': self.results['entropy_difference'],
            'sample_size': self.results['sample_size'],
            'zero_pairs_removed': self.results['zero_pairs_removed'],
            'col1': self.results['col1'],
            'col2': self.results['col2']
        }

        results_df = pd.DataFrame([main_results])
        results_path = os.path.join(output_dir, 'entropy_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"主要结果已保存到: {results_path}")
        
        # 保存详细的熵值
        if 'entropy_values' in self.results:
            entropy_df = pd.DataFrame({
                'entropy_value': self.results['entropy_values'],
                'derivative': self.results['derivatives'],
                'x_point': self.results['x_points']
            })
            entropy_path = os.path.join(output_dir, 'entropy_values.csv')
            entropy_df.to_csv(entropy_path, index=False)
            logger.info(f"详细熵值已保存到: {entropy_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于信息熵的因果方向推断')
    parser.add_argument('--data', help='输入CSV文件路径', default='../data/processed/zh_dataset/0105/data_processed.csv')
    parser.add_argument('--col1', help='第一列名称', default='nvidia_smi_ecc_errors_uncorrected_volatile_total_10.104.128.205:9835')
    parser.add_argument('--col2', help='第二列名称', default='nv_inference_request_failure_10.104.128.205:8002')
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
        analyzer = EntropyCausalDirection(data, args.col1, args.col2)
        
        # 执行推断
        results = analyzer.infer_direction()
        
        # 输出结果
        print("\n" + "="*60)
        print("信息熵因果方向推断结果")
        print("="*60)
        print(f"数据文件: {args.data}")
        print(f"分析变量: {args.col1} vs {args.col2}")
        print(f"最终样本数量: {results['sample_size']}")
        print(f"移除的零值对数量: {results['zero_pairs_removed']}")
        print(f"熵值 C({args.col1}->{args.col2}): {results['c_xy']:.6f}")
        print(f"熵值 C({args.col2}->{args.col1}): {results['c_yx']:.6f}")
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