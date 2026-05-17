#!/usr/bin/env python3
"""
Copula因果方向推断脚本

基于Copula理论的因果方向推断方法，用于确定两个变量之间的因果方向。
假设：正确的因果方向中，连接函数（copula）更简单。

使用方法:
    python scripts/copula_causal_direction.py --data data/processed/zh_dataset/1116/data_processed.csv --col1 column1 --col2 column2

输出:
    - 因果方向 (X->Y 或 Y->X)
    - 详细的Copula复杂度计算结果
    - 可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import os
from typing import Tuple, Dict, Any
from scipy.stats import rankdata

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CopulaCausalDirection:
    """
    基于Copula理论的因果方向推断器
    """

    def __init__(self, data: pd.DataFrame, col1: str, col2: str, n_bins: int = 20):
        """
        初始化因果方向推断器

        Args:
            data: 输入数据
            col1: 第一列名称
            col2: 第二列名称
            n_bins: 分箱数量
        """
        self.data = data
        self.col1 = col1
        self.col2 = col2
        self.n_bins = n_bins
        self.results = {}

        # 验证列是否存在
        if col1 not in data.columns:
            raise ValueError(f"列 '{col1}' 不存在于数据中")
        if col2 not in data.columns:
            raise ValueError(f"列 '{col2}' 不存在于数据中")

        logger.info(f"初始化Copula因果方向推断器: {col1} vs {col2}, n_bins={n_bins}")

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理数据：移除NaN值

        Returns:
            清理后的X和Y数组
        """
        # 提取两列数据
        x = self.data[self.col1].values
        y = self.data[self.col2].values

        # 移除NaN值
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x = x[valid_mask]
        y = y[valid_mask]

        logger.info(f"原始数据点数: {len(self.data)}, 有效数据点数: {len(x)}")

        if len(x) < self.n_bins + 5:
            raise ValueError(f"有效数据点数太少 ({len(x)}), 至少需要 {self.n_bins + 5} 个点")

        logger.info(f"数据预处理完成: {self.col1} 范围 [{x.min():.3f}, {x.max():.3f}], "
                   f"{self.col2} 范围 [{y.min():.3f}, {y.max():.3f}]")

        return x, y

    def transform_to_ranks(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将数据转换为秩次（uniform marginal）

        Args:
            x: 自变量数组
            y: 因变量数组

        Returns:
            秩次转换后的u和v数组
        """
        # 转换为秩次，范围在[0,1]
        u = rankdata(x) / (len(x) + 1)
        v = rankdata(y) / (len(y) + 1)

        logger.info(f"秩次转换完成: u范围 [{u.min():.3f}, {u.max():.3f}], "
                   f"v范围 [{v.min():.3f}, {v.max():.3f}]")

        return u, v

    def compute_copula_complexity(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        计算Copula复杂度
        通过条件分布的方差来衡量复杂度

        Args:
            u: 第一个变量的秩次
            v: 第二个变量的秩次

        Returns:
            Copula复杂度值
        """
        # 对u进行分箱
        bins = np.linspace(0, 1, self.n_bins + 1)
        
        complexities = []
        bin_centers = []
        bin_counts = []
        
        for i in range(self.n_bins):
            mask = (u >= bins[i]) & (u < bins[i+1])
            bin_count = np.sum(mask)
            
            if bin_count > 5:  # 至少需要5个样本点
                v_conditional = v[mask]
                # 条件分布的方差
                var_cond = np.var(v_conditional)
                complexities.append(var_cond)
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                bin_counts.append(bin_count)

        logger.info(f"有效分箱数: {len(complexities)}/{self.n_bins}")

        # 存储中间结果
        self.results['complexities'] = complexities
        self.results['bin_centers'] = bin_centers
        self.results['bin_counts'] = bin_counts

        if not complexities:
            logger.warning("没有有效的分箱，返回默认复杂度1.0")
            return 1.0

        avg_complexity = np.mean(complexities)
        logger.info(f"Copula复杂度: {avg_complexity:.6f}")

        return avg_complexity

    def determine_causal_direction_copula(self) -> Dict[str, Any]:
        """
        基于Copula理论的因果方向推断

        Returns:
            包含推断结果的字典
        """
        logger.info("开始Copula因果方向推断...")

        # 预处理数据
        x, y = self.preprocess_data()

        # 转换为秩次
        u, v = self.transform_to_ranks(x, y)

        # 计算两个方向的复杂度
        logger.info(f"计算 {self.col1} -> {self.col2} 方向的复杂度...")
        comp_xy = self.compute_copula_complexity(u, v)  # X->Y

        # 存储第一个方向的中间结果
        temp_results = {}
        if 'complexities' in self.results:
            temp_results['complexities_xy'] = self.results['complexities']
            temp_results['bin_centers_xy'] = self.results['bin_centers']
            temp_results['bin_counts_xy'] = self.results['bin_counts']

        # 清空中间结果以准备第二个方向
        self.results.clear()
        self.results.update(temp_results)

        logger.info(f"计算 {self.col2} -> {self.col1} 方向的复杂度...")
        comp_yx = self.compute_copula_complexity(v, u)  # Y->X

        # 比较复杂度
        complexity_diff = abs(comp_xy - comp_yx)

        # 较小的复杂度对应的方向更可能是因果方向
        if complexity_diff < 0.001:  # 阈值，表示差异太小
            direction = "undetermined"
            confidence = "low"
            reason = f"复杂度差异太小 ({complexity_diff:.6f} < 0.001)"
        elif comp_xy < comp_yx:
            direction = f"{self.col1} -> {self.col2}"
            confidence = "high" if complexity_diff > 0.01 else "medium"
            reason = f"Complexity({self.col1}->{self.col2}) = {comp_xy:.6f} < Complexity({self.col2}->{self.col1}) = {comp_yx:.6f}"
        else:
            direction = f"{self.col2} -> {self.col1}"
            confidence = "high" if complexity_diff > 0.01 else "medium"
            reason = f"Complexity({self.col2}->{self.col1}) = {comp_yx:.6f} < Complexity({self.col1}->{self.col2}) = {comp_xy:.6f}"

        # 整理结果
        final_results = {
            'direction': direction,
            'confidence': confidence,
            'reason': reason,
            'comp_xy': comp_xy,
            'comp_yx': comp_yx,
            'complexity_difference': complexity_diff,
            'sample_size': len(x),
            'n_bins': self.n_bins,
            'col1': self.col1,
            'col2': self.col2
        }

        # 合并所有中间结果
        final_results.update(self.results)
        self.results = final_results

        logger.info(f"Copula因果方向推断完成: {direction}")
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
        u, v = self.transform_to_ranks(x, y)

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Copula因果方向分析: {self.col1} vs {self.col2}', fontsize=16)

        # 子图1: 原始数据散点图
        axes[0, 0].scatter(x, y, alpha=0.6, s=20)
        axes[0, 0].set_xlabel(f'{self.col1}')
        axes[0, 0].set_ylabel(f'{self.col2}')
        axes[0, 0].set_title('原始数据散点图')
        axes[0, 0].grid(True, alpha=0.3)

        # 子图2: 秩次散点图
        axes[0, 1].scatter(u, v, alpha=0.6, s=20)
        axes[0, 1].set_xlabel(f'{self.col1} (秩次)')
        axes[0, 1].set_ylabel(f'{self.col2} (秩次)')
        axes[0, 1].set_title('秩次散点图 (Copula空间)')
        axes[0, 1].grid(True, alpha=0.3)

        # 子图3: X->Y 复杂度分箱
        if 'complexities_xy' in self.results:
            axes[1, 0].bar(self.results['bin_centers_xy'], self.results['complexities_xy'], 
                          width=1.0/self.n_bins, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 0].set_xlabel(f'{self.col1} 秩次分箱')
            axes[1, 0].set_ylabel(f'{self.col2} 条件方差')
            axes[1, 0].set_title(f'{self.col1} -> {self.col2} Copula复杂度')
            axes[1, 0].grid(True, alpha=0.3)

        # 子图4: 复杂度比较
        complexity_values = [self.results['comp_xy'], self.results['comp_yx']]
        labels = [f'Complexity({self.col1}->{self.col2})', f'Complexity({self.col2}->{self.col1})']
        colors = ['lightblue', 'lightcoral']

        bars = axes[1, 1].bar(labels, complexity_values, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Copula复杂度')
        axes[1, 1].set_title('复杂度比较')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, value in zip(bars, complexity_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(complexity_values)*0.01,
                           f'{value:.6f}', ha='center', va='bottom')

        plt.tight_layout()

        # 保存图表
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt_path = os.path.join(output_dir, 'copula_causal_direction.png')
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
            'comp_xy': self.results['comp_xy'],
            'comp_yx': self.results['comp_yx'],
            'complexity_difference': self.results['complexity_difference'],
            'sample_size': self.results['sample_size'],
            'n_bins': self.results['n_bins'],
            'col1': self.results['col1'],
            'col2': self.results['col2']
        }

        results_df = pd.DataFrame([main_results])
        results_path = os.path.join(output_dir, 'copula_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"主要结果已保存到: {results_path}")

        # 保存详细的复杂度结果
        if 'complexities_xy' in self.results and 'complexities_yx' in self.results:
            max_len = max(len(self.results['complexities_xy']), len(self.results['complexities_yx']))
            
            detailed_data = {
                'bin_center_xy': self.results['bin_centers_xy'] + [np.nan] * (max_len - len(self.results['bin_centers_xy'])),
                'complexity_xy': self.results['complexities_xy'] + [np.nan] * (max_len - len(self.results['complexities_xy'])),
                'bin_count_xy': self.results['bin_counts_xy'] + [np.nan] * (max_len - len(self.results['bin_counts_xy'])),
                'complexity_yx': self.results.get('complexities_yx', []) + [np.nan] * (max_len - len(self.results.get('complexities_yx', []))),
            }
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_path = os.path.join(output_dir, 'copula_complexities.csv')
            detailed_df.to_csv(detailed_path, index=False)
            logger.info(f"详细复杂度已保存到: {detailed_path}")


def determine_causal_direction_copula(X, Y, n_bins=20):
    """
    使用Copula理论检测因果方向
    假设：正确的因果方向中，连接函数（copula）更简单
    
    Args:
        X: 第一个变量数组
        Y: 第二个变量数组
        n_bins: 分箱数量
        
    Returns:
        tuple: (因果方向, (复杂度_xy, 复杂度_yx))
    """
    # 转换为秩次
    u = rankdata(X) / (len(X) + 1)
    v = rankdata(Y) / (len(Y) + 1)
    
    def compute_copula_complexity(u, v):
        """通过条件分布计算复杂度"""
        # 对u进行分箱
        bins = np.linspace(0, 1, n_bins + 1)
        
        complexities = []
        for i in range(n_bins):
            mask = (u >= bins[i]) & (u < bins[i+1])
            if np.sum(mask) > 5:
                v_conditional = v[mask]
                # 条件分布的方差
                var_cond = np.var(v_conditional)
                complexities.append(var_cond)
        
        return np.mean(complexities) if complexities else 1.0
    
    # 计算两个方向的复杂度
    comp_xy = compute_copula_complexity(u, v)  # X->Y
    comp_yx = compute_copula_complexity(v, u)  # Y->X
    
    if comp_xy < comp_yx:
        return 'X->Y', (comp_xy, comp_yx)
    else:
        return 'Y->X', (comp_xy, comp_yx)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于Copula理论的因果方向推断')
    parser.add_argument('--data', help='输入CSV文件路径', default='../data/raw/zh_dataset/1116/prometheus_data.csv')
    parser.add_argument('--col1', help='第一列名称', default='network_traffic_in_veth8aa8110')
    parser.add_argument('--col2', help='第二列名称', default='15_IO_block')
    parser.add_argument('--n_bins', type=int, default=30, help='分箱数量 (默认: 20)')
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
        analyzer = CopulaCausalDirection(data, args.col1, args.col2, args.n_bins)

        # 执行推断
        results = analyzer.determine_causal_direction_copula()

        # 输出结果
        print("\n" + "="*60)
        print("Copula因果方向推断结果")
        print("="*60)
        print(f"数据文件: {args.data}")
        print(f"分析变量: {args.col1} vs {args.col2}")
        print(f"样本数量: {results['sample_size']}")
        print(f"分箱数量: {results['n_bins']}")
        print(f"复杂度 Complexity({args.col1}->{args.col2}): {results['comp_xy']:.8f}")
        print(f"复杂度 Complexity({args.col2}->{args.col1}): {results['comp_yx']:.8f}")
        print(f"复杂度差: {results['complexity_difference']:.8f}")
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