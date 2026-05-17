#!/usr/bin/env python3
"""
快速启发式因果方向推断脚本

基于拟合优度和残差正态性的快速因果方向推断方法。
使用三个指标的综合评分：残差正态性、残差方差和拟合优度。

使用方法:
    python scripts/fast_causal_direction.py --data data/processed/zh_dataset/1116/data_processed.csv --col1 column1 --col2 column2

输出:
    - 因果方向 (X->Y 或 Y->X)
    - 详细的回归分析和残差统计结果
    - 可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import os
from typing import Tuple, Dict, Any
from scipy.stats import normaltest

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FastCausalDirection:
    """
    基于快速启发式的因果方向推断器
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

        logger.info(f"初始化快速启发式因果方向推断器: {col1} vs {col2}")

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

        if len(x) < 10:
            raise ValueError(f"有效数据点数太少 ({len(x)}), 至少需要10个点")

        # 标准化数据
        x_std = (x - np.mean(x)) / np.std(x)
        y_std = (y - np.mean(y)) / np.std(y)

        logger.info(f"数据标准化完成: {self.col1} 均值={np.mean(x):.3f}, 标准差={np.std(x):.3f}")
        logger.info(f"数据标准化完成: {self.col2} 均值={np.mean(y):.3f}, 标准差={np.std(y):.3f}")

        return x_std, y_std

    def linear_regression_analysis(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        执行线性回归分析

        Args:
            x: 自变量（标准化）
            y: 因变量（标准化）

        Returns:
            回归分析结果字典
        """
        # 计算回归系数
        cov_xy = np.cov(x, y)[0, 1]
        var_x = np.var(x)
        var_y = np.var(y)
        
        coef = cov_xy / var_x  # 回归系数
        residuals = y - coef * x  # 残差
        
        # 计算各种指标
        residual_var = np.var(residuals)  # 残差方差
        r_squared = 1 - residual_var / var_y  # 拟合优度R²
        
        # 残差正态性检验 (D'Agostino's K2 test)
        _, p_normal = normaltest(residuals)
        
        logger.info(f"回归系数: {coef:.6f}, R²: {r_squared:.6f}, 残差方差: {residual_var:.6f}, 正态性p值: {p_normal:.6f}")

        return {
            'coefficient': coef,
            'residuals': residuals,
            'residual_var': residual_var,
            'r_squared': r_squared,
            'normality_p': p_normal
        }

    def determine_causal_direction_fast(self) -> Dict[str, Any]:
        """
        基于快速启发式的因果方向推断

        Returns:
            包含推断结果的字典
        """
        logger.info("开始快速启发式因果方向推断...")

        # 预处理数据
        x_std, y_std = self.preprocess_data()

        # 两个方向的回归分析
        logger.info(f"分析 {self.col1} -> {self.col2} 方向...")
        xy_results = self.linear_regression_analysis(x_std, y_std)
        
        logger.info(f"分析 {self.col2} -> {self.col1} 方向...")
        yx_results = self.linear_regression_analysis(y_std, x_std)

        # 计算三个指标并评分
        scores = {'X->Y': 0, 'Y->X': 0}
        score_details = {}

        # 1. 残差的正态性（正确方向的残差应更接近正态）
        p_xy = xy_results['normality_p']
        p_yx = yx_results['normality_p']
        score_details['normality'] = {'X->Y': p_xy, 'Y->X': p_yx}
        if p_xy > p_yx:
            scores['X->Y'] += 1
            score_details['normality']['winner'] = 'X->Y'
        else:
            scores['Y->X'] += 1
            score_details['normality']['winner'] = 'Y->X'

        # 2. 残差的方差（正确方向的残差方差通常更小）
        var_xy = xy_results['residual_var']
        var_yx = yx_results['residual_var']
        score_details['variance'] = {'X->Y': var_xy, 'Y->X': var_yx}
        if var_xy < var_yx:
            scores['X->Y'] += 1
            score_details['variance']['winner'] = 'X->Y'
        else:
            scores['Y->X'] += 1
            score_details['variance']['winner'] = 'Y->X'

        # 3. 拟合优度（R²，更大更好）
        r2_xy = xy_results['r_squared']
        r2_yx = yx_results['r_squared']
        score_details['r_squared'] = {'X->Y': r2_xy, 'Y->X': r2_yx}
        if r2_xy > r2_yx:
            scores['X->Y'] += 1
            score_details['r_squared']['winner'] = 'X->Y'
        else:
            scores['Y->X'] += 1
            score_details['r_squared']['winner'] = 'Y->X'

        # 决定胜者
        winner = max(scores, key=scores.get)
        
        # 计算置信度
        if scores[winner] == 3:
            confidence = "high"
            reason = f"所有三个指标都支持 {winner}"
        elif scores[winner] == 2:
            confidence = "medium"
            loser = 'Y->X' if winner == 'X->Y' else 'X->Y'
            reason = f"两个指标支持 {winner}，一个指标支持 {loser}"
        else:
            confidence = "low"
            reason = f"指标评分相同 ({scores['X->Y']}:{scores['Y->X']})，无法确定方向"
            winner = "undetermined"

        # 映射到实际变量名
        if winner == 'X->Y':
            direction = f"{self.col1} -> {self.col2}"
        elif winner == 'Y->X':
            direction = f"{self.col2} -> {self.col1}"
        else:
            direction = "undetermined"

        # 整理结果
        self.results = {
            'direction': direction,
            'confidence': confidence,
            'reason': reason,
            'scores': scores,
            'score_details': score_details,
            'xy_results': xy_results,
            'yx_results': yx_results,
            'sample_size': len(x_std),
            'col1': self.col1,
            'col2': self.col2
        }

        logger.info(f"快速启发式因果方向推断完成: {direction}")
        logger.info(f"置信度: {confidence}, 原因: {reason}")
        logger.info(f"详细评分: {scores}")

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

        x_std, y_std = self.preprocess_data()

        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'快速启发式因果方向分析: {self.col1} vs {self.col2}', fontsize=16)

        # 子图1: 原始数据散点图
        axes[0, 0].scatter(x_std, y_std, alpha=0.6, s=20)
        # 添加回归线
        coef_xy = self.results['xy_results']['coefficient']
        axes[0, 0].plot(x_std, coef_xy * x_std, 'r-', linewidth=2, label=f'y={coef_xy:.3f}x')
        axes[0, 0].set_xlabel(f'{self.col1} (标准化)')
        axes[0, 0].set_ylabel(f'{self.col2} (标准化)')
        axes[0, 0].set_title(f'{self.col1} -> {self.col2} 回归拟合')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 子图2: Y->X 回归拟合
        axes[0, 1].scatter(y_std, x_std, alpha=0.6, s=20)
        coef_yx = self.results['yx_results']['coefficient']
        axes[0, 1].plot(y_std, coef_yx * y_std, 'r-', linewidth=2, label=f'x={coef_yx:.3f}y')
        axes[0, 1].set_xlabel(f'{self.col2} (标准化)')
        axes[0, 1].set_ylabel(f'{self.col1} (标准化)')
        axes[0, 1].set_title(f'{self.col2} -> {self.col1} 回归拟合')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 子图3: 残差正态性比较
        p_values = [self.results['xy_results']['normality_p'], self.results['yx_results']['normality_p']]
        labels = [f'{self.col1}->{self.col2}', f'{self.col2}->{self.col1}']
        colors = ['lightblue', 'lightcoral']
        bars = axes[0, 2].bar(labels, p_values, color=colors, alpha=0.7)
        axes[0, 2].set_ylabel('正态性检验p值')
        axes[0, 2].set_title('残差正态性比较')
        axes[0, 2].tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, p_values):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.4f}', ha='center', va='bottom')
        axes[0, 2].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
        axes[0, 2].legend()

        # 子图4: 残差方差比较
        variances = [self.results['xy_results']['residual_var'], self.results['yx_results']['residual_var']]
        bars = axes[1, 0].bar(labels, variances, color=colors, alpha=0.7)
        axes[1, 0].set_ylabel('残差方差')
        axes[1, 0].set_title('残差方差比较')
        axes[1, 0].tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, variances):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(variances)*0.01,
                           f'{value:.4f}', ha='center', va='bottom')

        # 子图5: 拟合优度比较
        r_squared = [self.results['xy_results']['r_squared'], self.results['yx_results']['r_squared']]
        bars = axes[1, 1].bar(labels, r_squared, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('R² 拟合优度')
        axes[1, 1].set_title('拟合优度比较')
        axes[1, 1].tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, r_squared):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(r_squared)*0.01,
                           f'{value:.4f}', ha='center', va='bottom')

        # 子图6: 总体评分
        score_labels = ['残差正态性', '残差方差', '拟合优度']
        xy_scores = []
        yx_scores = []
        
        for metric in score_labels:
            if metric in score_details:
                xy_scores.append(1 if score_details[metric]['winner'] == 'X->Y' else 0)
                yx_scores.append(1 if score_details[metric]['winner'] == 'Y->X' else 0)
        
        x_pos = np.arange(len(score_labels))
        width = 0.35
        
        axes[1, 2].bar(x_pos - width/2, xy_scores, width, label=f'{self.col1}->{self.col2}', color='lightblue', alpha=0.7)
        axes[1, 2].bar(x_pos + width/2, yx_scores, width, label=f'{self.col2}->{self.col1}', color='lightcoral', alpha=0.7)
        axes[1, 2].set_xlabel('评估指标')
        axes[1, 2].set_ylabel('得分')
        axes[1, 2].set_title(f'综合评分 (总分: {self.results["scores"]["X->Y"]}:{self.results["scores"]["Y->X"]})')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(score_labels, rotation=45)
        axes[1, 2].legend()
        axes[1, 2].set_ylim(0, 1.2)

        plt.tight_layout()

        # 保存图表
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt_path = os.path.join(output_dir, 'fast_causal_direction.png')
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
            'score_X_to_Y': self.results['scores']['X->Y'],
            'score_Y_to_X': self.results['scores']['Y->X'],
            'xy_residual_var': self.results['xy_results']['residual_var'],
            'yx_residual_var': self.results['yx_results']['residual_var'],
            'xy_r_squared': self.results['xy_results']['r_squared'],
            'yx_r_squared': self.results['yx_results']['r_squared'],
            'xy_normality_p': self.results['xy_results']['normality_p'],
            'yx_normality_p': self.results['yx_results']['normality_p'],
            'sample_size': self.results['sample_size'],
            'col1': self.results['col1'],
            'col2': self.results['col2']
        }

        results_df = pd.DataFrame([main_results])
        results_path = os.path.join(output_dir, 'fast_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"主要结果已保存到: {results_path}")

        # 保存详细评分结果
        score_details_data = []
        for metric, details in self.results['score_details'].items():
            if isinstance(details, dict) and 'X->Y' in details and 'Y->X' in details:
                score_details_data.append({
                    'metric': metric,
                    'score_X_to_Y': details['X->Y'],
                    'score_Y_to_X': details['Y->X'],
                    'winner': details.get('winner', 'N/A')
                })

        if score_details_data:
            details_df = pd.DataFrame(score_details_data)
            details_path = os.path.join(output_dir, 'fast_score_details.csv')
            details_df.to_csv(details_path, index=False)
            logger.info(f"详细评分已保存到: {details_path}")

        # 保存残差数据
        residuals_data = {
            'x_standardized': self.preprocess_data()[0],
            'y_standardized': self.preprocess_data()[1],
            'residuals_xy': self.results['xy_results']['residuals'],
            'residuals_yx': self.results['yx_results']['residuals']
        }
        residuals_df = pd.DataFrame(residuals_data)
        residuals_path = os.path.join(output_dir, 'fast_residuals.csv')
        residuals_df.to_csv(residuals_path, index=False)
        logger.info(f"残差数据已保存到: {residuals_path}")


def determine_causal_direction_fast(X, Y):
    """
    快速启发式方法：基于拟合优度和残差正态性
    
    Args:
        X: 第一个变量数组
        Y: 第二个变量数组
        
    Returns:
        str: 因果方向 ('X->Y' 或 'Y->X')
    """
    # 标准化数据
    X_std = (X - np.mean(X)) / np.std(X)
    Y_std = (Y - np.mean(Y)) / np.std(Y)
    
    # 两个方向的回归
    # X->Y
    coef_xy = np.cov(X_std, Y_std)[0, 1] / np.var(X_std)
    residuals_xy = Y_std - coef_xy * X_std
    
    # Y->X
    coef_yx = np.cov(X_std, Y_std)[0, 1] / np.var(Y_std)
    residuals_yx = X_std - coef_yx * Y_std
    
    # 计算三个指标
    scores = {'X->Y': 0, 'Y->X': 0}
    
    # 1. 残差的正态性（正确方向的残差应更接近正态）
    _, p_xy = normaltest(residuals_xy)
    _, p_yx = normaltest(residuals_yx)
    print(p_xy, p_yx)
    if p_xy > p_yx:
        scores['X->Y'] += 1
    else:
        scores['Y->X'] += 1
    
    # 2. 残差的方差（正确方向的残差方差通常更小）
    var_xy = np.var(residuals_xy)
    var_yx = np.var(residuals_yx)
    if var_xy < var_yx:
        scores['X->Y'] += 1
    else:
        scores['Y->X'] += 1
    
    # 3. 拟合优度（R²，更大更好）
    r2_xy = 1 - var_xy / np.var(Y_std)
    r2_yx = 1 - var_yx / np.var(X_std)
    if r2_xy > r2_yx:
        scores['X->Y'] += 1
    else:
        scores['Y->X'] += 1
    
    # 决定胜者
    return max(scores, key=scores.get)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于快速启发式的因果方向推断')
    parser.add_argument('--data', help='输入CSV文件路径', default='../data/raw/zh_dataset/1116/prometheus_data.csv')
    parser.add_argument('--col1', help='第一列名称', default='network_traffic_in_veth8aa8110')
    parser.add_argument('--col2', help='第二列名称', default='15_IO_block')
    parser.add_argument('--output', default='results', help='输出目录 (默认: results)')
    parser.add_argument('--visualize', action='store_true', help='显示可视化图表')
    parser.add_argument('--verbose', action='store_true', help='详细输出')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # 读取数据
        logger.info(f"读取数据文件: {args.data}")
        data = pd.read_csv(args.data).iloc[280:640]
        logger.info(f"数据形状: {data.shape}")
        logger.info(f"可用列: {list(data.columns)}")

        # 创建推断器
        analyzer = FastCausalDirection(data, args.col1, args.col2)

        # 执行推断
        results = analyzer.determine_causal_direction_fast()

        # 输出结果
        print("\n" + "="*60)
        print("快速启发式因果方向推断结果")
        print("="*60)
        print(f"数据文件: {args.data}")
        print(f"分析变量: {args.col1} vs {args.col2}")
        print(f"样本数量: {results['sample_size']}")
        print("-"*60)
        print("回归分析结果:")
        print(f"  {args.col1} -> {args.col2}: R²={results['xy_results']['r_squared']:.6f}, 残差方差={results['xy_results']['residual_var']:.6f}, 正态性p={results['xy_results']['normality_p']:.6f}")
        print(f"  {args.col2} -> {args.col1}: R²={results['yx_results']['r_squared']:.6f}, 残差方差={results['yx_results']['residual_var']:.6f}, 正态性p={results['yx_results']['normality_p']:.6f}")
        print("-"*60)
        print("综合评分:")
        print(f"  {args.col1} -> {args.col2}: {results['scores']['X->Y']} 分")
        print(f"  {args.col2} -> {args.col1}: {results['scores']['Y->X']} 分")
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