#!/usr/bin/env python3
"""
LiNGAM因果方向检验脚本

使用lingam包的DirectLiNGAM方法进行因果方向推断，
并提取因果贡献权重矩阵。

使用方法:
    python scripts/lingam_causal_direction.py --data data/processed/zh_dataset/1116/data_processed.csv --vars network,IO_block,rrt

输出:
    - 因果贡献权重矩阵
    - 因果方向推断
    - 详细的LiNGAM模型结果

示例代码:
    import lingam
    import pandas as pd
    
    model = lingam.DirectLiNGAM()
    model.fit(df[['network', 'IO_block', 'rrt']])
    
    # 提取权重矩阵
    adj = pd.DataFrame(model.adjacency_matrix_, 
                   columns=['net', 'IO', 'rrt'], 
                   index=['net', 'IO', 'rrt'])
    print("LiNGAM Adjacency Matrix (Effect from column to row):")
    print(adj)
    # 重点看 adj.loc['rrt', 'IO'] 的值
"""

import pandas as pd
import numpy as np
import argparse
import logging
import sys
from typing import List, Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LiNGAMCausalDirection:
    """
    基于LiNGAM的因果方向推断器
    """

    def __init__(self, data: pd.DataFrame, var_names: List[str]):
        """
        初始化LiNGAM因果方向推断器

        Args:
            data: 输入数据
            var_names: 变量名称列表
        """
        self.data = data
        self.var_names = var_names
        self.results = {}
        self.model = None
        self.adjacency_matrix = None

        # 验证列是否存在
        missing_vars = [var for var in var_names if var not in data.columns]
        if missing_vars:
            raise ValueError(f"以下变量不存在于数据中: {missing_vars}")

        logger.info(f"初始化LiNGAM因果方向推断器:")
        logger.info(f"  变量: {var_names}")
        logger.info(f"  数据形状: {data.shape}")

    def prepare_data(self) -> np.ndarray:
        """
        准备数据：移除NaN值，标准化

        Returns:
            准备好的numpy数组
        """
        # 提取指定列的数据
        data_subset = self.data[self.var_names]
        
        # 移除包含NaN的行
        data_clean = data_subset.dropna()
        
        logger.info(f"原始数据点数: {len(self.data)}, 有效数据点数: {len(data_clean)}")

        if len(data_clean) < 10:
            raise ValueError(f"有效数据点数太少 ({len(data_clean)}), 至少需要10个点")

        # 标准化数据
        data_array = data_clean.values
        data_standardized = (data_array - np.mean(data_array, axis=0)) / np.std(data_array, axis=0)

        logger.info(f"数据标准化完成，形状: {data_standardized.shape}")

        return data_standardized

    def fit_lingam_model(self) -> None:
        """
        拟合DirectLiNGAM模型

        Returns:
            拟合后的模型对象
        """
        try:
            import lingam
            logger.info("开始拟合DirectLiNGAM模型...")

            # 准备数据
            data_array = self.prepare_data()

            # 创建并拟合模型
            self.model = lingam.DirectLiNGAM(random_state=42)
            self.model.fit(data_array)

            # 获取权重矩阵
            self.adjacency_matrix = self.model.adjacency_matrix_

            logger.info(f"DirectLiNGAM模型拟合完成")
            logger.info(f"权重矩阵形状: {self.adjacency_matrix.shape}")

        except ImportError:
            logger.error("lingam未安装，无法执行LiNGAM因果方向推断")
            logger.info("请安装: pip install lingam")
            sys.exit(1)
        except Exception as e:
            logger.error(f"LiNGAM模型拟合失败: {e}")
            raise

    def get_causal_directions(self) -> Dict[str, Any]:
        """
        从权重矩阵推断因果方向

        Returns:
            因果方向字典
        """
        if self.adjacency_matrix is None:
            raise ValueError("模型尚未拟合，请先调用fit_lingam_model()")

        n_vars = len(self.var_names)
        causal_directions = []
        causal_strengths = []

        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    effect_strength = abs(self.adjacency_matrix[j, i])  # 从j到i的因果强度
                    
                    if effect_strength > 1e-10:  # 避免数值误差
                        direction = f"{self.var_names[j]} -> {self.var_names[i]}"
                        causal_directions.append(direction)
                        causal_strengths.append(effect_strength)

        # 排序并获取最强的因果方向
        if causal_directions:
            sorted_results = sorted(zip(causal_directions, causal_strengths), 
                                     key=lambda x: x[1], reverse=True)
            
            top_direction, top_strength = sorted_results[0]
            
            logger.info(f"最强因果方向: {top_direction} (强度: {top_strength:.6f})")
            logger.info(f"找到 {len(causal_directions)} 个潜在的因果方向")

        return {
            'all_directions': causal_directions,
            'all_strengths': causal_strengths,
            'top_direction': top_direction if causal_directions else None,
            'top_strength': top_strength if causal_directions else None,
            'n_directions': len(causal_directions)
        }

    def print_adjacency_matrix(self) -> None:
        """
        打印权重矩阵
        """
        if self.adjacency_matrix is None:
            raise ValueError("模型尚未拟合，请先调用fit_lingam_model()")

        # 创建DataFrame以便更好地显示
        adj_df = pd.DataFrame(
            self.adjacency_matrix,
            columns=self.var_names,
            index=self.var_names
        )

        print("\n" + "="*80)
        print("LiNGAM Adjacency Matrix (Effect from column to row):")
        print("="*80)
        print("注: 正值表示列变量影响行变量")
        print("    负值表示行变量影响列变量")
        print("-"*80)
        print(adj_df.round(6))
        print("="*80)

        # 高亮显示重要的因果权重
        print("\n重要的因果权重 (> 0.1):")
        important_edges = []
        
        for i in range(len(self.var_names)):
            for j in range(len(self.var_names)):
                if i != j:
                    weight = abs(self.adjacency_matrix[j, i])
                    if weight > 0.1:
                        direction = f"{self.var_names[j]} -> {self.var_names[i]}"
                        important_edges.append((direction, weight))
        
        if important_edges:
            important_edges.sort(key=lambda x: x[1], reverse=True)
            for edge, weight in important_edges:
                print(f"  {edge}: {weight:.6f}")
        else:
            print("  未找到权重 > 0.1 的因果关系")

    def analyze_causal_structure(self) -> Dict[str, Any]:
        """
        分析因果结构

        Returns:
            结构分析结果
        """
        if self.adjacency_matrix is None:
            raise ValueError("模型尚未拟合，请先调用fit_lingam_model()")

        n_vars = len(self.var_names)
        
        # 计算每个变量的入度和出度
        in_degrees = np.zeros(n_vars)
        out_degrees = np.zeros(n_vars)

        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    # 从j到i的边
                    if abs(self.adjacency_matrix[j, i]) > 1e-10:
                        in_degrees[i] += 1
                        out_degrees[j] += 1

        # 分析因果层次
        var_stats = []
        for i, var_name in enumerate(self.var_names):
            stats = {
                'variable': var_name,
                'in_degree': int(in_degrees[i]),
                'out_degree': int(out_degrees[i]),
                'total_connections': int(in_degrees[i] + out_degrees[i])
            }
            
            # 判断因果角色
            if out_degrees[i] > in_degrees[i]:
                stats['role'] = 'cause'  # 更多的输出连接，更可能是原因
            elif in_degrees[i] > out_degrees[i]:
                stats['role'] = 'effect'  # 更多的输入连接，更可能是结果
            else:
                stats['role'] = 'neutral'
            
            var_stats.append(stats)

        # 统计信息
        # n_causal_edges = np.sum(np.abs(self.adjacency_matrix) > 1e-10) - np.diag(np.abs(self.adjacency_matrix) > 1e-10)
        max_in_degree = np.max(in_degrees)
        max_out_degree = np.max(out_degrees)

        return {
            'variable_stats': var_stats,
            'max_in_degree': int(max_in_degree),
            'max_out_degree': int(max_out_degree)
        }

    def run_lingam_analysis(self) -> Dict[str, Any]:
        """
        运行完整的LiNGAM因果分析

        Returns:
            分析结果字典
        """
        logger.info("开始LiNGAM因果方向分析...")

        # 拟合模型
        self.fit_lingam_model()

        # 打印权重矩阵
        self.print_adjacency_matrix()

        # 获取因果方向
        causal_directions = self.get_causal_directions()

        # 分析因果结构
        structure_analysis = self.analyze_causal_structure()

        # 整理结果
        self.results = {
            'model_type': 'DirectLiNGAM',
            'variables': self.var_names,
            'n_samples': len(self.prepare_data()),
            'adjacency_matrix': self.adjacency_matrix.tolist(),
            'causal_directions': causal_directions,
            'structure_analysis': structure_analysis,
            'random_state': 42
        }

        logger.info("LiNGAM因果方向分析完成")

        return self.results

    def print_results(self) -> None:
        """
        打印分析结果
        """
        if not self.results:
            logger.warning("没有结果可打印")
            return

        print("\n" + "="*80)
        print("LiNGAM因果方向分析结果")
        print("="*80)
        print(f"模型类型: {self.results['model_type']}")
        print(f"变量数量: {len(self.results['variables'])}")
        print(f"样本数量: {self.results['n_samples']}")
        print("-"*80)

        # 因果方向结果
        print("因果方向分析:")
        if self.results['causal_directions']['top_direction']:
            print(f"  最强因果方向: {self.results['causal_directions']['top_direction']}")
            print(f"  因果强度: {self.results['causal_directions']['top_strength']:.6f}")
        print(f"  总因果方向数: {self.results['causal_directions']['n_directions']}")
        print("-"*80)

        # 结构分析
        print("因果结构分析:")
        print(f"  最大入度: {self.results['structure_analysis']['max_in_degree']}")
        print(f"  最大出度: {self.results['structure_analysis']['max_out_degree']}")

        print("\n变量角色分析:")
        for var_stat in self.results['structure_analysis']['variable_stats']:
            print(f"  {var_stat['variable']}: ")
            print(f"    入度: {var_stat['in_degree']}, 出度: {var_stat['out_degree']}")
            print(f"    角色: {var_stat['role']}")

    def save_results(self, output_file: str = None) -> None:
        """
        保存结果到文件

        Args:
            output_file: 输出文件路径
        """
        if not self.results:
            logger.warning("没有结果可保存")
            return

        if output_file:
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"结果已保存到: {output_file}")
            
            # 单独保存权重矩阵为CSV
            adj_df = pd.DataFrame(
                self.results['adjacency_matrix'],
                columns=self.results['variables'],
                index=self.results['variables']
            )
            adj_file = output_file.replace('.json', '_adjacency.csv')
            adj_df.to_csv(adj_file)
            logger.info(f"权重矩阵已保存到: {adj_file}")
        else:
            # 默认保存
            import json
            with open('lingam_results.json', 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info("结果已保存到: lingam_results.json")
            
            # 保存权重矩阵
            adj_df = pd.DataFrame(
                self.results['adjacency_matrix'],
                columns=self.results['variables'],
                index=self.results['variables']
            )
            adj_df.to_csv('lingam_adjacency_matrix.csv')
            logger.info("权重矩阵已保存到: lingam_adjacency_matrix.csv")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于LiNGAM的因果方向检验')
    parser.add_argument('--data', help='输入CSV文件路径', default='../data/processed/zh_dataset/0105/data_processed.csv')
    parser.add_argument('--vars', help='变量名称，用逗号分隔', default='nv_inference_request_failure_10.104.128.205:8002,nvidia_smi_ecc_errors_uncorrected_volatile_total_10.104.128.205:9835,worker_cpu_usage_percent_10.104.128.205:9093,full_request_duration_ms_new_10.104.128.205:9093,triton_inference_duration_ms_10.104.128.205:9093')
    parser.add_argument('--output', help='输出文件路径 (可选)')
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

        # 解析变量名
        var_names = [var.strip() for var in args.vars.split(',')]
        logger.info(f"分析变量: {var_names}")

        # 创建分析器
        analyzer = LiNGAMCausalDirection(data, var_names)

        # 运行分析
        results = analyzer.run_lingam_analysis()

        # 打印结果
        analyzer.print_results()

        # 保存结果
        if args.output:
            analyzer.save_results(args.output)
        else:
            analyzer.save_results()

    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise


if __name__ == "__main__":
    main()