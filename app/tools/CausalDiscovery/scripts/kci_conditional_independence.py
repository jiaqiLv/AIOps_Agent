#!/usr/bin/env python3
"""
KCI条件独立性检验脚本

使用causal-learn的CIT类和KCI方法检验两个变量在给定条件集下的条件独立性。
基于核条件独立性检验，适用于连续变量和混合类型变量。

使用方法:
    # 准备数据
    import pandas as pd
    import numpy as np
    from causallearn.utils.PC import PC
    from causallearn.utils.cit import CIT
    
    # 示例数据
    data = df[['network', 'IO_block', 'rrt']].values
    
    # 创建CIT对象
    kci_obj = CIT(data, 'kci')
    
    # 检验：在已知 network (索引 0) 的条件下，IO (1) 和 rrt (2) 是否独立
    p_value = kci_obj(1, 2, [0])
    print(f"KCI Non-linear p-value: {p_value:.4f}")
    
    # p < 0.05 则说明边存在，PC 的 Fisher-Z 失效了

使用命令行:
    python scripts/kci_conditional_independence.py --data data/processed/zh_dataset/1116/data_processed.csv --col1 network --col2 IO_block --condition_vars rrt

参数:
    --data: 输入CSV文件路径
    --col1: 第一个变量列名 (用于X)
    --col2: 第二个变量列名 (用于Y)
    --condition_vars: 条件变量列名，用逗号分隔
    --alpha: 显著性水平 (默认: 0.05)
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


class KCIConditionalIndependence:
    """
    基于causal-learn CIT和KCI的条件独立性检验器
    """

    def __init__(self, data: pd.DataFrame, col1: str, col2: str, 
                 condition_vars: List[str], alpha: float = 0.05):
        """
        初始化KCI条件独立性检验器

        Args:
            data: 输入数据
            col1: 第一个变量列名 (用于X)
            col2: 第二个变量列名 (用于Y)
            condition_vars: 条件变量列名列表 (用于Z)
            alpha: 显著性水平
        """
        self.data = data
        self.col1 = col1
        self.col2 = col2
        self.condition_vars = condition_vars
        self.alpha = alpha
        self.results = {}
        
        # 获取所有需要的列
        self.all_vars = [col1, col2] + condition_vars
        self.data_vars = [col1, col2] + condition_vars

        # 验证列是否存在
        missing_cols = [col for col in self.all_vars if col not in data.columns]
        if missing_cols:
            raise ValueError(f"以下列不存在于数据中: {missing_cols}")

        logger.info(f"初始化KCI条件独立性检验器:")
        logger.info(f"  变量X ({self.col1}): 索引 0")
        logger.info(f"  变量Y ({self.col2}): 索引 1")
        logger.info(f"  条件变量: {[(var, i+2) for i, var in enumerate(self.condition_vars)]}")
        logger.info(f"  显著性水平: {alpha}")

    def prepare_data(self) -> np.ndarray:
        """
        准备数据，按照CIT要求的格式
        
        Returns:
            准备好的numpy数组
        """
        # 提取需要的列并转换
        data_matrix = self.data[self.all_vars].values
        
        # 移除包含NaN的行
        valid_mask = ~np.isnan(data_matrix).any(axis=1)
        data_clean = data_matrix[valid_mask]
        
        logger.info(f"原始数据点数: {len(self.data)}")
        logger.info(f"有效数据点数: {len(data_clean)}")
        
        if len(data_clean) < 10:
            raise ValueError(f"有效数据点数太少 ({len(data_clean)}), 至少需要10个点")
        
        return data_clean

    def get_variable_indices(self) -> List[int]:
        """
        获取各变量的索引，用于CIT方法
        
        Returns:
            索引列表 [X_idx, Y_idx, Z1_idx, Z2_idx, ...]
        """
        # 按照CIT的约定：
        # X为第一个变量，索引为0
        # Y为第二个变量，索引为1
        # 条件变量从索引2开始
        indices = [0, 1]  # X和Y的索引
        
        # 条件变量的索引从2开始
        for i in range(len(self.condition_vars)):
            indices.append(2 + i)
        
        return indices

    def kci_test(self) -> Dict[str, Any]:
        """
        执行KCI条件独立性检验

        Returns:
            检验结果字典
        """
        try:
            from causallearn.utils.cit import CIT
            logger.info("使用causal-learn的CIT类执行KCI检验")
            
            # 准备数据
            data_matrix = self.prepare_data()
            
            # 获取变量索引
            indices = self.get_variable_indices()
            
            # 创建CIT对象，指定使用KCI方法
            cit = CIT(data_matrix,'kci')
            
            # 获取变量索引
            X_idx = indices[0]  # col1
            Y_idx = indices[1]  # col2
            Z_indices = indices[2:]  # 条件变量
            
            logger.info(f"执行检验: X_idx={X_idx} ({self.col1}), Y_idx={Y_idx} ({self.col2})")
            logger.info(f"条件变量索引: {[(self.condition_vars[i], Z_indices[i]) for i in range(len(Z_indices))]}")
            logger.info(f"完整条件集: {Z_indices}")
            
            # 执行条件独立性检验
            # 检验H0: X ⊥ Y | Z
            p_value = cit(X_idx, Y_idx, Z_indices)
            
            result = {
                'p_value': p_value,
                'method': 'KCI (causal-learn CIT)',
                'kci_method': 'kci',
                'alpha': self.alpha,
                'X_idx': X_idx,
                'Y_idx': Y_idx,
                'Z_indices': Z_indices,
                'conditioning_vars': self.condition_vars
            }
            
            logger.info(f"KCI检验完成: p-value={p_value:.6f}")
            
            return result
            
        except ImportError:
            logger.error("causal-learn未安装，无法执行KCI检验")
            logger.info("请安装: pip install causal-learn")
            sys.exit(1)
        except Exception as e:
            logger.error(f"KCI检验失败: {e}")
            raise

    def test_independence(self) -> Dict[str, Any]:
        """
        执行条件独立性检验

        Returns:
            检验结果字典
        """
        logger.info("开始KCI条件独立性检验...")
        logger.info(f"假设检验: H0: {self.col1} ⊥ {self.col2} | {', '.join(self.condition_vars)}")
        
        # 执行KCI检验
        result = self.kci_test()

        # 添加数据信息
        result.update({
            'n_samples': len(self.prepare_data()),
            'n_conditions': len(self.condition_vars),
            'col1': self.col1,
            'col2': self.col2,
            'condition_vars': self.condition_vars,
            'alpha': self.alpha,
            'is_independent': result['p_value'] > self.alpha,
            'conclusion': '独立' if result['p_value'] > self.alpha else '不独立',
            'edge_exists': result['p_value'] <= self.alpha  # p < 0.05 说明边存在
        })

        self.results = result
        logger.info(f"条件独立性检验完成: {result['conclusion']}")
        logger.info(f"p-value: {result['p_value']:.6f} (α={self.alpha})")
        logger.info(f"边存在: {result['edge_exists']}")
        
        return result

    def save_results(self, output_file: str = None):
        """
        保存结果到文件

        Args:
            output_file: 输出文件路径
        """
        if not self.results:
            logger.warning("没有结果可保存")
            return

        if output_file:
            if output_file.endswith('.json'):
                import json
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2, ensure_ascii=False)
                logger.info(f"结果已保存到: {output_file}")
            else:
                # 保存为CSV
                import pandas as pd
                results_df = pd.DataFrame([self.results])
                output_path = output_file if output_file.endswith('.csv') else f"{output_file}.csv"
                results_df.to_csv(output_path, index=False)
                logger.info(f"结果已保存到: {output_path}")
        else:
            # 默认保存为CSV
            import pandas as pd
            results_df = pd.DataFrame([self.results])
            output_path = "kci_conditional_independence_results.csv"
            results_df.to_csv(output_path, index=False)
            logger.info(f"结果已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于causal-learn CIT和KCI的条件独立性检验')
    parser.add_argument('--data', help='输入CSV文件路径',
                        default='../data/processed/zh_dataset/0922/data_processed.csv')
    parser.add_argument('--col1', help='第一个变量列名', default='ai_request_avg_duration')
    parser.add_argument('--col2', help='第二个变量列名', default='202_IO_Utilization')
    parser.add_argument('--condition_vars', help='条件变量列名，用逗号分隔', default='202_Network_Traffic_out')
    parser.add_argument('--alpha', type=float, default=0.05, help='显著性水平 (默认: 0.05)')
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

        # 解析条件变量
        condition_vars = [col.strip() for col in args.condition_vars.split(',')]
        
        # 创建检验器
        tester = KCIConditionalIndependence(
            data=data,
            col1=args.col1,
            col2=args.col2,
            condition_vars=condition_vars,
            alpha=args.alpha
        )

        # 执行检验
        results = tester.test_independence()

        # 保存结果
        if args.output:
            tester.save_results(args.output)
        else:
            tester.save_results()

    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise


if __name__ == "__main__":
    main()