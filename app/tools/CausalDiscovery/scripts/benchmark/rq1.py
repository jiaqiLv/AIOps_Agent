"""
RQ1: 因果发现算法性能评估脚本

该脚本用于评估不同因果发现算法在合成数据集上的性能，
主要回答论文中的Research Question 1: 因果发现的性能如何？

支持的算法: PC, FCI, GES, LiNGAM系列, Granger, PCMCI, NOTEARS, NTLR等
支持的数据集: CIRCA (10/50节点), RCD (10/50节点), Causil (10/50节点)
评估指标: F1分数, F1-Skeleton分数, SHD (结构汉明距离)
"""

import argparse
import glob
import math
import itertools
import json
import os
import pickle
import warnings
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
from os.path import abspath, basename, dirname, exists, join

# 忽略警告信息，保持输出清洁
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.score.LocalScoreFunction import local_score_BIC
from tqdm import tqdm

# 导入项目内部模块
from RCAEval.benchmark.evaluation import Evaluator
from RCAEval.benchmark.metrics import F1, SHD, F1_Skeleton
from RCAEval.classes.graph import MemoryGraph, Node
from RCAEval.graph_heads import finalize_directed_adj
from RCAEval.io.time_series import drop_constant, drop_extra, drop_time
from RCAEval.utility import (
    dump_json,
    download_syn_rcd_dataset,
    download_syn_circa_dataset,
    download_syn_causil_dataset,
    is_py310,
    load_json,
)

# 根据Python版本条件导入不同的因果发现算法
# Python 3.10支持更多先进的算法
if is_py310():
    # 约束基础的因果发现算法
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.search.ConstraintBased.PC import pc
    # LiNGAM系列算法 (线性非高斯无环模型)
    from causallearn.search.FCMBased.lingam import DirectLiNGAM, ICALiNGAM, VARLiNGAM
    # 分数基础的算法
    from causallearn.search.ScoreBased.GES import ges
    # 条件独立性检验方法
    from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
    # 自定义实现的算法
    from RCAEval.graph_construction.granger import granger
    from RCAEval.graph_construction.pcmci import pcmci
    from RCAEval.graph_construction.cmlp import cmlp
    try:
        # 深度学习基础的因果发现算法
        from RCAEval.graph_construction.dag_gnn import dag_gnn
        from RCAEval.graph_construction.dag_gnn import notears_low_rank as ntlr
        from RCAEval.graph_construction.notears import notears
    except Exception as e:
        print(f"导入深度学习算法失败: {e}")

else:
    # Python 3.8只支持FGES算法 (快速贪婪等价搜索)
    from RCAEval.graph_construction.fges import fges

# 所有可用的因果发现方法列表
AVAILABLE_METHODS = sorted(
    [
        "pc",           # Peter-Clark算法
        "ppc",          # Parallel PC算法
        "pcmci",        # PCMCI算法 (时间序列)
        "fci",          # Fast Causal Inference
        "fges",         # Fast Greedy Equivalence Search
        "notears",      # NOTEARS算法
        "ntlr",         # NOTEARS Low Rank
        "DirectLiNGAM", # Direct LiNGAM
        "VARLiNGAM",    # Vector Autoregression LiNGAM
        "ICALiNGAM",    # ICA-based LiNGAM
        "ges",          # Greedy Equivalence Search
        "granger",      # Granger因果检验
    ]
)


def parse_args():
    """
    解析命令行参数

    Returns:
        args: 包含所有命令行参数的命名空间对象
    """
    parser = argparse.ArgumentParser(description="RCAEval RQ1: 因果发现算法性能评估")

    # 数据集相关参数
    parser.add_argument("--dataset", type=str, default="data",
                       help="选择数据集",
                       choices=["circa10", "circa50", "rcd10", "rcd50", "causil10", "causil50"])
    parser.add_argument("--method", type=str, help="因果发现方法名称")
    parser.add_argument("--length", type=int, default=None,
                       help="时间序列长度 (用于RQ4实验)")
    parser.add_argument("--test", action="store_true",
                       help="执行烟雾测试，只处理少量数据而不完整运行")

    args = parser.parse_args()

    # 验证方法是否可用
    if args.method not in globals():
        raise ValueError(f"方法 {args.method} 未定义。可用方法: {AVAILABLE_METHODS}")

    # 验证数据集是否可用
    if args.dataset not in ["circa10", "circa50", "rcd10", "rcd50", "causil10", "causil50"]:
        print(f"数据集 {args.dataset} 未定义。可用数据集: circa10, circa50, rcd10, rcd50, causil10, causil50")
        exit()

    return args


# 解析命令行参数
args = parse_args()

# 数据集路径映射表
# 将数据集名称映射到实际的数据目录路径
DATASET_MAP = {
    "circa10": "data/syn_circa/10",    # CIRCA数据集，10个节点
    "circa50": "data/syn_circa/50",    # CIRCA数据集，50个节点
    "causil10": "data/syn_causil/10",  # Causil数据集，10个节点
    "causil50": "data/syn_causil/50",  # Causil数据集，50个节点
    "rcd10": "data/syn_rcd/10",        # RCD数据集，10个节点
    "rcd50": "data/syn_rcd/50"         # RCD数据集，50个节点
}
dataset = DATASET_MAP[args.dataset]

# 创建临时输出目录
output_path = TemporaryDirectory().name
print(output_path)
report_path = join(output_path, "report.xlsx")  # Excel报告路径 (未使用)
result_path = join(output_path, "results")       # 结果保存路径
os.makedirs(result_path, exist_ok=True)

# 发现所有数据文件路径
# 递归搜索指定数据集目录下的所有data.csv文件
data_paths = list(glob.glob(os.path.join(dataset, "**/data.csv"), recursive=True))

# 如果是测试模式，只处理前2个数据文件以节省时间
if args.test is True:
    data_paths = data_paths[:2]


def evaluate():
    """
    评估因果发现算法的性能

    该函数遍历所有处理后的结果文件，与真实因果图进行比较，
    计算各种评估指标并输出平均性能。

    评估指标:
    - F1-Score: 精确率和召回率的调和平均
    - F1-Skeleton: 骨架图的F1分数 (忽略边的方向)
    - SHD: 结构汉明距离 (预测图与真实图的差异)
    """
    # 初始化评估数据存储字典
    eval_data = {
        "Case": [],           # 案例名称
        "Precision": [],      # 精确率
        "Recall": [],         # 召回率
        "F1-Score": [],       # F1分数
        "Precision-Skel": [],# 骨架图精确率
        "Recall-Skel": [],    # 骨架图召回率
        "F1-Skel": [],        # 骨架图F1分数
        "SHD": [],            # 结构汉明距离
    }

    # 遍历所有数据路径，评估每个案例的结果
    for data_path in data_paths:
        # 解析不同数据集的路径结构，提取索引信息
        if "circa" in data_path or "rcd" in data_path:
            # CIRCA和RCD数据集的路径结构: .../num_nodes/graph_idx/case_idx/data.csv
            num_node = int(basename(dirname(dirname(dirname(dirname(data_path))))))
            graph_idx = int(basename(dirname(dirname(dirname(data_path)))))
            case_idx = int(basename(dirname(data_path)))

        if "causil" in data_path:
            # Causil数据集的路径结构不同
            graph_idx = int(basename(dirname(data_path))[-1:])  # 从目录名提取最后一个字符作为图索引
            case_idx = 0  # Causil数据集只有一个案例

        # ===== 读取估计的因果图结果 =====
        est_graph_name = f"{graph_idx}_{case_idx}_est_graph.json"
        est_graph_path = join(result_path, est_graph_name)

        # 如果结果文件不存在，跳过该案例
        if not exists(est_graph_path):
            continue
        est_graph = MemoryGraph.load(est_graph_path)

        # ===== 读取真实因果图 =====
        if "circa" in data_path:
            # CIRCA数据集: 真实图存储为JSON格式
            true_graph_path = join(dirname(dirname(dirname(data_path))), "graph.json")
            true_graph = MemoryGraph.load(true_graph_path)

        if "causil" in data_path:
            # Causil数据集: 真实图存储为pickle格式
            dag_gt = pickle.load(open(join(dirname(data_path), "DAG.gpickle"), "rb"))
            true_graph = MemoryGraph(dag_gt)

        if "rcd" in data_path:
            # RCD数据集: 有两个真实图文件
            dag_gt = pickle.load(
                open(join(dirname(dirname(dirname(data_path))), "g_graph.pkl"), "rb")
            )
            true_graph = MemoryGraph(dag_gt)
            # 优先使用JSON格式的真实图
            true_graph = MemoryGraph.load(
                join(dirname(dirname(dirname(data_path))), "true_graph.json")
            )

        # 计算评估指标
        e = F1(true_graph, est_graph)           # 完整图的F1分数
        e_skel = F1_Skeleton(true_graph, est_graph)  # 骨架图的F1分数
        shd = SHD(true_graph, est_graph)        # 结构汉明距离

        # 存储评估结果
        eval_data["Case"].append(est_graph_name)
        eval_data["Precision"].append(e["precision"])
        eval_data["Recall"].append(e["recall"])
        eval_data["F1-Score"].append(e["f1"])
        eval_data["Precision-Skel"].append(e_skel["precision"])
        eval_data["Recall-Skel"].append(e_skel["recall"])
        eval_data["F1-Skel"].append(e_skel["f1"])
        eval_data["SHD"].append(shd)

    # 计算所有案例的平均性能指标
    avg_precision = np.mean(eval_data["Precision"])
    avg_recall = np.mean(eval_data["Recall"])
    avg_f1 = np.mean(eval_data["F1-Score"])
    avg_precision_skel = np.mean(eval_data["Precision-Skel"])
    avg_recall_skel = np.mean(eval_data["Recall-Skel"])
    avg_f1_skel = np.mean(eval_data["F1-Skel"])
    avg_shd = np.mean(eval_data["SHD"])

    # 输出评估结果 (与论文中的格式保持一致)
    print(f"F1:   {avg_f1:.2f}")      # 完整图F1分数
    print(f"F1-S: {avg_f1_skel:.2f}")  # 骨架图F1分数
    print(f"SHD:  {math.floor(avg_shd)}")  # 结构汉明距离 (向下取整)



def process(data_path):
    """
    处理单个数据文件，执行因果发现算法并保存结果

    Args:
        data_path (str): 数据文件的路径

    该函数执行以下步骤:
    1. 解析数据路径，提取图索引和案例索引
    2. 读取和预处理数据
    3. 执行指定的因果发现算法
    4. 保存估计的因果图
    """
    # 解析不同数据集的路径结构，提取索引信息
    if "circa" in data_path:
        # CIRCA数据集路径结构: .../num_nodes/graph_idx/case_idx/data.csv
        num_node = int(basename(dirname(dirname(dirname(dirname(data_path))))))
        graph_idx = int(basename(dirname(dirname(dirname(data_path)))))
        case_idx = int(basename(dirname(data_path)))

    if "causil" in data_path:
        # Causil数据集路径结构: .../num_nodes/graph_idx/data.csv
        num_node = int(basename(dirname(dirname(dirname(data_path)))).split("_")[0])
        graph_idx = int(basename(dirname(data_path))[-1:])
        case_idx = 0  # Causil数据集只有一个案例

    if "rcd" in data_path:
        # RCD数据集路径结构: .../num_nodes/graph_idx/case_idx/data.csv
        num_node = int(basename(dirname(dirname(dirname(dirname(data_path))))))
        graph_idx = int(basename(dirname(dirname(dirname(data_path)))))
        case_idx = int(basename(dirname(data_path)))

    # 读取数据文件
    if "circa" in data_path:
        # CIRCA数据集没有列标题，需要手动生成
        data = pd.read_csv(data_path, header=None)
        data.header = list(map(str, range(0, data.shape[1])))
    else:
        # 其他数据集有列标题
        data = pd.read_csv(data_path)

    # ===== 数据预处理 =====
    # 前向填充缺失值 (用前一个有效值填充NaN)
    data = data.fillna(method="ffill")
    # 剩余的NaN用0填充
    data = data.fillna(value=0)
    # 转换为numpy数组并取绝对值，确保数据为正数
    np_data = np.absolute(data.to_numpy().astype(float))

    # 如果指定了数据长度，截取前N行数据 (用于RQ4实验)
    if args.length is not None:
        np_data = np_data[: args.length, :]

    # 初始化邻接矩阵和图对象
    adj = []
    G = None

    # 记录开始时间，用于计算算法运行时间
    st = datetime.now()

    try:
        # ===== 执行因果发现算法 =====
        if args.method == "pc":
            # PC算法: 基于约束的因果发现
            adj = pc(
                np_data,
                stable=False,        # 不使用稳定版本
                show_progress=False, # 不显示进度条
            ).G.graph

        elif args.method == "fci":
            # FCI算法: 快速因果推断，适用于存在潜在变量的情况
            adj = fci(
                np_data,
                show_progress=False, # 不显示进度条
                verbose=False,       # 不输出详细信息
            )[0].graph

        elif args.method == "fges":
            # FGES算法: 快速贪婪等价搜索
            adj = fges(pd.DataFrame(np_data))

        elif args.method == "ICALiNGAM":
            # ICA-LiNGAM: 基于独立成分分析的线性非高斯无环模型
            model = ICALiNGAM()
            model.fit(np_data)
            adj = model.adjacency_matrix_
            # 转换为二进制邻接矩阵
            adj = adj.astype(bool).astype(int)

        elif args.method == "VARLiNGAM":
            # VAR-LiNGAM: 向量自回归LiNGAM (未实现)
            raise NotImplementedError

        elif args.method == "DirectLiNGAM":
            # Direct-LiNGAM: 直接LiNGAM算法
            model = DirectLiNGAM()
            model.fit(np_data)
            adj = model.adjacency_matrix_
            adj = adj.astype(bool).astype(int)

        elif args.method == "ges":
            # GES算法: 贪婪等价搜索
            record = ges(np_data)
            adj = record["G"].graph

        elif args.method == "granger":
            # Granger因果检验: 适用于时间序列数据
            adj = granger(data)

        elif args.method == "pcmci":
            # PCMCI算法: 适用于时间序列的因果发现
            adj = pcmci(pd.DataFrame(np_data))

        elif args.method == "ntlr":
            # NOTEARS Low Rank: 低秩约束的NOTEARS算法
            adj = ntlr(pd.DataFrame(np_data))

        else:
            raise ValueError(f"方法 {args.method} 未定义。可用方法: {AVAILABLE_METHODS}")

        # ===== 构建并保存估计的因果图 =====
        if "circa" in data_path:
            # CIRCA数据集使用SIM_0, SIM_1, ...作为节点名称
            est_graph = MemoryGraph.from_adj(
                adj, nodes=[Node("SIM", str(i)) for i in range(len(adj))]
            )
        else:
            # 其他数据集使用数据列名作为节点名称
            est_graph = MemoryGraph.from_adj(adj, nodes=data.columns.to_list())

        # 保存估计的因果图到JSON文件
        est_graph.dump(join(result_path, f"{graph_idx}_{case_idx}_est_graph.json"))

    except Exception as e:
        # 处理算法执行失败的情况
        raise e
        print(f"方法 {args.method} 在数据文件 {data_path} 上执行失败")
        # 创建空的图对象并标记为失败
        est_graph = MemoryGraph.from_adj([], nodes=[])
        est_graph.dump(join(result_path, f"{graph_idx}_{case_idx}_failed.json"))


# ===== 主执行流程 =====

# 记录开始时间，用于计算总体运行时间
start_time = datetime.now()

# 遍历所有数据文件，执行因果发现算法
# 使用tqdm显示进度条
for data_path in tqdm(data_paths):
    output = process(data_path)

# 记录结束时间
end_time = datetime.now()
time_taken = end_time - start_time
# 计算平均处理速度 (秒/文件)
avg_speed = round(time_taken.total_seconds() / len(data_paths), 2)

# ===== 评估和输出结果 =====

# 评估所有结果的性能指标
evaluate()

# 输出平均处理速度
print("Avg speed:", avg_speed)
