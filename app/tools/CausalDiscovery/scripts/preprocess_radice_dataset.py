"""
RADICE 数据集预处理脚本

功能：
1. 将每个 artificialResults 子目录下的 total.csv 重命名为 data.csv
2. 将 gt.txt 拆分成 4 个文件：root_cause.txt, layer.txt, edges.txt, subgraph.txt
3. 按照统一的目录结构保存到 data/raw/RADICE/ 下

源目录: data/raw/24471163/{N5,N10,N15,N25}/{Nx}/artificialResults_*/
目标目录: data/raw/RADICE/{N5,N10,N15,N25}/artificialResults_*/
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict, deque


def find_subgraph_edges(edges, root_cause, target):
    """
    找出从根因到目标的所有有向路径上的边

    Args:
        edges: 边列表，每个边是 (from_node, to_node)
        root_cause: 根因节点
        target: 目标节点

    Returns:
        属于根因到目标路径的边列表
    """
    # 构建邻接表
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)

    for src, dst in edges:
        graph[src].append(dst)
        reverse_graph[dst].append(src)

    # 找出所有在从根因到目标路径上的节点
    # 1. 找出所有能到达目标的节点（从目标反向遍历）
    can_reach_target = set()
    queue = deque([target])
    can_reach_target.add(target)

    while queue:
        node = queue.popleft()
        for predecessor in reverse_graph[node]:
            if predecessor not in can_reach_target:
                can_reach_target.add(predecessor)
                queue.append(predecessor)

    # 2. 找出所有从根因出发能到达的节点
    reachable_from_root = set()
    queue = deque([root_cause])
    reachable_from_root.add(root_cause)

    while queue:
        node = queue.popleft()
        for successor in graph[node]:
            if successor not in reachable_from_root:
                reachable_from_root.add(successor)
                queue.append(successor)

    # 3. 路径上的节点是两个集合的交集
    path_nodes = can_reach_target & reachable_from_root

    # 4. 过滤出只涉及路径节点的边
    subgraph_edges = [
        (src, dst) for src, dst in edges
        if src in path_nodes and dst in path_nodes
    ]

    return subgraph_edges


def parse_gt_file(gt_path):
    """
    解析 gt.txt 文件，返回各个部分的内容

    Args:
        gt_path: gt.txt 文件路径

    Returns:
        dict: 包含 root_cause_line, layer_line, edges, subgraph_edges, root_cause, target
    """
    with open(gt_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) < 4:
        raise ValueError(f"{gt_path} 格式不正确")

    # 解析各部分
    # 第一行: root_cause target
    root_cause_line = lines[0]
    parts = root_cause_line.split()
    root_cause = int(parts[0])
    target = int(parts[1])

    # 第二行: 层级信息
    layer_line = lines[1]

    # 第三行: 边的数量
    num_edges = int(lines[2])

    # 第四行开始: 边列表
    edges = []
    for i in range(3, min(3 + num_edges, len(lines))):
        edge_parts = lines[i].split()
        if len(edge_parts) >= 2:
            src = int(edge_parts[0])
            dst = int(edge_parts[1])
            edges.append((src, dst))

    # 计算子图边
    subgraph_edges = find_subgraph_edges(edges, root_cause, target)

    return {
        'root_cause_line': root_cause_line,
        'layer_line': layer_line,
        'edges': edges,
        'subgraph_edges': subgraph_edges,
        'root_cause': root_cause,
        'target': target
    }


def write_gt_files(output_dir, gt_data):
    """
    将解析后的 gt 数据写入 4 个文件

    Args:
        output_dir: 输出目录
        gt_data: parse_gt_file 返回的数据字典
    """
    output_dir = Path(output_dir)

    # 1. root_cause.txt
    with open(output_dir / 'root_cause.txt', 'w') as f:
        f.write(gt_data['root_cause_line'] + '\n')

    # 2. layer.txt
    with open(output_dir / 'layer.txt', 'w') as f:
        f.write(gt_data['layer_line'] + '\n')

    # 3. edges.txt
    with open(output_dir / 'edges.txt', 'w') as f:
        for src, dst in gt_data['edges']:
            f.write(f"{src} {dst}\n")

    # 4. subgraph.txt
    with open(output_dir / 'subgraph.txt', 'w') as f:
        for src, dst in gt_data['subgraph_edges']:
            f.write(f"{src} {dst}\n")


def process_single_case(source_dir, target_dir, case_name):
    """
    处理单个 artificialResults_* 目录

    Args:
        source_dir: 源目录路径
        target_dir: 目标目录路径
        case_name: 案例名称（如 artificialResults_0）

    Returns:
        bool: 处理是否成功
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    # 检查源文件
    source_csv = source_dir / 'total.csv'
    source_gt = source_dir / 'gt.txt'

    if not source_csv.exists():
        print(f"  警告: {source_csv} 不存在，跳过")
        return False

    if not source_gt.exists():
        print(f"  警告: {source_gt} 不存在，跳过")
        return False

    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. 复制并重命名 total.csv -> data.csv
    target_csv = target_dir / 'data.csv'
    shutil.copy2(source_csv, target_csv)

    # 2. 解析并写入 gt 文件
    gt_data = parse_gt_file(source_gt)
    write_gt_files(target_dir, gt_data)

    return True


def process_dataset(source_base, target_base, dataset_name):
    """
    处理单个数据集（N5/N10/N15/N25）

    Args:
        source_base: 源数据根目录
        target_base: 目标数据根目录
        dataset_name: 数据集名称（N5/N10/N15/N25）

    Returns:
        int: 成功处理的案例数量
    """
    source_base = Path(source_base)
    target_base = Path(target_base)

    # 源目录: 24471163/N5/N5/artificialResults_*/
    source_dataset_dir = source_base / dataset_name / dataset_name

    if not source_dataset_dir.exists():
        print(f"警告: 源目录 {source_dataset_dir} 不存在")
        return 0

    # 目标目录: RADICE/N5/
    target_dataset_dir = target_base / dataset_name
    target_dataset_dir.mkdir(parents=True, exist_ok=True)

    # 找到所有 artificialResults_* 目录
    result_dirs = sorted([
        d for d in source_dataset_dir.iterdir()
        if d.is_dir() and d.name.startswith('artificialResults')
    ])

    success_count = 0
    for result_dir in result_dirs:
        case_name = result_dir.name
        target_case_dir = target_dataset_dir / case_name

        # 检查并解析 gt.txt 用于输出信息
        source_gt = result_dir / 'gt.txt'
        if source_gt.exists():
            gt_data = parse_gt_file(source_gt)

        if process_single_case(result_dir, target_case_dir, case_name):
            success_count += 1
            print(f"    ✓ {case_name}: 根因={gt_data['root_cause']}, 目标={gt_data['target']}, "
                  f"边数={len(gt_data['edges'])}, 子图边数={len(gt_data['subgraph_edges'])}")

    return success_count


def main():
    # 设置路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    source_base = project_root / 'data' / 'raw' / '24471163'
    target_base = project_root / 'data' / 'raw' / 'RADICE'

    # 数据集列表
    datasets = ['N5', 'N10', 'N15', 'N25']

    print("=" * 60)
    print("RADICE 数据集预处理脚本")
    print("=" * 60)
    print(f"源目录: {source_base}")
    print(f"目标目录: {target_base}")
    print()

    total_processed = 0

    for dataset in datasets:
        print(f"处理数据集: {dataset}")
        print("-" * 40)

        count = process_dataset(source_base, target_base, dataset)
        total_processed += count
        print(f"  完成: {count} 个案例")
        print()

    print("=" * 60)
    print(f"处理完成！共处理 {total_processed} 个案例")
    print(f"输出目录: {target_base}")
    print("=" * 60)


if __name__ == '__main__':
    main()
