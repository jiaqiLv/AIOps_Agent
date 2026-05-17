#!/usr/bin/env python3
"""
绘制指标折线图的脚本

该脚本用于读取指标CSV文件，为每个指标绘制折线图，并标记异常时间段。
要求：
1. 下午16:02-16:25是异常时间段，将折线图中这个区域制成红色透明底色，并增加图例
2. 横坐标是时间，纵坐标是指标值
3. 所有折线图保存到指定文件夹下

使用方法:
    python plot_metrics_with_anomaly.py <csv_file> <output_dir>

示例:
    python plot_metrics_with_anomaly.py ../data/aiops_challenge_2025/failures/8c1e8ce9-237/test_data/merged_metrics_adservice0.csv ../data/aiops_challenge_2025/failures/8c1e8ce9-237/test_data/figures
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from pathlib import Path
import argparse
from datetime import datetime
import pytz
from matplotlib.dates import DateFormatter



def plot_metric_with_anomaly(df, metric_name, output_dir, anomaly_start, anomaly_end):
    """
    为单个指标绘制折线图，并标记异常时间段

    Args:
        df (pd.DataFrame): 包含时间和指标值的数据框
        metric_name (str): 指标名称
        output_dir (str): 输出目录
        anomaly_start (datetime): 异常开始时间
        anomaly_end (datetime): 异常结束时间
    """
    # 创建图形
    plt.figure(figsize=(15, 8))
    
    # 绘制折线图
    plt.plot(df['time'], df[metric_name],
             linewidth=1.5, color='blue', label=metric_name)
    
    # 标记异常时间段（红色透明背景）
    plt.axvspan(anomaly_start, anomaly_end, alpha=0.3, color='red', 
                label='Anomaly (13:05-14:05)')
    
    # 设置标题和标签
    plt.title(f'{metric_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    
    # 设置时间格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=20))
    plt.gca().xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))

    # my_timezone = pytz.timezone('Asia/Shanghai')
    #
    # # 设置 x 轴的主刻度格式，并强制指定 tz 参数
    # # 格式字符串 '%Y-%m-%d %H:%M' 可以根据需要修改
    # formatter = DateFormatter('%Y-%m-%d %H:%M', tz=my_timezone)
    # plt.gca().xaxis.set_major_formatter(formatter)
    
    # 旋转x轴标签
    plt.xticks(rotation=45)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    plt.legend(loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    
    # 创建安全的文件名
    safe_filename = "".join(c for c in metric_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    if not safe_filename:
        safe_filename = f"metric_{metric_name}"
    
    # 保存图形
    output_path = os.path.join(output_dir, f"{safe_filename}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()  # 关闭图形以释放内存
    
    return output_path


def plot_all_metrics(csv_path, output_dir):
    """
    为CSV文件中的所有指标绘制折线图

    Args:
        csv_path (str): CSV文件路径
        output_dir (str): 输出目录路径
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取CSV文件: {csv_path}")
        print(f"数据形状: {df.shape}")
    except Exception as e:
        print(f"读取CSV文件时出错: {str(e)}")
        return
    
    # 检查必要列
    if 'time' not in df.columns:
        print("错误: CSV文件中未找到 'time' 列")
        return
    
    # 转换时间列为datetime类型
    try:
        df['time'] = pd.to_datetime(df['time'])
        print("时间列转换成功")
    except Exception as e:
        print(f"转换时间列时出错: {str(e)}")
        return
    
    # 获取所有指标列（除时间列外的所有列）
    metric_columns = [col for col in df.columns if col != 'time']
    
    if not metric_columns:
        print("错误: 未找到指标列")
        return
    
    print(f"找到 {len(metric_columns)} 个指标需要绘制")
    
    # 定义异常时间段（16:02-16:25）
    # 假设日期是2025-06-12（根据数据中的时间戳）
    anomaly_start = datetime(2026, 1, 5, 5, 30, 0)
    anomaly_end = datetime(2026, 1, 5, 8, 0, 0)
    
    print(f"异常时间段: {anomaly_start.strftime('%H:%M')} - {anomaly_end.strftime('%H:%M')}")
    
    # 为每个指标绘制图形
    successful_plots = 0
    for i, metric_name in enumerate(metric_columns):
        print(f"正在绘制 {i+1}/{len(metric_columns)}: {metric_name}")
        
        try:
            output_path = plot_metric_with_anomaly(df, metric_name, output_dir, 
                                              anomaly_start, anomaly_end)
            print(f"  保存到: {output_path}")
            successful_plots += 1
        except Exception as e:
            print(f"  绘制 {metric_name} 时出错: {str(e)}")
    
    print(f"\n绘制完成! 成功绘制 {successful_plots}/{len(metric_columns)} 个指标")
    print(f"所有图形已保存到: {output_dir}")


def main():
    """主函数"""
    # Configuration variables - modify these values as needed
    csv_file = '../data/raw/zh_dataset/zh_dataset/0105/data_processed.csv'
    output_dir = '../data/raw/zh_dataset/zh_dataset/0105/figures_processed/'

    # 检查输入文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误: 文件 {csv_file} 不存在")
        return

    try:
        print("=" * 60)
        print("开始绘制指标折线图")
        print("=" * 60)
        print(f"输入文件: {csv_file}")
        print(f"输出目录: {output_dir}")
        print("=" * 60)

        # 绘制所有指标
        plot_all_metrics(csv_file, output_dir)

        print("=" * 60)
        print("处理完成!")
        print("=" * 60)

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")


if __name__ == "__main__":
    main()