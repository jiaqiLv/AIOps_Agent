#!/usr/bin/env python3
"""
绘制指标折线图并标记事件的脚本

该脚本用于读取指标CSV文件，为指定的指标列表绘制折线图，并在指定时间点标记事件。
要求：
1. 输入CSV文件和指标名称列表
2. 每个指标绘制一张子图，纵向排列成一张大图
3. 在指定时间点用竖线标记事件

使用方法:
    python plot_metrics_with_events.py <csv_file> <metrics_list> <events_dict> [output_file]

示例:
    python plot_metrics_with_events.py ../data/raw/zh_dataset/1116/data.csv \
        ["15_host_CPU","15_mem_usage","ai_request_avg_duration"] \
        '{"2025-11-16 12:30:00": "CPU峰值", "2025-11-16 13:00:00": "内存告警"}' \
        metrics_plot.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import argparse
import json
import sys
from pathlib import Path


def parse_time_column(df, time_col='time'):
    """解析时间列为datetime对象"""
    try:
        # 尝试常见的时间格式
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
        ]
        
        for fmt in formats:
            try:
                df[time_col] = pd.to_datetime(df[time_col], format=fmt)
                return df
            except ValueError:
                continue
        
        # 如果都不行，使用pandas自动检测
        df[time_col] = pd.to_datetime(df[time_col])
        return df
        
    except Exception as e:
        raise ValueError(f"解析时间列失败: {e}")


def parse_events(events_str):
    """解析事件字符串为字典"""
    try:
        # 将字符串时间转换为datetime对象
        parsed_events = {}
        for time_str, event_name in events_str.items():
            parsed_events[datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')] = event_name
        return parsed_events
    except json.JSONDecodeError:
        print("错误: 事件字典格式不正确，请使用JSON格式")
        sys.exit(1)
    except ValueError as e:
        print(f"错误: 时间格式不正确，请使用 'YYYY-MM-DD HH:MM:SS' 格式: {e}")
        sys.exit(1)


def plot_metrics_with_events(csv_file, metrics_list, events_dict, output_file=None):
    """
    绘制指标折线图并标记事件
    
    Args:
        csv_file: CSV文件路径
        metrics_list: 指标名称列表
        events_dict: 事件字典 {时间戳: 事件名称}
        output_file: 输出文件路径（可选）
    """
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file)
        print(f"成功读取CSV文件: {csv_file}")
        print(f"数据形状: {df.shape}")
    except Exception as e:
        print(f"错误: 读取CSV文件失败: {e}")
        sys.exit(1)
    
    # 解析时间列
    try:
        df = parse_time_column(df)
        print("时间列解析成功")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 检查指标是否存在
    missing_metrics = []
    available_metrics = df.columns.tolist()
    for metric in metrics_list:
        if metric not in available_metrics:
            missing_metrics.append(metric)
    
    if missing_metrics:
        print(f"错误: 以下指标在CSV文件中不存在: {missing_metrics}")
        print(f"可用指标: {available_metrics}")
        sys.exit(1)
    
    # 创建子图
    n_metrics = len(metrics_list)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 4 * n_metrics))
    
    # 如果只有一个指标，axes不是数组，需要转换为数组
    if n_metrics == 1:
        axes = [axes]
    
    # 为每个指标绘制折线图
    for i, metric in enumerate(metrics_list):
        ax = axes[i]
        
        # 绘制折线图（取绝对值）
        ax.plot(df['time'], df[metric].abs(), linewidth=1.5, color='blue', label=metric)
        
        # 标记事件
        for event_time, event_name in events_dict.items():
            # 检查事件时间是否在数据范围内
            if df['time'].min() <= event_time <= df['time'].max():
                ax.axvline(event_time, color='red', linestyle='--', alpha=0.7, linewidth=2)
                # 在第一个子图上添加事件标签
                if i == 0:
                    ax.text(event_time, ax.get_ylim()[1] * 0.9, event_name, 
                           rotation=90, verticalalignment='top', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 设置标题和标签
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # 格式化x轴时间显示
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        
        # 旋转x轴标签
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 设置总标题
    # fig.suptitle('Metrics Time Series with Events', fontsize=16, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图片
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    """主函数"""
    # Configuration variables - modify these values as needed
    csv_file = '../data/raw/zh_dataset/zh_dataset/0105/merged.csv'
    output_file = '../data/processed/zh_dataset/0105/metrics.png'
    metrics_list = ['nv_inference_request_failure_10.104.128.205:8002', 'triton_inference_duration_ms_10.104.128.205:9093']
    events_dict = {
        '2026-01-05 05:30:00': 'failure start',
        '2026-01-05 08:00:00': 'failure end',
    }

    # 解析事件字典
    events_dict = parse_events(events_dict)

    # 绘制图表
    plot_metrics_with_events(csv_file, metrics_list, events_dict, output_file)


if __name__ == '__main__':
    main()