import numpy as np
import pandas as pd


def manual_igci(x, y, reference='uniform'):
    """
    手动实现基础版 IGCI
    返回值 < 0 则推断 X -> Y
    """

    # 1. 预处理：归一化到 [0, 1] (基于均匀分布参考)
    def normalize(v):
        return (v - v.min()) / (v.max() - v.min())

    x_norm = normalize(x)
    y_norm = normalize(y)

    # 2. 按照 x 的顺序排序，计算 X -> Y 的得分
    idx_x = np.argsort(x_norm)
    x_s, y_s = x_norm[idx_x], y_norm[idx_x]

    # 计算相邻点的差值
    dx = np.diff(x_s)
    dy = np.diff(y_s)

    # 过滤掉 dx 或 dy 为 0 的点避免 log(0)
    mask = (dx > 0) & (np.abs(dy) > 0)
    score_xy = np.mean(np.log(np.abs(dy[mask] / dx[mask])))

    # 3. 按照 y 的顺序排序，计算 Y -> X 的得分
    idx_y = np.argsort(y_norm)
    x_s2, y_s2 = x_norm[idx_y], y_norm[idx_y]
    dx2 = np.diff(x_s2)
    dy2 = np.diff(y_s2)

    mask2 = (dy2 > 0) & (np.abs(dx2) > 0)
    score_yx = np.mean(np.log(np.abs(dx2[mask2] / dy2[mask2])))

    return score_xy, score_yx

df = pd.read_csv('../data/raw/zh_dataset/zh_dataset/0105/data.csv')
x = df['nvidia_smi_ecc_errors_uncorrected_volatile_total_10.104.128.205:9835']
y = df['nv_inference_request_failure_10.104.128.205:8002']

# 测试
s1, s2 = manual_igci(x, y)
print(f"X->Y 得分: {s1:.4f}, Y->X 得分: {s2:.4f}")
if s1 < s2:
    print("结论: X 是原因 (X -> Y)")
else:
    print("结论: Y 是原因 (Y -> X)")