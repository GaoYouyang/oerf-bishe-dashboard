"""渲染输出的可视化和体素网格分辨率辅助函数。"""

# Python 标准数学库，用于求立方根。
import math

# OpenCV 用于把单通道深度数值映射成伪彩色图。
import cv2
# NumPy 用于数组归一化、极值与数据类型转换。
import numpy as np


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """【主线】把 ``[H, W]`` 的深度数组转成便于人眼观察的伪彩色图。

    输入：``depth`` 是 NumPy 数组，形状为 ``[H, W]``；``minmax`` 可选择固定显示范围。
    输出：``x_`` 是 ``[H, W, 3]`` 的 uint8 彩色图，另返回实际使用的 ``[mi, ma]``。
    【注意】颜色只是显示手段，不会改变原始三维重建数据。
    """

    # 把 NaN 等非有限值换成可处理的数，避免可视化崩溃。
    x = np.nan_to_num(depth)  # change nan to 0
    # 如果调用者没有指定颜色映射范围，就从当前数据自动估计。
    if minmax is None:
        # 用最小正深度作为下界，从而忽略通常表示背景的 0。
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        # 用当前深度数组的最大值作为上界。
        ma = np.max(x)
    else:
        # 如果已给出范围，直接拆成最小值和最大值。
        mi, ma = minmax

    # 把深度线性归一化到约 0～1；1e-8 防止 ma 和 mi 相等时除零。
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    # 映射到图像常用的 0～255，并转成 8 位无符号整数。
    x = (255 * x).astype(np.uint8)
    # 调用 OpenCV 的颜色表，把单通道数值图变为三通道伪彩色图。
    x_ = cv2.applyColorMap(x, cmap)
    # 同时返回彩色图与范围，便于多幅图保持一致色标。
    return x_, [mi, ma]


def N_to_reso(n_voxels, bbox, adjusted_grid=True):
    """把总体素数换算为三维网格分辨率 ``[Nx, Ny, Nz]``。

    ``bbox`` 是 ``[2, 3]`` 的包围盒：第 0 行为 xyz 最小值，第 1 行为最大值。
    【物理直觉】调整模式会尽量让体素是立方体，长边自然分配更多格子。
    """
    # 选择是否按包围盒实际长宽高调整三个方向的格子数。
    if adjusted_grid:
        # 从 ``[2, 3]`` 包围盒中取出 xyz 下界和上界。
        xyz_min, xyz_max = bbox
        # 先用包围盒体积除以体素总数，再开立方根得到近似立方体边长。
        voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)
        # 每个方向的物理长度除以体素边长，取整并转成 Python 列表。
        return ((xyz_max - xyz_min) / voxel_size).long().tolist()
    else:
        # grid_each = n_voxels.pow(1 / 3)
        # 【暂时不用深究】不调整时，假定三边等长，直接对总体素数开立方根。
        grid_each = math.pow(n_voxels, 1 / 3)
        # 三个方向都使用相同整数分辨率。
        return [int(grid_each), int(grid_each), int(grid_each)]
