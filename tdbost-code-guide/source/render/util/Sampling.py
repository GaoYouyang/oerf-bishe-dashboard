"""渲染阶段的采样与鲁棒残差小工具。

【主线】这里没有完成整幅 BOS 图像的渲染，只提供两类基础计算：
1. Geman–McClure（GM）鲁棒函数，降低离群误差的影响；
2. 根据三维网格分辨率估算一条光线要取多少个采样点。
"""

# NumPy 主要用于计算三维分辨率向量的欧氏长度。
import numpy as np
# PyTorch 用于对张量求范数，并保留自动求导能力。
import torch


def GM_function(x, gamma):
    """【暂时不用深究】将普通误差映射为 Geman–McClure 鲁棒代价。

    参数：``x`` 是任意形状的误差张量；``gamma`` 是控制饱和速度的尺度。
    返回：与 ``x`` 同形状的数值，范围趋近 0到1。大误差会逐渐饱和，不会无限放大。
    """
    # 分子是平方误差；分母加上 gamma² 避免小误差被过度放大。
    return x**2 / (x**2 + gamma**2)


def GM_Resi(x, y, gamma):
    """计算两组三维量之间的 GM 鲁棒残差。

    【物理直觉】``x`` 和 ``y`` 可以看成两个三分量物理量，末维通常为 3。
    返回张量会消去最后一维；例如输入 ``[..., 3]``，输出为 ``[...]``。
    """
    # 先做逐元素差，经 GM 函数压制离群值，再对最后一维求 L1 范数并除以 3。
    return 1 / 3 * torch.norm(GM_function(x - y, gamma), p=1, dim=-1)


def cal_n_samples(reso, step_ratio=0.5):
    """根据三维网格分辨率估算每条光线的采样数。

    参数：``reso`` 通常是 ``[Nx, Ny, Nz]``；``step_ratio`` 越小，采样越密。
    返回：Python 整数。【注意】这是经验估算，不是严格的物理积分公式。
    """
    # np.linalg.norm(reso) 得到分辨率向量的对角线长度，除以步长比后取整。
    return int(np.linalg.norm(reso) / step_ratio)
