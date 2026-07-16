"""评估渲染结果与真值图像相似程度的指标函数。

【主线】LPIPS 更关心人眼感知上是否相似，SSIM 则比较局部亮度、对比度和结构。
这些是「结果评分」，不是三维场或 BOS 位移的生成过程。
"""

# NumPy 负责图像数组、高斯核和统计量计算。
import numpy as np
# SciPy 提供二维卷积，用于对图像做高斯模糊。
import scipy
# PyTorch 负责把 NumPy 图像转成 LPIPS 网络需要的张量。
import torch

# 【暂时不用深究】用字典缓存已加载的 LPIPS 模型，避免每张图重新创建网络。
__LPIPS__ = {}


def init_lpips(net_name, device):
    """创建并返回一个 LPIPS 感知距离模型。

    ``net_name`` 只能是 ``"alex"`` 或 ``"vgg"``；``device`` 指定 CPU 或 GPU。
    返回值是已切换到评估模式、并移到目标设备的 LPIPS 网络。
    """
    # 先检查网络名称；不在允许列表中时立即报错。
    assert net_name in ["alex", "vgg"]
    # 在真正需要时才导入 lpips，减少不计算该指标时的依赖。
    import lpips

    # 在终端打印当前初始化的骨干网络，便于确认配置。
    print(f"init_lpips: lpips_{net_name}")
    # 创建 0.1 版 LPIPS，转为 eval 模式，再搬到 device 上。
    return lpips.LPIPS(net=net_name, version="0.1").eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
    """计算真值图 ``np_gt`` 与预测图 ``np_im`` 的 LPIPS 距离。

    输入预期是 NumPy ``[H, W, 3]`` RGB 数组；返回 Python 浮点数。
    【物理直觉】LPIPS 越小，两幅图在深度特征上越相似。
    """
    # 如果这种骨干网络还没有缓存，则只初始化一次。
    if net_name not in __LPIPS__:
        # 用网络名作字典键，保存已加载模型。
        __LPIPS__[net_name] = init_lpips(net_name, device)
    # 把 HWC 的真值 NumPy 数组转为 CHW 张量，再移到指定设备。
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    # 对预测图做同样的 HWC → CHW 转换。
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    # 计算感知距离；normalize=True 让库内部处理数值范围，.item() 取出标量。
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def rgb_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    """计算两幅 RGB 图像的结构相似性（SSIM）。

    ``img0``/``img1`` 预期形状均为 ``[H, W, 3]``，``max_val`` 是图像最大数值尺度。
    ``return_map=False`` 时返回整幅图平均 SSIM；为 True 时返回局部 SSIM 图。
    【注意】该函数原样要求三通道输入；若上游传入 ``[H, W]`` 会触发断言。
    """
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    # 确认输入是 HWC 三维图像。
    assert len(img0.shape) == 3
    # 确认最后一维有 3 个颜色通道。
    assert img0.shape[-1] == 3
    # 确保真值与预测尺寸完全相同，才能逐位置比较。
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    # 高斯窗的半宽；默认 11 对应中心两侧各 5 个点。
    hw = filter_size // 2
    # 针对偶数/奇数滤波器调整采样中心。
    shift = (2 * hw - filter_size + 1) / 2
    # 生成每个滤波位置到中心的归一化平方距离。
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    # 把平方距离经高斯函数转换为权重。
    filt = np.exp(-0.5 * f_i)
    # 让所有权重之和为 1，使模糊后亮度尺度不变。
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        """用一个二维滤波核 ``f`` 对单通道图 ``z`` 做 valid 卷积。"""
        # valid 模式只保留滤波核完全落在图像内部的结果。
        return scipy.signal.convolve2d(z, f, mode="valid")

    # 对每个颜色通道先纵向、再横向卷积，最后重新堆成 HWC 图像。
    filt_fn = lambda z: np.stack(
        [
            convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
            for i in range(z.shape[-1])
        ],
        -1,
    )
    # 计算第一幅图的局部均值μ0。
    mu0 = filt_fn(img0)
    # 计算第二幅图的局部均值μ1。
    mu1 = filt_fn(img1)
    # 缓存μ0²，后面计算方差和 SSIM 分母都要使用。
    mu00 = mu0 * mu0
    # 缓存μ1²。
    mu11 = mu1 * mu1
    # 缓存μ0μ1，用于两幅图亮度与协方差比较。
    mu01 = mu0 * mu1
    # 由 E[x²]-E[x]² 得到第一幅图的局部方差。
    sigma00 = filt_fn(img0**2) - mu00
    # 计算第二幅图的局部方差。
    sigma11 = filt_fn(img1**2) - mu11
    # 由 E[xy]-E[x]E[y] 计算两幅图的局部协方差。
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    # 浮点误差可能造成微小负方差，这里将它截为 0。
    sigma00 = np.maximum(0.0, sigma00)
    # 同样保证第二幅图的方差非负。
    sigma11 = np.maximum(0.0, sigma11)
    # 协方差绝对值不应超过两个标准差之积，同时保留原符号。
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    # SSIM 中的亮度稳定常数，避免局部均值接近 0 时不稳定。
    c1 = (k1 * max_val) ** 2
    # SSIM 中的对比度/结构稳定常数。
    c2 = (k2 * max_val) ** 2
    # 按 SSIM 公式计算分子：同时比较亮度和局部结构。
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    # 计算 SSIM 分母，用两幅图各自的能量作归一化。
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    # 逐位置相除得到局部 SSIM 图。
    ssim_map = numer / denom
    # 对所有位置和通道求平均，得到一个整体分数。
    ssim = np.mean(ssim_map)
    # 由 return_map 决定返回整张局部分数图，还是返回一个平均标量。
    return ssim_map if return_map else ssim
