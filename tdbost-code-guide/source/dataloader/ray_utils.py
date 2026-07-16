"""相机光线、坐标投影、光线采样和 PFM 深度图读取工具。

【主线】BOS 重建需要知道「每个背景像素对应哪条三维光线」。本文件先用相机内参
把像素变成相机坐标系方向，再用相机外参变到世界坐标，最后可沿光线采样三维点。
"""

# re 用正则表达式解析 PFM 文件头中的宽和高。
import re

# NumPy 用于增加张量维度、测试随机数和读取 PFM 浮点数据。
import numpy as np
# PyTorch 提供光线/采样点张量运算。
import torch
# 【暂时不用深究】Kornia 像素网格函数仅出现在下方已注释的旧实现中。
from kornia import create_meshgrid
# 单独导入 torch.searchsorted，用于通过累积分布函数做逆变换采样。
from torch import searchsorted


def depth2dist(z_vals, cos_angle):
    """把光线上的深度样本 ``z_vals`` 转成相邻样本的实际路程长。

    ``z_vals`` 形状为 ``[N_rays, N_samples]``；``cos_angle`` 通常为 ``[N_rays]``，
    用来将深度轴方向间隔投影到光线方向。返回同形状 ``[N_rays,N_samples]``。
    """
    # z_vals: [N_ray N_sample]
    # 记住 z_vals 当前所在 CPU/GPU，新建的末尾距离也必须在同一设备。
    device = z_vals.device
    # 后一个深度减前一个深度，得到 ``[N_rays,N_samples-1]`` 相邻间隔。
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # 在末尾补一个极大距离 1e10，使样本数量恢复为 N_samples。
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1
    )  # [N_rays, N_samples]
    # 将每条光线的 cos_angle 在末尾增加一维，广播乘到它的所有采样间隔。
    dists = dists * cos_angle.unsqueeze(-1)
    # 返回经角度校正的每段光程长。
    return dists


def ndc2dist(ndc_pts, cos_angle):
    """计算 NDC 坐标中相邻光线采样点的欧氏距离。

    ``ndc_pts`` 形状通常为 ``[N_rays,N_samples,3]``；返回 ``[N_rays,N_samples]``。
    【注意】末尾样本没有下一点，因此用极大值补齐。
    """
    # 对每条光线的相邻三维点做差，求 xyz 欧氏范数。
    dists = torch.norm(ndc_pts[:, 1:] - ndc_pts[:, :-1], dim=-1)
    # 在采样维末尾追加 ``1e10*cos_angle``，保持输出样本数与输入相同。
    dists = torch.cat(
        [dists, 1e10 * cos_angle.unsqueeze(-1)], -1
    )  # [N_rays, N_samples]
    # 返回每条光线的采样区间长度。
    return dists


def get_ray_directions(point,H, W, focal, center=None):
    """把背景点像素坐标转成相机坐标系下的光线方向。

    ``point`` 预期末维包含 xy，``focal=[fx,fy]``，``center=[cx,cy]``。
    返回 ``[...,3]``，三分量是归一化像平面 x、y 和固定 z=1。
    【注意】原代码把 ``j`` 也取自 point[...,0]，注释版不修改这一行为。
    """
    '''grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5

    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    '''
    # 去掉长度为 1 的维度，使像素表更容易索引。
    point=point.squeeze()
    # 把最后一维第 0 列取作水平像素坐标 i。
    i=point[...,0]
    # 【注意】此处原样重复取第 0 列作 j，而非常见的 point[...,1]。
    j=point[...,0]
    # 如果调用者给了主点 center 就使用它，否则假定主点在图像正中心。
    cent = center if center is not None else [W / 2, H / 2]
    # 【暂时不用深究】原代码重复执行同一主点赋值，结果不变。
    cent = center if center is not None else [W / 2, H / 2]
    # 【物理直觉】像素减主点后除以焦距，得到该像素从光心看出去的方向斜率。
    directions = torch.stack(
        [(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1
    )  # (H, W, 3)

    # 返回未归一化长度的相机坐标方向。
    return directions


def get_ray_directions_blender(point,H, W, focal, center=None):
    """按 Blender/NeRF 坐标约定把像素 xy 转为相机光线方向。

    与 ``get_ray_directions`` 的主要区别是 y 分量取负，z 分量固定为 -1，
    表示相机向本坐标系的 -z 方向观看。返回形状 ``[...,3]``。
    """
    # grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5
    # i, j = grid.unbind(-1)
    # # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # # see https://github.com/bmild/nerf/issues/24
    # 去除长度为 1 的多余维度。
    point=point.squeeze()
    # 取每个背景点的 x 像素坐标。
    i=point[...,0]
    # 取每个背景点的 y 像素坐标。
    j=point[...,1]
    
    # 优先使用传入主点，否则使用图像几何中心。
    cent = center if center is not None else [W / 2, H / 2]
    # 打印 x 坐标形状、主点 x 和焦距 x，便于检查标定数据。
    print(i.shape,cent[0],focal[0])
    # 将像素偏移除焦距，y/z 符号按 Blender 坐标系设定。
    directions = torch.stack(
        [(i - cent[0]) / focal[0], -(j - cent[1]) / focal[1], -torch.ones_like(i)], -1
    )  # (H, W, 3)

    # 返回每个像素在相机坐标系中的射线方向。
    return directions


def get_rays(directions, c2w):
    """【主线】把相机坐标光线转成世界坐标的起点和方向。

    ``directions`` 末维为 3，可以是 ``[H,W,3]`` 或 ``[N,3]``；``c2w`` 至少含
    左上 ``[3,3]`` 旋转与最后一列平移。返回展平的 ``rays_o/rays_d``，均为 ``[N_rays,3]``。
    【注意】原 docstring 称方向已归一化，但此函数未显式除以范数。
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    # 用 c2w 的 3×3 旋转部分乘每个相机方向，广播实现为逐项乘后求和。
    rays_d = torch.sum(directions[..., np.newaxis, :] * c2w[:3, :3], -1)
    # The origin of all rays is the camera origin in world coordinate
    # 一幅图的所有光线都从同一相机光心出发，因此将 c2w 平移向量扩展至每条光线。
    rays_o = c2w[:3, -1].expand(rays_d.shape)  # (H, W, 3)

    # 无论输入原来是 H×W 网格还是点表，都展平成 ``[N_rays,3]``。
    rays_d = rays_d.view(-1, 3)
    # 以同样顺序展平光线起点。
    rays_o = rays_o.view(-1, 3)

    # 返回世界坐标的光线起点和方向。
    return rays_o, rays_d


def ndc_rays_blender(H, W, focal, near, rays_o, rays_d):
    """按 Blender/NeRF 符号约定将世界光线投影到 NDC 坐标。

    ``rays_o/rays_d`` 形状均为 ``[...,3]``；返回同形状的 NDC 起点与方向。
    【暂时不用深究】NDC 是把相机视锥压到标准立方体的投影坐标。
    """
    # Shift ray origins to near plane
    # 解光线参数 t，使起点沿方向移动后落到 z=-near 平面。
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    # 用 o+t*d 把光线起点前移至近裁剪平面。
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    # 将起点 x/z 按图宽和焦距缩放至 NDC x。
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    # 将起点 y/z 按图高和焦距缩放至 NDC y。
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    # 把近平面深度映射到 NDC z。
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    # 由光线方向斜率与起点斜率之差得到 NDC x 方向。
    d0 = (
        -1.0
        / (W / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    # 同理计算 NDC y 方向。
    d1 = (
        -1.0
        / (H / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    # 计算 NDC z 方向分量。
    d2 = -2.0 * near / rays_o[..., 2]

    # 将三个标量分量沿末维堆成 ``[...,3]`` NDC 起点。
    rays_o = torch.stack([o0, o1, o2], -1)
    # 堆成 ``[...,3]`` NDC 方向。
    rays_d = torch.stack([d0, d1, d2], -1)

    # 返回投影后的光线起点与方向。
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """将光线投影到 NDC 坐标（使用与 Blender 版不同的 z/符号约定）。

    输入/输出形状与 ``ndc_rays_blender`` 一致，均是 ``[...,3]`` 起点和方向。
    """
    # Shift ray origins to near plane
    # 解 t 使光线起点移到 z=near 平面。
    t = (near - rays_o[..., 2]) / rays_d[..., 2]
    # 沿光线方向移动起点。
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    # 投影并缩放 x 起点。
    o0 = 1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    # 投影并缩放 y 起点。
    o1 = 1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    # 映射 z 起点到 NDC 深度。
    o2 = 1.0 - 2.0 * near / rays_o[..., 2]

    # 计算 NDC x 方向分量。
    d0 = (
        1.0
        / (W / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    # 计算 NDC y 方向分量。
    d1 = (
        1.0
        / (H / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    # 计算 NDC z 方向分量。
    d2 = 2.0 * near / rays_o[..., 2]

    # 拼成 NDC 起点三分量。
    rays_o = torch.stack([o0, o1, o2], -1)
    # 拼成 NDC 方向三分量。
    rays_d = torch.stack([d0, d1, d2], -1)

    # 返回 NDC 光线。
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """【主线】根据每个区间的权重做分层采样，把更多新点放在重要区域。

    ``bins`` 是采样区间位置，``weights`` 是区间重要性，通常形状为
    ``[batch,N_bins-1]`` 和 ``[batch,N_bins-1]``/相容形状。返回 ``[batch,N_samples]`` 新样本。
    ``det=True`` 使用确定性等距数；``pytest=True`` 用 NumPy 固定随机序列。
    """
    # 新采样张量需与 weights 处在同一 CPU/GPU。
    device = weights.device
    # Get pdf
    # 每个权重加很小的正数，防止所有权重为 0 导致除零/NaN。
    weights = weights + 1e-5  # prevent nans
    # 沿最后一维归一化，得到总和为 1 的离散概率密度 PDF。
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # 对 PDF 累积求和，得到单调从 0 趋近 1 的 CDF。
    cdf = torch.cumsum(pdf, -1)
    # 在 CDF 最前补 0，表示第一个区间左边界的累积概率。
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    # 确定性模式下，在 [0,1] 上等距取 N_samples 个概率位置。
    if det:
        # 创建一维等距 u。
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
        # 将同一组 u 扩展到每个 batch。
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        # 随机模式下，为每个 batch 独立生成 [0,1) 均匀随机数。
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Pytest, overwrite u with numpy's fixed random numbers
    # 测试模式覆盖上面的 PyTorch u，使结果在不同平台上可复现。
    if pytest:
        # 固定 NumPy 随机种子。
        np.random.seed(0)
        # 构造目标 u 形状。
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        # 测试中的确定性分支仍使用等距数。
        if det:
            # 用 NumPy 在 [0,1] 生成 N_samples 个数。
            u = np.linspace(0.0, 1.0, N_samples)
            # 将 u 广播到所有 batch。
            u = np.broadcast_to(u, new_shape)
        else:
            # 测试随机分支使用种子 0 的 NumPy 均匀随机数。
            u = np.random.rand(*new_shape)
        # 把 NumPy u 转回 PyTorch 张量。
        u = torch.Tensor(u)

    # Invert CDF
    # 保证 u 内存连续，searchsorted 对连续数据处理更稳定。
    u = u.contiguous()
    # 在已排序 CDF 中查找每个 u 应插入的右侧索引，定位概率区间。
    inds = searchsorted(cdf.detach(), u, right=True)
    # 区间下界索引是 inds-1，但不能小于 0。
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    # 区间上界索引是 inds，但不能超过 CDF 最后一项。
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    # 将每个样本的[下界,上界]索引堆成 ``[batch,N_samples,2]``。
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # 构造扩展 CDF/bins 时的形状：[batch,N_samples,N_cdf]。
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    # 扩展 CDF 后按 inds_g 收集每个 u 左右边界的 CDF 值。
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # 以相同索引收集 bins 中对应的实际位置边界。
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # 计算当前 CDF 小区间的概率宽度。
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    # 若区间过小，将分母替换为 1，避免数值爆炸。
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    # 计算 u 在当前 CDF 区间内的相对位置 t。
    t = (u - cdf_g[..., 0]) / denom
    # 用 t 在 bins 左右边界间线性插值，得到真正采样位置。
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    # 返回按权重分布得到的新采样点。
    return samples


def dda(rays_o, rays_d, bbox_3D):
    """计算每条光线进入和离开三维轴对齐包围盒的参数 near/far。

    ``rays_o/rays_d`` 形状为 ``[N_rays,3]``，``bbox_3D`` 为 ``[2,3]``。
    返回 ``t_min/t_max``，均为 ``[N_rays,1]``。这是光线与 AABB 的 slab 相交算法。
    """
    # 逐分量计算方向倒数；1e-6 减少方向分量为 0 时的除零问题。
    inv_ray_d = 1.0 / (rays_d + 1e-6)
    # 计算光线与 bbox xyz 最小三个平面的 t，得 ``[N_rays,3]``。
    t_min = (bbox_3D[:1] - rays_o) * inv_ray_d  # N_rays 3
    # 计算与 xyz 最大三个平面的 t。
    t_max = (bbox_3D[1:] - rays_o) * inv_ray_d
    # 将两组边界 t 堆成 ``[2,N_rays,3]``，便于兼容负方向光线。
    t = torch.stack((t_min, t_max))  # 2 N_rays 3
    # 每个轴先取近端，三个轴再取最大，得到真正进入盒子的 t。
    t_min = torch.max(torch.min(t, dim=0)[0], dim=-1, keepdim=True)[0]
    # 每轴先取远端，三轴再取最小，得到离开盒子的 t。
    t_max = torch.min(torch.max(t, dim=0)[0], dim=-1, keepdim=True)[0]
    # 返回每条光线在包围盒内的参数范围。
    return t_min, t_max


def ray_marcher(rays, N_samples=64, lindisp=False, perturb=0, bbox_3D=None):
    """【主线】在每条光线的 near→far 区间上取样，生成三维查询点。

    ``rays`` 预期形状为 ``[N_rays,≥8]``：前 3 列是起点，3:6 是方向，
    6/7 列是 near/far。``bbox_3D`` 非空时会用包围盒交点覆盖 near/far。
    返回：``xyz_coarse_sampled [N_rays,N_samples,3]``、``rays_o``、``rays_d``、
    ``z_vals [N_rays,N_samples]``。
    """

    # Decompose the inputs
    # 读取光线条数。
    N_rays = rays.shape[0]
    # 拆出起点 xyz 和方向 xyz，两者均为 ``[N_rays,3]``。
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    # 拆出每条光线的近端/远端参数，均为 ``[N_rays,1]``。
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

    # 如果给出三维包围盒，就只在光线穿过盒子的部分采样。
    if bbox_3D is not None:
        # cal aabb boundles
        # 计算光线进出 AABB 的 t，覆盖光线中原带的 near/far。
        near, far = dda(rays_o, rays_d, bbox_3D)

    # Sample depth points
    # 在 0～1 之间生成 N_samples 个插值比例。
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
    # 默认在深度值上线性插值。
    if not lindisp:  # use linear sampling in depth space
        # z_steps=0 得 near，z_steps=1 得 far，中间点均匀过渡。
        z_vals = near * (1 - z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        # 视差空间里均匀采样会在近处放更多点，再取倒数回到深度。
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    # 将深度样本显式扩展为 ``[N_rays,N_samples]``。
    z_vals = z_vals.expand(N_rays, N_samples)

    # 训练时可在每个采样小区间内做随机抖动，减少固定样本的周期伪影。
    if perturb > 0:  # perturb sampling depths (z_vals)
        # 计算每对相邻深度的中点，形状 ``[N_rays,N_samples-1]``。
        z_vals_mid = 0.5 * (
            z_vals[:, :-1] + z_vals[:, 1:]
        )  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        # 上界由中点组成，末尾使用最后一个原深度。
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        # 下界首项用第一个原深度，其余使用中点。
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        # 为每个小区间生成 [0,perturb) 的独立随机比例。
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        # 在 lower 和 upper 间线性插值，替换固定深度。
        z_vals = lower + (upper - lower) * perturb_rand

    # 【物理直觉】用 ``起点 + 方向×距离`` 将每个 z 变成真正三维坐标。
    xyz_coarse_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(
        2
    )  # (N_rays, N_samples, 3)

    # 同时返回三维样本、光线起点/方向与深度，供后续物理积分使用。
    return xyz_coarse_sampled, rays_o, rays_d, z_vals


def read_pfm(filename):
    """读取 PFM（Portable Float Map）浮点图像。

    ``filename`` 是 PFM 路径。返回 ``(data, scale)``：``data`` 为 ``[H,W]`` 灰度图
    或 ``[H,W,3]`` 彩色图，``scale`` 为文件头中记录的正尺度值。
    """
    # 以二进制只读模式打开文件。
    file = open(filename, "rb")
    # 以下变量先置空，然后从 PFM 头部逐项解析。
    color = None
    # 图像宽度占位符。
    width = None
    # 图像高度占位符。
    height = None
    # 浮点数尺度占位符。
    scale = None
    # 字节序占位符：小端或大端。
    endian = None

    # 读第一行、UTF-8 解码并去掉末尾换行，得到 PF/Pf 标识。
    header = file.readline().decode("utf-8").rstrip()
    # 大写 PF 表示 3 通道彩色 PFM。
    if header == "PF":
        # 记录为彩色图。
        color = True
    # 小写 Pf 表示单通道 PFM。
    elif header == "Pf":
        # 记录为灰度/深度图。
        color = False
    else:
        # 标识既不是 PF 也不是 Pf，说明不是支持的 PFM 文件。
        raise Exception("Not a PFM file.")

    # 读尺寸行，用正则提取两个正整数（宽、高）。
    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    # 若正则匹配成功，解析尺寸。
    if dim_match:
        # 将两个字符串组转为 Python 整数。
        width, height = map(int, dim_match.groups())
    else:
        # 尺寸行格式错误时停止读取。
        raise Exception("Malformed PFM header.")

    # 读取第三行尺度/字节序标志，转为浮点数。
    scale = float(file.readline().rstrip())
    # PFM 规约用负 scale 表示小端字节序。
    if scale < 0:  # little-endian
        # NumPy dtype 字符 ``<`` 表示小端。
        endian = "<"
        # 返回的尺度使用正值。
        scale = -scale
    else:
        # NumPy dtype 字符 ``>`` 表示大端。
        endian = ">"  # big-endian

    # 从当前文件指针读取所有指定字节序的 32 位浮点数。
    data = np.fromfile(file, endian + "f")
    # 根据 color 标志决定重排为 H×W×3 还是 H×W。
    shape = (height, width, 3) if color else (height, width)

    # 将一维浮点序列重排为图像网格。
    data = np.reshape(data, shape)
    # PFM 行顺序与常见图像坐标相反，因此上下翻转。
    data = np.flipud(data)
    # 显式关闭文件句柄。
    file.close()
    # 返回图像数组与正尺度值。
    return data, scale


def ndc_bbox(all_rays):
    """从一组 NDC 光线估计包含起点和单位参数终点的轴对齐包围盒。

    ``all_rays`` 末维至少为 6，前 3 项为起点，后 3 项为方向。
    返回 ``[2,3]``：第 0 行 xyz 最小值，第 1 行 xyz 最大值。
    """
    # 展平所有光线起点并沿光线数取 xyz 最小值。
    near_min = torch.min(all_rays[..., :3].view(-1, 3), dim=0)[0]
    # 取光线起点的 xyz 最大值。
    near_max = torch.max(all_rays[..., :3].view(-1, 3), dim=0)[0]
    # 用 ``origin+direction`` 作为 t=1 的终点，取其 xyz 最小值。
    far_min = torch.min((all_rays[..., :3] + all_rays[..., 3:6]).view(-1, 3), dim=0)[0]
    # 取 t=1 终点的 xyz 最大值。
    far_max = torch.max((all_rays[..., :3] + all_rays[..., 3:6]).view(-1, 3), dim=0)[0]
    # 在终端打印起点/终点各自范围，便于检查 NDC 是否异常。
    print(
        f"===> ndc bbox near_min:{near_min} near_max:{near_max} far_min:{far_min} far_max:{far_max}"
    )
    # 对起点和终点再逐轴取总最小/最大，堆成完整 AABB。
    return torch.stack(
        (torch.minimum(near_min, far_min), torch.maximum(near_max, far_max))
    )
