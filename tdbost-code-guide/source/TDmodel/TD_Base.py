"""
【主线】张量分解三维重建模型的公共基类。

这个文件串起整条核心链路：

1. 把相机光线从近端到远端分成许多三维采样点；
2. 由 CPmodel/MMmodel 在低秩特征线/平面中查询每个点的特征；
3. 用 MLP/KAN/直接输出方式解码密度或折射率；
4. 用中心有限差分估计折射率的 x/y/z 梯度；
5. 沿光线积分梯度，再投影到相机平面，得到 BOS 图像上的水平/竖直偏移。

零基础读者建议先看 ``TD_Base.forward`` 的【主线】注释，再回看辅助函数。
"""

# 【暂时不用深究】time 只用来统计射线过滤花费的时间。
import time
# 【暂时不用深究】这些名称用于类型标注，帮助读者和开发工具理解形状/可选参数。
from typing import List, Optional, Tuple, Union
# 【主线】PyTorch 用于张量计算、自动求导和 GPU 加速。
import torch
# 【主线】nn 包含神经网层与 Module 基类。
import torch.nn as nn
# 【主线】F 是函数式神经网工具，用到 grid_sample、softplus、池化等。
from torch.nn import functional as F
# 【暂时不用深究】导入第三方 FastKAN，它是 MLP 之外的备选解码网络。
from fastkan import FastKAN as KAN
# 【主线】导入项目自己的通用 MLP 解码器。
from .mlp import General_MLP

# ==========================================
# 1. KAN Layers (Custom Networks)
# ==========================================
class fastKAN(nn.Module):
    """
    【暂时不用深究】将 FastKAN 包装成与本项目解码器相同的调用接口。

    输入 ``xyz``、``x``、``t`` 前导形状需一致，最后一维合计为 ``indim``；
    输出形状的最后一维为 ``outdim``。
    """

    def __init__(self, indim, outdim):
        """【暂时不用深究】创建层宽 ``indim -> 8 -> 4 -> outdim`` 的 FastKAN。"""
        # 初始化 nn.Module 基类。
        super(fastKAN, self).__init__()
        # 创建真正的 FastKAN 子网络。
        self.KAN = KAN([indim, 8, 4, outdim])

    def forward(self, xyz, x, t):
        """
        【暂时不用深究】拼接坐标、潜在特征和时间，再让 FastKAN 生成物理输出。
        """
        # 沿最后通道维拼接三类输入，再调用 KAN。
        return self.KAN(torch.cat((xyz, x, t), dim=-1))


class ChebyKANLayer(nn.Module):
    """
    【暂时不用深究】使用切比雪夫多项式基的 KAN 层实验实现。

    输入形状 ``[..., input_dim]``，内部展平成批量，输出 ``[-1, output_dim]``。
    degree 表示多项式最高阶数。当前 TD_Base 主线未实例化 ChebyKAN。
    """

    def __init__(self, input_dim, output_dim, degree):
        """【暂时不用深究】保存维度，并初始化每个输入-输出-阶数的系数。"""
        # 初始化 nn.Module 基类。
        super(ChebyKANLayer, self).__init__()
        # 保存输入特征数。
        self.inputdim = input_dim
        # 保存输出特征数。
        self.outdim = output_dim
        # 保存切比雪夫多项式的最高阶。
        self.degree = degree

        # 创建形状 [input_dim,output_dim,degree+1] 的可训练多项式系数。
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        # 用均值 0 的小高斯噪声初始化系数。
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        # 【暂时不用深究】buffer 会随模型移动设备/保存，但不是可训练参数。
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        """
        【暂时不用深究】计算切比雪夫基值，再与可训练系数缩并得到输出。
        """
        # Normalize x to [-1, 1]
        # tanh 把任意实数压到 (-1,1)，满足后续 acos 的定义域。
        x = torch.tanh(x)
        # 展平前导维并增加阶数轴，然后复制 degree+1 份。
        x = x.view((-1, self.inputdim, 1)).expand(-1, -1, self.degree + 1)
        
        # Chebyshev polynomial computation
        # 【数学直觉】T_n(x)=cos(n*arccos(x))，下面三行直接实现此定义。
        x = x.acos()
        # 将角度乘上 0〜degree 各阶数。
        x *= self.arange
        # 取余弦得到每阶切比雪夫基值。
        x = x.cos()

        # einsum 对输入维 i 和多项式阶 d 求和，生成每个输出通道 o。
        y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs)
        # 整理并返回 [批量, outdim]。
        return y.view(-1, self.outdim)


class ChebyKAN(nn.Module):
    """
    【暂时不用深究】将一个 16 维输入、1 维输出的 ChebyKANLayer 包装为解码器接口。
    """

    def __init__(self):
        """【暂时不用深究】创建 16 -> 1、最高 8 阶的切比雪夫层。"""
        # 初始化 nn.Module 基类。
        super(ChebyKAN, self).__init__()
        # 创建内部切比雪夫 KAN 层。
        self.chebykan3 = ChebyKANLayer(16, 1, 8)

    def forward(self, xyz, x, t):
        """【暂时不用深究】按原实现只将中间特征 ``x`` 交给 ChebyKANLayer。"""
        # xyz 和 t 在此处仅为了接口统一而保留，不参与当前计算。
        return self.chebykan3(x)


# ==========================================
# 2. Rendering & Physics Utilities
# ==========================================
def raw2alpha(dndx: torch.Tensor, dndy: torch.Tensor, dndz: torch.Tensor, 
              dist: torch.Tensor, W2C: torch.Tensor, level=0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    【主线】沿每条相机射线积分折射率梯度，并投影为图像平面偏移。

    ``dndx/dndy/dndz``：通常为 ``[B,S]`` 或 ``[B,S,1]``，B 是光线数，S 是沿线采样点数。
    ``dist``：``[B,S]``，相邻采样点的步长。``W2C``：``[B,3,3]`` 世界到相机旋转矩阵。
    返回 ``detax``、``detay`` 两个 ``[B]`` 相机平面分量，以及
    ``dnwxyz`` 形状 ``[B,S,3]`` 的世界坐标梯度。

    【数学直觉】BOS 中，光线偏折来自折射率的空间梯度；离散代码用“梯度×步长再求和”近似连续积分。
    """
    # 去掉可能存在的末尾单位轴，并把 x/y/z 梯度堆成 [B,S,3]。
    dnwxyz = torch.stack((dndx.squeeze(), dndy.squeeze(), dndz.squeeze()), dim=-1)
    # 给 [B,S] 步长增加向量轴，以便与三维梯度相乘。
    dist_expanded = dist.unsqueeze(-1)
    
    # 【数学直觉】cumsum 得到从射线起点到每个采样点的累积梯度效应。
    dn = torch.cumsum(dnwxyz * dist_expanded, dim=1)
    # 再沿整条射线求和；level 控制是否加入累积高阶项。
    dntotal = torch.sum((dnwxyz + dn * level) * dist_expanded, dim=1)
    
    # Project to camera coordinates using W2C matrix
    # 用 W2C 第 0 行与世界梯度积分点乘，得到相机水平分量。
    detax = torch.sum(W2C[:, 0, :3] * dntotal, dim=-1)
    # 用 W2C 第 1 行得到相机竖直分量。
    detay = torch.sum(W2C[:, 1, :3] * dntotal, dim=-1)
    
    # 返回两个相机平面偏移分量和逐采样点三维梯度。
    return detax, detay, dnwxyz


def RGBRender(xyz_sampled: torch.Tensor, features: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """
    【暂时不用深究】plain 外观模式的恒等解码器：直接返回 features。

    ``xyz_sampled`` 和 ``time`` 只是为了与 MLP/KAN 接口一致；输出形状与 features 完全相同。
    """
    # 不做额外网络映射，直接交付分解模型特征。
    return features


def DensityRender(xyz_sampled: torch.Tensor, features: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """
    【主线】plain 密度模式的恒等解码器：把 ``[N,1]`` 密度特征直接当预测。

    ``xyz_sampled`` 和 ``time`` 在此分支不参与计算；输出形状与 features 相同。
    """
    # 不经 MLP，直接返回密度基矩阵投影结果。
    return features


class EmptyGridMask(nn.Module):
    """
    【暂时不用深究】用三维体素掩码标记哪些区域可以跳过，以节省射线计算。

    ``empty_volume`` 后三维对应 ``[D,H,W]``；``sample_empty`` 接收归一化坐标
    ``[...,3]`` 并返回每个点的掩码插值。
    """

    def __init__(self, device: torch.device, aabb: torch.Tensor, empty_volume: torch.Tensor):
        """【暂时不用深究】保存包围盒并把空体素整理成 grid_sample 所需形状。"""
        # 初始化 nn.Module 基类。
        super().__init__()
        # 记录 CPU/GPU 设备。
        self.device = device
        # 将 [2,3] 轴对齐包围盒移到模型设备。
        self.aabb = aabb.to(self.device)
        # 计算包围盒的 x/y/z 物理尺寸。
        self.aabbSize = self.aabb[1] - self.aabb[0]
        # 保存物理坐标到 [-1,1] 尺度的逆缩放系数。
        self.invgridSize = 1.0 / self.aabbSize * 2
        # 把 [D,H,W] 变成 3D grid_sample 要求的 [N=1,C=1,D,H,W]。
        self.empty_volume = empty_volume.view(1, 1, *empty_volume.shape[-3:])
        # 按 x/y/z 顺序记录网格分辨率，并移到目标设备。
        self.gridSize = torch.LongTensor(
            [empty_volume.shape[-1], empty_volume.shape[-2], empty_volume.shape[-3]]
        ).to(self.device)

    def sample_empty(self, xyz_sampled):
        """
        【暂时不用深究】在三维空体素掩码中插值查询任意归一化坐标。

        ``xyz_sampled`` 末维为 3，返回展平后的一维掩码值。
        """
        # 将所有查询点整理为 [1,N,1,1,3]，再对 3D 体素做三线性插值。
        empty_vals = F.grid_sample(
            self.empty_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True
        ).view(-1)
        # 返回 N 个坐标的掩码值。
        return empty_vals


# ==========================================
# 3. Base Tensor Decomposition Model
# ==========================================
class TD_Base(nn.Module):
    """
    【主线】时变流场/折射率场张量分解模型的公共基类。

    子类 CPmodel 和 MMmodel 负责“怎样用低秩张量存储并查询场特征”；
    本基类负责“怎样从相机射线取样、解码密度、求空间梯度并积分为 BOS 偏移”。

    核心形状约定：

    - B：一个批次的相机光线数；S：每条光线上采样点数；N：展平后的有效点数。
    - ``rays_chunk``：``[B,6]``，前 3 维是光线起点，后 3 维是方向。
    - ``xyz_sampled``：``[B,S,3]``，沿每条光线的三维坐标。
    - ``frame_time``：通常从 ``[B,1]`` 扩展到 ``[B,S,1]``。
    - 密度解码器对每个有效点输出 1 个标量。

    【注意】本文件混合了主线、备选网络和若干实验性/已注释路径。
    中文注释仅导读实际代码，不修复或重构原作者的行为。
    """
    def __init__(
        self,
        aabb: torch.Tensor,
        gridSize: List[int],
        device: torch.device,
        time_grid: int,
        near_far: List[float],
        density_n_comp: Union[int, List[int]] = 24,
        app_n_comp: Union[int, List[int]] = 24,
        density_dim: int = 1,
        app_dim: int = 27,
        DensityMode: str = "plain",
        AppMode: str = "general_MLP",
        emptyMask: Optional[EmptyGridMask] = None,
        fusion_one: str = "multiply",
        fusion_two: str = "concat",
        fea2denseAct: str = "softplus",
        init_scale: float = 0.1,
        init_shift: float = 0.0,
        normalize_type: str = "normal",
        **kwargs,
    ):
        """
        【主线】保存场景/网格/物理配置，初始化张量分解参数和解码器。

        重要参数：

        - ``aabb``：``[2,3]``，场景轴对齐包围盒，第 0 行是最小 xyz，第 1 行是最大 xyz。
        - ``gridSize``：``[X,Y,Z]`` 体素分辨率；``time_grid`` 是时间轴分辨率。
        - ``near_far``：沿每条相机射线采样的近、远深度。
        - ``density_n_comp/app_n_comp``：张量分解秩/隐特征通道数。
        - ``density_dim/app_dim``：特征基矩阵投影后的通道数。
        - ``fusion_one/fusion_two``：CP/MM 子类中低秩因子的融合方式。
        """
        # 【暂时不用深究】初始化 nn.Module，使后续属性中的 Parameter/子网络被自动登记。
        super().__init__()
        # 记录模型运行的 CPU/GPU 设备。
        self.device = device
        # 将场景包围盒移到同一设备。
        self.aabb = aabb.to(device)
        
        # Networks
        # 【主线】创建射线/路径畸变回归器；当前 forward 中其实际调用被注释。
        self.distortion_regressor = General_MLP(
            # 不使用分解特征，输出 6 维，只用坐标与视线原值，隐层 128，4 个线性层。
            0, 6, -1, -1, 0, 0, 128, 4, use_sigmoid=False, zero_init=False
        # 将回归器移到模型设备。
        ).to(device)
        
        # FIXME: MLPRF is not imported. Ensure it's defined or imported at the top.
        # self.Dpmodel = MLPRF(D=8, W=128, input_ch=4, output_ch=3, Ncord=0).to(device)

        # 【主线】保存时间特征网格的离散采样数 T。
        self.time_grid = time_grid
        # time_scale 把数据原始时间缩放到网络使用的 [-1,1]。
        self.time_scale = kwargs.get("time_scale", 1.0)
        # 保存射线的近、远采样边界。
        self.near_far = near_far
        # step_ratio 用体素大小缩放射线采样步长。
        self.step_ratio = kwargs.get("step_ratio", 1.0)
        # 根据网格和包围盒计算体素尺寸、步长和默认采样数。
        self.update_stepSize(gridSize)

        # 保存密度分解秩/通道数。
        self.density_n_comp = density_n_comp
        # 保存外观分解秩/通道数。
        self.app_n_comp = app_n_comp
        # 密度基矩阵输出特征数，plain 模式必须是 1。
        self.density_dim = density_dim
        # 外观基矩阵输出特征数。
        self.app_dim = app_dim
        # 控制 grid_sample/interpolate 的边角对齐定义。
        self.align_corners = kwargs.get("align_corners", True)

        # 可训练特征平面/特征线的随机初始振幅。
        self.init_scale = init_scale
        # 初始特征的整体平移值。
        self.init_shift = init_shift
        # 保存子类中第一级特征融合方式。
        self.fusion_one = fusion_one
        # 保存第二级特征融合方式。
        self.fusion_two = fusion_two

        # 保存可选的数据场景最小边界，当前本文件未直接使用。
        self.scene_bbox_min = kwargs.get("scene_bbox_min")
        # 保存可选的数据场景最大边界。
        self.scene_bbox_max = kwargs.get("scene_bbox_max")
        
        # 【主线】三个二维空间平面所使用的坐标轴：xy、xz、yz。
        self.matMode = [[0, 1], [0, 2], [1, 2]]
        # 与上述平面互补的单一空间轴：z、y、x，用于构造 zt、yt、xt。
        self.vecMode = [2, 1, 0]
        # 保存坐标归一化策略名称。
        self.normalize_type = normalize_type

        # Initialize Planes (Implemented in child classes)
        # 【主线】这是动态分派：实际执行 CPmodel/MMmodel 各自的 init_planes。
        self.init_planes(gridSize, device)

        # Density Settings
        # 保存将密度特征转为非线性密度的激活函数名称。
        self.fea2denseAct = fea2denseAct
        # 保存密度解码方式：plain、general_MLP 或 KAN。
        self.DensityMode = DensityMode
        # 密度/折射率变化量的数值边界，主要使用其差值作缩放。
        self.rho_bd = kwargs.get("rho_bd", [0.0, 1.0])
        # 基准密度/折射率。
        self.rho0 = kwargs.get("rho0", 1.1)
        
        # 【主线】根据配置创建密度解码器，并传入各类位置编码阶数。
        self.init_density_func(
            DensityMode,
            kwargs.get("density_t_pe", -1),
            kwargs.get("density_pos_pe", -1),
            kwargs.get("density_view_pe", -1),
            kwargs.get("density_fea_pe", 6),
            kwargs.get("density_featureC", 128),
            kwargs.get("density_n_layers", 3),
            device,
        )
        
        # Appearance Settings
        # 保存外观/物理量解码方式。
        self.AppMode = AppMode
        # 根据配置创建 app_regressor。
        self.init_app_func(
            AppMode,
            kwargs.get("app_t_pe", -1),
            kwargs.get("app_pos_pe", -1),
            kwargs.get("app_view_pe", 6),
            kwargs.get("app_fea_pe", 6),
            kwargs.get("app_featureC", 128),
            kwargs.get("app_n_layers", 3),
            device,
        )

        # 保存可选空体素掩码，用于跳过无效空间。
        self.emptyMask = emptyMask
        # 空间掩码的二值化阈值。
        self.emptyMask_thres = kwargs.get("emptyMask_thres", 0.001)
        # 原渲染管线的射线行进权重阈值，当前主 forward 未使用。
        self.rayMarch_weight_thres = kwargs.get("rayMarch_weight_thres", 0.0001)

   
        # ------------------------------------------
        # 【主线】BOS 物理尺度系数 D_LEVEL，默认为 2.2/3200。
        self.D_LEVEL = kwargs.get("physical_D_level", 2.2 / 3200)
        # Gladstone-Dale 类比例系数 GD，将密度变化与折射率/光线偏折联系起来。
        self.GD = kwargs.get("physical_GD", 2.48 / 10000)
        # 将 app_regressor 生成的 dp 路径变量缩放到物理损失量级。
        self.dp_scale = kwargs.get("dp_scale", 100.0)
        # dp 物理约束在总损失中的权重。
        self.dp_loss_weight = kwargs.get("dp_loss_weight", 0.001)

    def init_density_func(self, DensityMode, t_pe, pos_pe, view_pe, fea_pe, featureC, n_layers, device):
        """
        【主线】根据 ``DensityMode`` 选择密度特征到标量密度的解码器。

        plain 直接返回 1 维特征；general_MLP 可同时使用特征、位置和时间；
        KAN 使用备选 FastKAN。``t_pe/pos_pe/view_pe/fea_pe`` 控制各输入的位置编码。
        """
        # plain 模式无可训练解码网络。
        if DensityMode == "plain":
            # 【注意】既然特征直接当密度，基矩阵输出必须只有 1 维。
            assert self.density_dim == 1
            # 将恒等函数 DensityRender 作为解码器。
            self.density_regressor = DensityRender
        # general_MLP 模式使用可训练多层感知机。
        elif DensityMode == "general_MLP":
            # 密度被设计为与观看方向无关，因此禁用 view 输入。
            assert view_pe < 0
            # 创建 density_dim -> 1 的通用 MLP 解码器。
            self.density_regressor = General_MLP(
                self.density_dim, 1, t_pe, fea_pe, pos_pe, view_pe, featureC, n_layers, use_sigmoid=False, zero_init=False
            ).to(device)
        # KAN 模式使用 7 维拼接输入、1 维输出。
        elif DensityMode == "KAN": 
            self.density_regressor = fastKAN(7, 1).to(device)
        else:
            # 配置了未支持的模式时立即报错。
            raise NotImplementedError("Invalid Density Regression Mode")

    def init_app_func(self, AppMode, t_pe, pos_pe, view_pe, fea_pe, featureC, n_layers, device):
        """
        【主线】根据 ``AppMode`` 选择 app 特征的解码器。

        general_MLP 输出 3 维，plain 原样返回特征，KAN 在原代码中输出 4 维。
        在本项目 forward 的实验路径里，app_regressor 还用来预测沿射线的 dp 三维变量。
        """
        # 通用 MLP 把 app_dim 特征与可选坐标/时间/视线映射到 3 维。
        if AppMode == "general_MLP":
            self.app_regressor = General_MLP(
                self.app_dim, 3, t_pe, fea_pe, pos_pe, view_pe, featureC, n_layers, use_sigmoid=False, zero_init=False
            ).to(device)
        # plain 模式不引入额外可训练解码器。
        elif AppMode == "plain":
            self.app_regressor = RGBRender
        # KAN 模式使用 7 -> 4 的 FastKAN。
        elif AppMode == "KAN": 
            self.app_regressor = fastKAN(7, 4).to(device)
        else:
            # 不支持的模式直接报错。
            raise NotImplementedError("Invalid App Regression Mode")

    def update_stepSize(self, gridSize):
        """
        【主线】根据三维网格分辨率，重新计算体素尺寸、射线步长和采样数。

        ``gridSize`` 是长度 3 的 [X,Y,Z]。本方法不返回值，而是更新模型属性。
        """
        # 计算包围盒在 x/y/z 方向的物理长度。
        self.aabbSize = self.aabb[1] - self.aabb[0]
        # 将物理坐标换到 [-1,1] 时所用的逆尺度系数。
        self.invaabbSize = 2.0 / self.aabbSize
        # 将网格数转为设备上的长整型张量。
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        # 每轴相邻体素中心的物理间距；有 gridSize-1 个区间。
        self.units = self.aabbSize / (self.gridSize - 1)
        # 用三轴体素大小的平均值乘 step_ratio 作射线步长。
        self.stepSize = torch.mean(self.units) * self.step_ratio
        # 计算三维包围盒对角线长度。
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        # 根据半条对角线与步长估计每条射线默认采样数。
        self.nSamples = int((self.aabbDiag / self.stepSize / 2).item()) + 1

    # Placeholders for child classes
    def init_planes(self, res, device):
        """【主线】子类必须覆盖：初始化 CP/MM 特征线或平面。"""
        # 基类仅作接口占位，按原实现不执行操作。
        pass

    def compute_features(self, xyz_sampled):
        """【暂时不用深究】为可选子类预留的通用特征查询接口。"""
        # 基类占位实现。
        pass

    def compute_densityfeature(self, xyz_sampled, frame_time):
        """【主线】子类必须覆盖：查询指定时空点的密度潜在特征。"""
        # 基类占位实现。
        pass

    def compute_appfeature(self, xyz_sampled, frame_time):
        """【主线】子类必须覆盖：查询指定时空点的外观/物理潜在特征。"""
        # 基类占位实现。
        pass

    def normalize_coord(self, xyz_sampled):
        """
        【主线】把物理世界坐标线性映射到特征网格使用的 [-1,1] 范围。

        ``xyz_sampled`` 的最后一维必须为 3，其他前导维保持不变。
        """
        # 当前仅实现 normal 归一化策略。
        if self.normalize_type == "normal":
            # 先减最小边界得 [0,aabbSize]，再乘 2/aabbSize 得 [0,2]，最后减 1。
            return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1.0

    def feature2density(self, density_features: torch.Tensor) -> torch.Tensor:
        """
        【主线】用配置的非线性函数将原始特征换成密度/占据强度。

        输入输出形状相同。softplus 输出大于 0，tanh 输出位于 (-1,1)。
        """
        # softplus 是平滑版 ReLU，可确保密度为正。
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features)
        # tanh 允许正负对称输出。
        elif self.fea2denseAct == "tanh":
            return torch.tanh(density_features)
        # 其他激活名称未实现。
        raise NotImplementedError("Invalid activation function")

    def sample_rays(self, rays_o, rays_d, is_train=True, N_samples=-1):
        """
        【主线】在每条相机射线上从 near 到 far 等距采样三维点。

        ``rays_o/rays_d`` 为 ``[B,3]``；返回 ``rays_pts [B,S,3]``、
        ``interpx [1,S]`` 深度位置和 ``mask_inbbox [B,S]`` 包围盒有效掩码。
        训练时会对采样位置加小随机抖动，降低固定网格造成的伪影。
        """
        # N_samples<=0 时使用根据网格自动估计的 self.nSamples。
        N_samples = N_samples if N_samples > 0 else self.nSamples
        # 拆出近、远采样边界。
        near, far = self.near_far
        # 在 [near,far] 上生成 S 个等距深度，增加批量轴并移到与 rays_o 相同设备/类型。
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        
        # 训练模式下对每个深度加一个小的正随机偏移。
        if is_train:
            interpx += torch.rand_like(interpx) * ((far - near) / N_samples / 1.5)
            
        # 【数学直觉】射线方程 p(s)=o+s*d；将 [B,3] 与 [1,S] 广播得 [B,S,3]。
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        # 逐点检查 x/y/z 是否都严格位于包围盒最小值与最大值之间。
        mask_inbbox = (rays_pts[..., 0] < self.aabb[1, 0]) & (rays_pts[..., 0] > self.aabb[0, 0]) & \
                      (rays_pts[..., 1] < self.aabb[1, 1]) & (rays_pts[..., 1] > self.aabb[0, 1]) & \
                      (rays_pts[..., 2] < self.aabb[1, 2]) & (rays_pts[..., 2] > self.aabb[0, 2])
        # 返回三维采样点、射线参数和包围盒掩码。
        return rays_pts, interpx, mask_inbbox
    
    @torch.no_grad()
    def filtering_rays(
        self,
        all_rays: torch.Tensor,
        all_rgbs: torch.Tensor,
        all_times: torch.Tensor,
        all_w2c:torch.Tensor,
        all_depths: Optional[torch.Tensor] = None,
        N_samples: int = 256,
        chunk: int = 10240 * 5,
        bbox_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        【暂时不用深究】训练前剔除不穿过场景包围盒/非空区域的光线。

        ``all_rays`` 最后一维为 6，即 ``[起点 xyz, 方向 xyz]``；``all_rgbs``、
        ``all_times``、``all_w2c`` 与光线前导维对齐，``all_depths`` 可选。
        ``bbox_only=True`` 时用射线-包围盒相交判断；否则在光线上采样并查空体素掩码。

        返回布尔掩码筛选后的 rays/RGB/time/W2C/depth。
        【注意】原注解的 Tuple 元素数与实际返回 5 项不完全一致；保留原签名。
        """
        # 【暂时不用深究】装饰器 no_grad 禁止记录此数据筛选过程的梯度。
        # 打印过滤阶段开始标记。
        print("========> filtering rays ...")
        # 记录开始时间。
        tt = time.time()
        # 把光线最后通道维之前的所有维度相乘，得总光线数 N。
        N = torch.tensor(all_rays.shape[:-1]).prod()
        # 创建列表分块收集每条光线的保留布尔值。
        mask_filtered = []
        # 将 0〜N-1 的光线索引按 chunk 大小切块，避免一次占用过多显存。
        idx_chunks = torch.split(torch.arange(N), chunk)
        # 逐块处理光线。
        for idx_chunk in idx_chunks:
            # 取当前块并移到模型设备。
            rays_chunk = all_rays[idx_chunk].to(self.device)
            # 前 3 维是光线起点，后 3 维是方向。
            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            # Filter based on bounding box.
            
            # 【主线】只按包围盒时，使用 slab 射线-盒相交算法。
            if bbox_only:
                # 避免某个方向分量为 0 时除零，用 1e-6 替代 0。
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                # 计算光线到包围盒最大面的三轴参数。
                rate_a = (self.aabb[1] - rays_o) / vec
                # 计算光线到包围盒最小面的三轴参数。
                rate_b = (self.aabb[0] - rays_o) / vec
                # 每轴近交点取 min，再在三轴取最晚进入时刻 t_min。
                t_min = torch.minimum(rate_a, rate_b).amax(
                    -1
                )  # clamp(min=near, max=far)
                # 每轴远交点取 max，再在三轴取最早离开时刻 t_max。
                t_max = torch.maximum(rate_a, rate_b).amin(
                    -1
                )  # clamp(min=near, max=far)
                # 若离开时刻晚于进入时刻，射线与包围盒有非空交段。
                mask_inbbox = t_max > t_min
            # Filter based on emptiness mask.
            else:
                # 【注意】原代码此处调用 self.sample_ray（单数），而本类定义的是 sample_rays；不修改。
                xyz_sampled, _, _ = self.sample_ray(
                    rays_o, rays_d, N_samples=N_samples, is_train=False
                )
                #print(xyz_sampled.shape)
                
                # 将采样点从物理坐标换到 [-1,1] 网格坐标。
                xyz_sampled = self.normalize_coord(xyz_sampled)
                # 查询空体素掩码，只要一条光线上有任一采样点 >0 就保留。
                mask_inbbox = (
                    self.emptyMask.sample_empty(xyz_sampled).view(
                        xyz_sampled.shape[:-1]
                    )
                    > 0
                ).any(-1)
            
            # 将当前块的布尔掩码移回 CPU 并收集。
            mask_filtered.append(mask_inbbox.cpu())
        #print(all_rgbs.shape)
        # 按块顺序拼接掩码，并整理为与 RGB 前导维相同的形状。
        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        # 打印总耗时和保留光线比例。
        print(
            f"Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}"
        )
        # 如果数据集提供深度，一起用同一掩码筛选。
        if all_depths is not None:
            # 返回筛选后的五项数据。
            return (
                all_rays[mask_filtered],
                all_rgbs[mask_filtered],
                all_times[mask_filtered],
                all_w2c[mask_filtered],
                all_depths[mask_filtered],
            )
        else:
            # 没有深度时，前四项照常筛选，第五项返回 None。
            return (
                all_rays[mask_filtered],
                all_rgbs[mask_filtered],
                all_times[mask_filtered],
                all_w2c[mask_filtered],
                None,
            )
    def forward(
        self,
        rays_chunk: torch.Tensor,
        frame_time: torch.Tensor,
        W2C: torch.Tensor,
        white_bg: bool = True,
        is_train: bool = False,
        ndc_ray: bool = False,
        N_samples: int = -1,
        is_rendear: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        【主线】模型的核心前向计算：从相机光线重建折射率梯度，再得到 BOS 偏移。

        渲染/射线模式（``is_rendear=True``）的实际流程：

        1. 把 ``rays_chunk [B,6]`` 分为起点和方向，在每条光线上采样 S 个三维点；
        2. 对每个有效点构造 ``x±ds``、``y±ds``、``z±ds`` 六个邻居；
        3. 用 CP/MM 特征和 density_regressor 重建六个邻居的密度/折射率；
        4. 中心差分得 ``dn/dx,dn/dy,dn/dz``，再用 ``raw2alpha`` 沿光线积分并转到相机坐标；
        5. 返回两组水平/竖直偏移，以及路径和边界损失辅助项。

        参数形状：

        - ``rays_chunk``：``[B,6] = [起点 xyz, 方向 xyz]``。
        - ``frame_time``：通常 ``[B,1]``，原始时间尺度。
        - ``W2C``：每条光线的世界到相机矩阵，至少含左上 ``3×3`` 旋转块。
        - ``N_samples``：每条光线采样数，负数表示使用自动估计。
        - ``is_rendear``：原代码的拼写保留。False 时不处理射线，而是直接查输入坐标密度。

        【注意】原类型标注和旧英文说明声称返回 4 项 RGB 渲染结果，
        但当前 ``is_rendear=True`` 实际返回 8 项：
        ``(+x,+y,-x,-y, dp_bc, loss_dp, lossz, lossz)``。本注释版不修改签名或返回值。
        """
        # 【主线】取 BOS 尺度系数，后面用它将积分值换成图像偏移。
        dlevel=self.D_LEVEL
        # 【注意】乘 1 不改变值，原代码用 distanelevel 作 raw2alpha 的 level。
        distanelevel=dlevel*1
        # 取 Gladstone-Dale/物理比例系数，乘 1 保持原值。
        GD=self.GD*1
        # 【注意】此表达式计算 N_samples 加随机数，但没有赋值，因此不会改变 N_samples。
        N_samples+torch.randint(-5, 5, (1,))
        # 先在模型设备上创建形状 [1] 的边界损失零值。
        lossbc = torch.zeros(1).to(self.device)
        # 【主线】True 进入沿相机射线重建 BOS 偏移的主分支。
        if is_rendear:
            # Prepare rays.
            # 从 [B,6] 的后三列取出每条光线的世界坐标方向 [B,3]。
            viewdirs = rays_chunk[:, 3:6]
            # 【暂时不用深究】射线畸变初值设为标量 0；真正回归器调用在下方被注释。
            drays=0
            # drays=self.distortion_regressor(rays_chunk[:, :3],
            #                             rays_chunk[:, :3],
            #                             rays_chunk[:, :3],
            #                             viewdirs)
            # 【主线】在 B 条射线上采样：得 [B,S,3] 点、[1,S] 深度和 [B,S] 包围盒掩码。
            xyz_sampled, z_vals, ray_valid = self.sample_rays(
                # 前三列是光线起点 [B,3]。
                rays_chunk[:, :3],#+drays[:,:3]/100, 
                # 光线方向 [B,3]。
                viewdirs,#+drays[:,3:6]/100, 
                # 训练时采样位置会加随机抖动。
                is_train=is_train, N_samples=N_samples
            )
            # 拆出射线采样的近远边界。
            near, far = self.near_far
            # 把深度 z_vals 从 [near,far] 线性换到 [0,1]，形状 [1,S]。
            s_norms=(z_vals-near)/(far-near)
            # 给深度增加末维，再扩展为 [B,S,3]，供 app_regressor 当 pts 输入。
            s_norms_e=s_norms.unsqueeze(-1).expand(viewdirs.shape[0], -1, 3)
            # 把每条光线的方向复制到 S 个采样点，得 [B,S,3]。
            viewdirs_e=viewdirs.unsqueeze(1).expand(-1,s_norms.shape[1], -1)
            
            #print(s_norms_e.shape,viewdirs_e.shape)
            # 把每条光线的时刻整理并扩展到每个采样点，得 [B,S,1]。
            xyztime = frame_time.view(-1, 1, 1).expand(
                xyz_sampled.shape[0], xyz_sampled.shape[1], 1
            )
            # 将沿射线的物理世界坐标归一化到 [-1,1]。
            xyz_sampled0 = self.normalize_coord(xyz_sampled)
            # 【暂时不用深究】用 app_regressor 计算候选路径变量 dp。
            dp = self.app_regressor(
                # pts 位置参数传入复制为 3 维的射线归一化深度。
                s_norms_e, 
                # features 参数传入归一化三维采样坐标。
                xyz_sampled0,
                # frame_time 传入 [B,S,1] 时间。
                xyztime,
                # views 传入 [B,S,3] 视线方向。
                views=viewdirs_e
            # 【注意】整个网络输出被乘以 0，因此当前 dp 全为 0，但计算图仍保留。
            )*0#xyz_sampled.shape[1]*10
            # 用布尔索引取归一化深度 <0.2 的 dp 元素，作为边界区输出。
            dp_bc=dp[s_norms_e<0.2]
            #print(dp.shape,dp)
            # 【主线】计算每条射线相邻深度采样点的间距 [B,S]。
            dists = torch.cat(
                # 前 S-1 个是 z[i+1]-z[i]，最后一个用 0 补齐长度。
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1
            )
            # 训练模式下允许 dp 对采样坐标做物理尺度修正。
            if is_train:
                # 【注意】当前 dp 被乘 0，所以这行实际不改变 xyz_sampled。
                xyz_sampled=xyz_sampled+dp*GD#torch.cumsum(dp*dists[:,:,None]*GD,dim=1)#.detach()
            # 使用 [B,S] 布尔掩码取出包围盒内所有点，展平为 [N,3]。
            xyz_sample1=xyz_sampled[ray_valid]
            #print(xyz_sampled.shape,ray_valid.shape)
            # 【暂时不用深究】在物理坐标中构造一个柱面外且 x>20 的掩码；后面主要借用其设备。
            mask=((xyz_sample1[:,2]**2+(xyz_sample1[:,1]-16)**2)>80**2)&(xyz_sample1[:,0]>20)
            # xyz_sampledODE,frame_timeODE = self.sample_point(
            #     rays_chunk[:, :3], 100000
            # )
            
            # rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            # if ndc_ray:
            #     dists = dists * rays_norm
            # viewdirs = viewdirs / rays_norm
            #print(xyz_sampled[0],dists)
            # viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
            #print('w2c0,frame_time0:',W2C.shape,frame_time.shape)
            # 【主线】仅保留世界到相机矩阵的左上 3×3 旋转块。
            W2C=W2C[...,:3,:3].view(-1,3,3).expand(
                # 扩展/对齐为每条射线一个 [3,3] 矩阵。
                xyz_sampled.shape[0], 3,3
            )
            # 再次将每条光线的时间扩展到 S 个采样点，得 [B,S,1]。
            frame_time = frame_time.view(-1, 1, 1).expand(
                xyz_sampled.shape[0], xyz_sampled.shape[1], 1
            )
            
            # # Normalize coordinates.
            # min_value = xyz_sampled.min()
            # max_value = xyz_sampled.max()
            #print(self.aabb,xyz_sampled.shape,xyz_sampled.max(),xyz_sampled.min(),xyz_sampled[0],frame_time.max(),frame_time.min())
            # print(f"Minimum Value: {min_value}")
            # print(f"Maximum Value: {max_value}")
            # 【主线】将全部射线采样点从物理坐标换到 [-1,1] 特征网格坐标。
            xyz_sampled = self.normalize_coord(xyz_sampled)
            # 用 ray_valid 取出有效归一化点，得 [N,3]；ODE 是原代码命名。
            xyz_sampledODE =xyz_sampled[ray_valid]# self.normalize_coord(xyz_sampledODE)
            # Assuming xyz_sampled is a PyTorch tensor
            ###
           
            # 为边界约束再保留一份同样的有效坐标 [N,3]。
            xyz_bc=xyz_sampled[(ray_valid)]
            # 将有效点时间从 [0,time_scale] 映射到 [-1,1]。
            xyztime_bc=2*frame_time[(ray_valid)]/self.time_scale-1
            # If emptiness mask is availabe, we first filter out rays with low opacities.
            # if self.emptyMask is not None:
            #     emptiness = self.emptyMask.sample_empty(xyz_sampled[ray_valid])
            #     empty_mask = emptiness > 0
            #     ray_invalid = ~ray_valid
            #     ray_invalid[ray_valid] |= ~empty_mask
            #     ray_valid = ~ray_invalid

            # RI Initialize sigma and rgb values.
            # 【暂时不用深究】创建 [B,S,3] 零张量 dnpre，当前活跃代码后续未写入/未使用。
            dnpre=torch.zeros((*xyz_sampled.shape[:2],3), device=xyz_sampled.device)
            # 【主线】创建 [B,S,3] 梯度容器，无效采样点保持为 0。
            dnpreND=torch.zeros((*xyz_sampled.shape[:2],3), device=xyz_sampled.device)
            
            # 取有效点的时间并映射到 [-1,1]，形状通常 [N,1]。
            xyztime=2*frame_time[ray_valid]/self.time_scale-1
            # 使用同一归一化时间作为后续六个差分邻居的时间。
            frame_timeODE=xyztime#2*frame_timeODE/self.time_scale-1
            # Compute density feature and density if there are valid rays.
            # 【主线】只有至少一个采样点位于包围盒内，才计算密度空间梯度。
            if ray_valid.any():
                #加入差分坐标xyz,x+0.01,y+0.01,z+0.01
                # 【数学直觉】选择有限差分步长 ds。随机数在 [0,1) 乘以负值 -0.0007，因此 ds 约在 (0.0043,0.005]。
                ds = 0.005+ torch.rand(1, device=xyz_sampled.device)*(0.001 - 0.0003)#3/N_samples#0.015#+ torch.rand(1, device=xyz_sampled.device) * (0.005 - 0.003)
                # 创建与有效坐标 [N,3] 同形状的零偏移 dx。
                dx=torch.zeros_like(xyz_sampledODE, device=xyz_sampledODE.device)
                # 仅将 x 分量设为 ds，所以 dx=(ds,0,0)。
                dx[...,0]=ds
                # 创建 y 方向零偏移。
                dy=torch.zeros_like(xyz_sampledODE, device=xyz_sampledODE.device)
                # 仅将 y 分量设为 ds，所以 dy=(0,ds,0)。
                dy[...,1]=ds
                # 创建 z 方向零偏移。
                dz=torch.zeros_like(xyz_sampledODE, device=xyz_sampledODE.device)
                # 仅将 z 分量设为 ds，所以 dz=(0,0,ds)。
                dz[...,2]=ds
                # 【主线】按 [-x,-y,-z,+x,+y,+z] 顺序拼接六组邻居，[N,3] -> [6N,3]。
                xyz=torch.cat((xyz_sampledODE-dx,xyz_sampledODE-dy,xyz_sampledODE-dz,xyz_sampledODE+dx,xyz_sampledODE+dy,xyz_sampledODE+dz),0)
                # 六个空间邻居属于同一时刻，因此将 [N,1] 时间拼六份得 [6N,1]。
                xyztimeODE=torch.cat((frame_timeODE,frame_timeODE,frame_timeODE,frame_timeODE,frame_timeODE,frame_timeODE),0)
                # 记录原始有效点数 N，用来从 [6N,...] 输出中切回六组。
                lens=xyz_sampledODE.shape[0]
                
                
                
                # # xyz=xyz_sampled[ray_valid]
                # # xyztime=frame_time[ray_valid]
                # #print(xyztime.mean(),self.time_scale)
            
                
                #print(xyztime)
                # 【主线】调用 CPmodel/MMmodel 的子类方法，查询 6N 个邻居的密度潜在特征。
                density_feature = self.compute_densityfeature(
                    xyz, xyztimeODE
                )
                # 用 plain/MLP/KAN 密度解码器将特征转成 [6N,1] 标量。
                density = self.density_regressor(
                    # 归一化三维坐标 [6N,3]。
                    xyz,
                    # 张量分解查出的特征；[...] 表示取全部元素。
                    density_feature[...],
                    # 归一化时间 [6N,1]。
                    xyztimeODE,
                # 用 rho_bd 上下界的差对网络输出做数值尺度缩放。
                )*(self.rho_bd[1]-self.rho_bd[0])#+self.rho0
                #densitytotal=density
                #density[density>self.rho0]=self.rho0
                #print(self.invaabbSize)
                # 【主线】x 中心差分：[n(x+ds)-n(x-ds)]/(2ds)，再用归一化尺度和 GD 项修正。
                dndx=(density[3*lens:4*lens]-density[0:lens])/(ds)/2*self.invaabbSize[0]/(1+density[:lens]*GD)
                # y 中心差分：第 5 组（+y）减第 2 组（-y）。
                dndy=(density[4*lens:5*lens]-density[lens:2*lens])/(ds)/2*self.invaabbSize[1]/(1+density[:lens]*GD)
                # z 中心差分：第 6 组（+z）减第 3 组（-z）。
                dndz=(density[5*lens:6*lens]-density[2*lens:3*lens])/(ds)/2*self.invaabbSize[2]/(1+density[:lens]*GD)
                #print(dnpreND.shape,ray_valid.shape,density.shape)
                # 将三个 [N,1] 梯度拼成 [N,3]，再按 ray_valid 写回 [B,S,3] 容器。
                dnpreND[ray_valid]=torch.cat((dndx,dndy,dndz),-1)
                
                # drho_feature = self.compute_appfeature(
                #     xyz_sampledODE, frame_timeODE,
                # )
                # drho = self.app_regressor(
                #     xyz_sampledODE,
                #     drho_feature[...],
                #     frame_timeODE,
                # )*(self.rho_bd[1]-self.rho_bd[0])/10
                #print(drho.shape,dndx.shape)
                #折射率梯度
                
                
                #print('w2c,ray_valid,frame_time:',W2C.shape,ray_valid.shape,frame_time.shape)
                #bc折射率
            # 【主线】另外在原始有效坐标上重建一次密度，用来构造边界/基准损失。
            if ((ray_valid)).any():#
                # 查询 [N,3] 有效点的密度潜在特征。
                density_feature = self.compute_densityfeature(
                    xyz_bc, xyztime_bc
                )
                # 解码密度并用 rho_bd 范围缩放。
                density = self.density_regressor(
                    xyz_bc,
                    density_feature[...],
                    xyztime_bc,
                )*(self.rho_bd[1]-self.rho_bd[0])#+self.rho0
                # 使用 densitytotal 名称保存全部有效点密度。
                densitytotal=density
                # 计算当前批次有效点的平均密度。
                densitymean=torch.mean(densitytotal)
                # 【注意】先用 Python 数值 0 覆盖初始 lossbc，下一行再会覆盖。
                lossbc=0
                #if densitymean<self.rho0:
                    #print(densitytotal.shape,mask.shape)
                # 只取高于当前平均的预测，惩罚它们与基准 rho0 的平方差。
                lossbc=torch.mean((densitytotal[(densitytotal>densitymean)]-self.rho0)**2)#(densitytotal<densitymean)&+torch.mean(densitytotal[mask]-self.rho0)**2
            else:
                # 【注意】没有有效点时，创建 [2,1] 的 rho0 常数密度，并用 .to(mask) 对齐设备。
                density=self.rho0*torch.ones(2,1).to(mask)
                #density[density>self.rho0]=self.rho0
            #Compute appearance feature and rgb if there are valid rays (whose weight are above a threshold).
            '''app_mask =ray_valid# weight > self.rayMarch_weight_thres
            if app_mask.any():
                app_features = self.compute_appfeature(
                    xyz, xyztime
                )
                validsigma = self.app_regressor(
                    xyz,
                    app_features,
                    xyztime,
                )
                sigma[0,ray_valid] = validsigma.view(-1)[0:lenidx]
                sigma[1,ray_valid] = validsigma.view(-1)[lenidx:lenidx*2]
                sigma[2,ray_valid] = validsigma.view(-1)[lenidx*2:lenidx*3]
                sigma[3,ray_valid] = validsigma.view(-1)[lenidx*3:lenidx*4]'''
                #print(Physparam.shape,validsigma.shape,ray_valid.shape)
                #Physparam[ray_valid]=validsigma
    
                #Physparam[app_mask] = valid_rgbs
            # acc_map = torch.sum(weight, -1)
            # #rgb_map = torch.sum(weight[..., None] * rgb, -2)

            # # If white_bg or (is_train and torch.rand((1,))<0.5):
            # if white_bg or not is_train:
            #     rgb_map = rgb_map + (1.0 - acc_map[..., None])
            # else:
            #     rgb_map = rgb_map + (1.0 - acc_map[..., None]) * torch.rand(
            #         size=(1, 3), device=rgb_map.device
            #     )
            # rgb_map = rgb_map.clamp(0, 1)
            #print(self.rho0,self.rho_bd)
            #print(dndx)
            # 【主线】沿射线积分 [B,S,3] 折射率梯度，再投影到相机 x/y 轴。
            detax,detay,dntotal = raw2alpha(
                 # 分别传入 x/y/z 梯度 [B,S]、采样间距、W2C，以及 level 高阶系数。
                 dnpreND[...,0],dnpreND[...,1],dnpreND[...,2],dists,W2C,level=distanelevel
            )  # alpha is the opacity, weight is the accumulated weight. bg_weight is the accumulated weight for last sampling point.
            #print(dists.shape,dp.shape,viewdirs_e[:,1:-1].shape,dnpreND[:,1:-1].shape)
            #ddpds=(dp[:,2:]-dp[:,:-2])/dists[:,1:-1].unsqueeze(-1)/2
            #ddp2ds2=(dp[:,2:]+dp[:,:-2]-2*dp[:,1:-1])/(dists[:,1:-1].unsqueeze(-1))**2
            #print(torch.cumsum(dntotal,dim=1).shape,torch.cumsum(dntotal,dim=1))
            # 【数学直觉】对梯度先累积一次得方向变化，再累积一次得路径偏移。
            # 将这个双重积分与 dp/dp_scale 比较，然后乘 dp_loss_weight 得物理约束残差。
            loss_dp=1*(torch.cumsum(torch.cumsum(dntotal*dists.unsqueeze(-1),dim=1
                                               )*dists.unsqueeze(-1),dim=1)-dp/self.dp_scale)*self.dp_loss_weight#ddp2ds2-viewdirs_e[:,1:-1]*torch.sum(dnpreND[:,1:-1]*GD*(viewdirs_e[:,1:-1]+ddpds),dim=-1).unsqueeze(-1)-dnpreND[:,1:-1]*GD
            
            #torch.cumsum(dntotal,dim=1)*2.4/10000-dp#ddp2ds2-viewdirs_e[:,1:-1]*torch.sum(dnpreND[:,1:-1]*2.4/10000*(viewdirs_e[:,1:-1]+ddpds),dim=-1).unsqueeze(-1)-dnpreND[:,1:-1]*2.4/10000
            #torch.sum(dp*dists[:,:,None],dim=1)
            #dpx,dpy=torch.sum(W2C[:,0,:3]*torch.sum(dp*dists[:,:,None],dim=1),dim=-1)*distanelevel,torch.sum(W2C[:,1,:3]*torch.sum(dp*dists[:,:,None],dim=1),dim=-1)*distanelevel
            #print(loss_dp.shape,loss_dp)
            # detaxd,detayd = raw2alpha(
            #      dnpre[...,0],dnpre[...,1],dnpre[...,2],dists,W2C
            # )  # alpha is the opacity, weight is the accumulated weight. bg_weight is the accumulated weight for last sampling point.
            
            #sigma1=validsigma[(xyz[...,0]>0.5)|(xyz[...,0]<-0.5)|(xyz[...,1]>0.5)|(xyz[...,1]<-0.5)]
            #print(dndx.shape,dndd.shape)
            # drho=drho.detach()
            # +torch.mean((drho[:,1:]-drho[:,:-1])**2)/10
            # 【注意】若所有 lossbc 都是 NaN（例如筛选后集合为空），则将它替换为 0。
            if torch.all(torch.isnan(lossbc)):
                lossbc=0
            # 将边界损失缩小 10 倍，再乘设备对齐的 [1] 张量。
            lossz=(lossbc/10)*torch.ones(1).to(mask)
            #print(density[:lens].shape,detax.shape)#dnpreND[:,1:]*0-dnpreND[:,:-1]*
            # 【主线】返回 8 项：前两项代数上等于 +detax/dlevel*GD、+detay/dlevel*GD；
            # 第 3/4 项是相反符号的偏移，第 5 项是近端 dp，第 6 项是路径物理残差，末两项是相同边界损失。
            return -(-detax)/dlevel*GD,-(-detay)/dlevel*GD,-detax/dlevel*GD,-detay/dlevel*GD,dp_bc,loss_dp,lossz,lossz#density-1.2#-detax,-detay,-detaxd,-detayd,torch.stack((dndx-dndd[...,0].detach(),dndy-dndd[...,1].detach(),dndz-dndd[...,2].detach()),-1), sigma1, z_vals,sigma[:,1]-sigma[:,-1]
        else:
            # If emptiness mask is availabe, we first filter out rays with low opacities.
            
            # 【主线】非射线模式：把 rays_chunk 直接视为待查询三维坐标，归一化到 [-1,1]。
            rays_chunk = self.normalize_coord(rays_chunk)
            #print(rays_chunk[...,2].max(),rays_chunk[...,2].min())
            #ray_valid = torch.ones_like(rays_chunk[..., 0], dtype=torch.bool, device=rays_chunk.device)
            # if self.emptyMask is not None:
            #     emptiness = self.emptyMask.sample_empty(rays_chunk[ray_valid])
            #     empty_mask = emptiness > 0
            #     ray_invalid = ~ray_valid
            #     ray_invalid[ray_valid] |= ~empty_mask
            #     ray_valid = ~ray_invalid
            # 将原始时间从 [0,time_scale] 映射到 [-1,1]。
            frame_time=2*frame_time/self.time_scale-1
            #print(frame_time.max(),frame_time.min())
            # 调用 CP/MM 子类查询每个坐标/时刻的密度潜在特征。
            density_feature = self.compute_densityfeature(
                rays_chunk,
                frame_time,
            )
            # 用密度解码器将潜在特征转为标量预测，并复用 density_feature 变量名保存结果。
            density_feature = self.density_regressor(
                rays_chunk,
                density_feature,
                frame_time,
            )
            # 用 rho_bd 上下界差值缩放解码输出，得最终直接查询结果 sigma1。
            sigma1 = density_feature*(self.rho_bd[1]-self.rho_bd[0])#+self.rho0#+self.rho_bd[0]#self.feature2density(density_feature)
            #sigma1[sigma1>self.rho0]=self.rho0
            #sigma =torch.zeros_like(frame_time, device=rays_chunk.device)
            #sigma[ray_valid]=sigma1 
            '''
            app_features = self.compute_appfeature(
                    rays_chunk,frame_time
                )
            sigma = self.app_regressor(
                rays_chunk,
                app_features,
                frame_time,
            )
            # '''
            
            # 【注意】非射线模式返回两份同一 sigma1 张量。
            return sigma1 ,sigma1#

    # ==========================================
    # 4. Extensibility & Helpers
    # ==========================================
    def PIVdatacal(self, xyz, t):
        """
        【主线】直接在给定的三维坐标与时刻上查询 app/PIV 物理量。

        ``xyz`` 形状通常为 ``[N,3]``，``t`` 为 ``[N]``；返回形状取决于 app_regressor，
        general_MLP 默认最后一维为 3。
        """
        # 将物理世界坐标换到 [-1,1] 特征网格坐标。
        rays_chunk = self.normalize_coord(xyz)
        # 将原始时间线性映射到 [-1,1]。
        frame_time = 2 * t / self.time_scale - 1
        # 【主线】调用 CP/MM 子类查询 app 潜在特征；给时间增加末尾通道轴。
        Physparam_feature = self.compute_appfeature(rays_chunk, frame_time.unsqueeze(-1))
        # 用 app_regressor 结合坐标、潜在特征和时间生成最终物理量。
        return self.app_regressor(rays_chunk, Physparam_feature, frame_time.unsqueeze(-1))

    def BOSdata(self, xyz, t):
        """
        【主线】直接在给定三维坐标与时刻上查询 BOS 密度/折射率场。

        ``xyz`` 通常为 ``[N,3]``，``t`` 为 ``[N]``，返回通常为 ``[N,1]``。
        本方法不沿相机光线积分，而是取某些具体体素点的场值。
        """
        # 归一化三维坐标。
        rays_chunk = self.normalize_coord(xyz)
        # 归一化时间到 [-1,1]。
        frame_time = 2 * t / self.time_scale - 1
        # 查询张量分解中的密度潜在特征。
        Physparam_feature = self.compute_densityfeature(rays_chunk, frame_time.unsqueeze(-1))
        # 用 plain/MLP/KAN 密度解码器生成原始预测。
        density = self.density_regressor(rays_chunk, Physparam_feature, frame_time.unsqueeze(-1))
        # 用 rho_bd 范围缩放，再加基准 rho0 得到最终场值。
        return density * (self.rho_bd[1] - self.rho_bd[0]) + self.rho0

    @torch.no_grad()
    def updateEmptyMask(self, gridSize=(200, 200, 200), time_grid=64):
        """
        【暂时不用深究】密集评估整个三维网格，并更新可跳过空间的体素掩码。

        ``gridSize`` 是掩码的 [X,Y,Z] 分辨率，``time_grid`` 是评估的时间采样数。
        本方法修改 ``self.emptyMask``，不返回值。
        """
        # 在密集三维网格上计算非空强度和对应坐标。
        emptiness, dense_xyz = self.getDenseEmpty(gridSize, time_grid)
        # 调换坐标网格第 0/2 轴并复制为连续内存；此变量后续未再使用。
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        # 将强度限制在 [0,1]，调换 x/z 轴，再增加批量和通道轴得 [1,1,D,H,W]。
        emptiness = emptiness.clamp(0, 1).transpose(0, 2).contiguous()[None, None]

        # 设置 3×3×3 的局部池化核。
        ks = 3
        # 【数学直觉】最大池化会膨胀非空区域一圈，降低错删真实场的风险。
        emptiness = F.max_pool3d(emptiness, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        # 【注意】当 emptyMask_thres 为正数时，同时 >=thres 且 <=-thres 的条件无法成立；保留原实现。
        emptiness[(emptiness >= self.emptyMask_thres) & (emptiness <= -self.emptyMask_thres)] = 1
        # 将位于 (-thres,thres) 的小幅值设为 0。
        emptiness[(emptiness < self.emptyMask_thres) & (emptiness > -self.emptyMask_thres)] = 0

        # 用新体素强度创建 EmptyGridMask 并挂到模型上。
        self.emptyMask = EmptyGridMask(self.device, self.aabb, emptiness)

    @torch.no_grad()
    def getDenseEmpty(self, gridSize=None, time_grid=None):
        """
        【暂时不用深究】生成密集三维坐标网格，并逐 x 切片计算非空强度。

        返回 ``emptiness [X,Y,Z]`` 和 ``dense_xyz [X,Y,Z,3]``，坐标已归一化到 [-1,1]。
        """
        # 未指定分辨率时，使用当前模型网格分辨率。
        gridSize = self.gridSize if gridSize is None else gridSize
        # 未指定时间采样数时，使用模型 time_grid。
        time_grid = self.time_grid if time_grid is None else time_grid

        # 【主线】每轴生成 0〜1 等距坐标，meshgrid 取笛卡尔积，再堆成 [X,Y,Z,3]。
        samples = torch.stack(
            torch.meshgrid([torch.linspace(0, 1, g) for g in gridSize], indexing='ij'), dim=-1
        ).to(self.device)
        
        # 将 [0,1] 坐标线性换到特征网格所用的 [-1,1]。
        dense_xyz = samples * 2.0 - 1.0
        # 创建 [X,Y,Z] 零张量存放非空强度。
        emptiness = torch.zeros_like(dense_xyz[..., 0])
        
        # 按 x 轴逐片计算，避免一次展开整个时空网格占用过多显存。
        for i in range(gridSize[0]):
            # 将第 i 个 yz 切片展平为 [Y*Z,3]，计算后再变回 [Y,Z]。
            emptiness[i] = self.compute_emptiness(
                dense_xyz[i].view(-1, 3).contiguous(), time_grid, self.stepSize
            ).view((gridSize[1], gridSize[2]))
        # 返回三维非空强度与坐标网格。
        return emptiness, dense_xyz

    def compute_emptiness(self, xyz_locs, time_grid=64, length=1):
        """
        【暂时不用深究】对一组三维坐标遍历多个时刻，估计其最大非空/占据强度。

        ``xyz_locs`` 为 ``[N,3]`` 归一化坐标；``time_grid`` 是时间采样数 T；
        ``length`` 是将密度转成类不透明度时的积分长度。返回 ``[N]``。
        """
        # 如果已存在旧掩码，只重新评估旧掩码 >0 的坐标。
        if self.emptyMask is not None:
            empty_mask = self.emptyMask.sample_empty(xyz_locs) > 0
        else:
            # 没有旧掩码时，将所有 N 个坐标视为候选点。
            empty_mask = torch.ones_like(xyz_locs[:, 0], dtype=torch.bool)

        # 创建 [N] 零张量存放每个坐标的最终密度强度。
        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        # 只有候选集非空时才运行张量分解查询。
        if empty_mask.any():
            # 取出候选坐标，设其个数为 N。
            xyz_sampled = xyz_locs[empty_mask]
            # 在归一化时间 [-1,1] 上生成 T 个等距时刻。
            time_samples = torch.linspace(-1, 1, time_grid, device=xyz_sampled.device)
            # 读取候选空间点数 N 和时刻数 T。
            N, T = xyz_sampled.shape[0], time_samples.shape[0]
            
            # 将每个空间点复制 T 份，[N,3] -> [N,T,3] -> [N*T,3]。
            xyz_sampled = xyz_sampled.unsqueeze(1).expand(-1, T, -1).contiguous().view(-1, 3)
            # 将 T 个时刻复制 N 份，并展平为 [N*T,1]，与坐标顺序对齐。
            time_samples = time_samples.unsqueeze(0).expand(N, -1).contiguous().view(-1, 1)

            # 查询所有 N*T 时空点的密度潜在特征。
            density_feature = self.compute_densityfeature(xyz_sampled, time_samples)
            # 【注意】原代码将特征切为 ``[..., :-3]`` 后才交给解码器；保留此行为。
            # 解码后将 [N*T] 重排为 [N,T]。
            sigma_feature = self.density_regressor(xyz_sampled, density_feature[..., :-3], time_samples).view(N, T)

            # 对每个空间点取所有 T 个时刻中的最大预测强度。
            sigma_feature = torch.amax(sigma_feature, dim=-1)
            # 经配置激活函数转成密度，并写回候选位置。
            sigma[empty_mask] = self.feature2density(sigma_feature)

        # 【数学直觉】用 1-exp(-sigma*length) 将密度积分换成 [0,1) 类占据/不透明强度。
        return 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])
    
