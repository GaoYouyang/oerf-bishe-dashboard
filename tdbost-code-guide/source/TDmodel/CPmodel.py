"""
【主线】CPmodel：用 CP（CANDECOMP/PARAFAC）思路压缩表示动态三维场。

零基础可以把它理解成：不在每个 ``(x, y, z, t)`` 体素上都存一大堆数，
而是分别学习 x、y、z、t 四根一维特征线。查询一个时空点时，
从四根线上做线性插值，再将四份特征相乘或拼接，得到密度/外观特征。

【三维重建意义】训练会调整这些一维参数，使通过它们重建出的三维折射率/
密度场能够解释相机观测到的 BOS 偏移。
"""

# 【主线】PyTorch 提供张量、可训练参数和自动求导。
import torch
# 【主线】F 是 PyTorch 的函数式神经网工具，下文使用 grid_sample/interpolate。
from torch.nn import functional as F

# 【主线】导入公共基类：射线采样、坐标归一化和物理积分都在基类中。
from TDmodel.TD_Base import TD_Base
# 【注意】下面的 torch 是原代码重复导入；保留它以不改变原文件顺序。
import torch
# 【暂时不用深究】nn 是神经网模块命名空间，用来继承 nn.Module。
import torch.nn as nn
# 【注意】这是对同一个 functional 模块的第二次导入，名称仍是 F。
import torch.nn.functional as F
# 【暂时不用深究】math 只用来计算高斯公式中的常数。
import math
# 【数学直觉】k = sqrt(2π)，是一维高斯分布归一化系数的一部分。
k=math.sqrt(2 * math.pi)

def sigmoid_weight(distance, smooth_factor=5):
        """
        【暂时不用深究】把距离变成 0〜1 之间的平滑权重；当前主线没有调用它。

        ``distance`` 可以是任意形状张量，返回形状不变；``smooth_factor`` 越大，
        以 0.5 为中心的过渡越陡。
        """
        # 这是逻辑斯谛函数：距离越大，权重越小。
        return 1 / (1 + torch.exp(smooth_factor * (distance - 0.5)))
 # Calculate Gaussian weights
def gaussian_weight(dist, sigma=1.0):
    """
    【暂时不用深究】按一维高斯曲线将距离换成权重，当前主线未调用。

    ``dist`` 与返回值形状相同；``sigma`` 控制权重曲线的宽度。
    """
    # 【数学直觉】距离 0 最重要，距离越大权重按指数速度衰减。
    return torch.exp(-0.5 * (dist / sigma)**2) / (k * sigma)

class BilinearInterpolationLayer(nn.Module):
    """
    【暂时不用深究】手写的双线性插值层。

    它用目标坐标周围四个网格值的加权平均估计该点特征。
    本类是备选实现；CPmodel 当前主线实际使用自己的一维插值代码。
    """

    def __init__(self):
        """【暂时不用深究】初始化无可训练参数的插值层。"""
        # 初始化 nn.Module 基类。
        super(BilinearInterpolationLayer, self).__init__()

    def forward(self, input, coords):
        """
        【数学直觉】根据归一化坐标，对四个相邻网格点做双线性插值。

        ``input``：``[N, C, Hin, Win]``，N 是批量，C 是特征通道。
        ``coords``：``[N, Hout, Wout, 2]``，坐标预期在 [-1, 1]。
        返回每个目标坐标的 C 维插值特征。
        """
        #print(input.shape, coords.shape,coords)
        # 【主线】拆出输入的批量、通道、高度和宽度。
        N, C, Hin, Win = input.shape
        # 读取目标坐标网格的高度和宽度。
        Hout, Wout = coords.shape[1], coords.shape[2]
        # 把第 0 个坐标分量从 [-1, 1] 线性换到 [0, 1]，原代码把它命名为 y。
        y = coords[..., 0]/2+0.5  # Corrected indexing
        # 把第 1 个坐标分量同样换到 [0, 1]，原代码把它命名为 x。
        x = coords[..., 1]/2+0.5  # Corrected indexing
        #print(x.min(),x.max(),y.min(),y.max())
        # Scale coordinates from [0, 1] to [0, H-1] for y and [0, W-1] for x
        # 将 [0, 1] 比例换成真实的行索引 [0, Hin-1]。
        y = y * (Hin - 1)
        # 将 [0, 1] 比例换成真实的列索引 [0, Win-1]。
        x = x * (Win - 1)

        # Calculate the coordinates of the 4 pixels to interpolate from
        # 向下取整得到上方/起始行索引，并转为整数类型。
        y0 = torch.floor(y).long()
        # 下方行是上方行的下一行。
        y1 = y0 + 1
        # 向下取整得到左侧/起始列索引。
        x0 = torch.floor(x).long()
        # 右侧列是左侧列的下一列。
        x1 = x0 + 1

        # Clip to range [0, H-1] for y and [0, W-1] for x to not go out of the image boundaries
        # 【注意】clamp 将索引限制在有效边界，防止访问图像之外。
        y0 = torch.clamp(y0, 0, Hin - 1)
        y1 = torch.clamp(y1, 0, Hin - 1)
        x0 = torch.clamp(x0, 0, Win - 1)
        x1 = torch.clamp(x1, 0, Win - 1)

        # Get pixel values
        # 取左上角特征 Ia；前两个冒号表示保留所有批次和通道。
        Ia = input[:, :, y0, x0]
        # 取右上角特征 Ib。
        Ib = input[:, :, y0, x1]
        # 取左下角特征 Ic。
        Ic = input[:, :, y1, x0]
        # 取右下角特征 Id。
        Id = input[:, :, y1, x1]

       

        # Compute cubic weights
        # wa = (1 - (x - x0)**2) * (1 - (y - y0)**2)
        # wb = (1-(x - x1)**2 )* (1 - (y - y0)**2)
        # wc = (1 - (x - x0)**2) * (1-(y - y1)**2)
        # wd = (1-(x - x1)**2) * (1-(y - y1)**2)
        # 【数学直觉】点离某个角越近，该角权重越大；水平和竖直权重相乘。
        wa = (x1 - x) * (y1 - y)
        # 右上角权重。
        wb = (x - x0) * (y1 - y)
        # 左下角权重。
        wc = (x1 - x) * (y - y0)
        # 右下角权重。
        wd = (x - x0) * (y - y0)
        # wa = (x1**2 - x**2) * (y1**2 - y**2)/(x1+x0)/(y1+y0)
        # wb = (x**2 - x0**2) * (y1**2 - y**2)/(x1+x0)/(y1+y0)
        # wc = (x1**2 - x**2) * (y**2 - y0**2)/(x1+x0)/(y1+y0)
        # wd = (x**2 - x0**2) * (y**2 - y0**2)/(x1+x0)/(y1+y0)
        # # wa = sigmoid_weight(torch.sqrt((x1.float() - x)**2 + (y1.float() - y)**2))
        # # wb = sigmoid_weight(torch.sqrt((x - x0.float())**2 + (y1.float() - y)**2))
        # # wc = sigmoid_weight(torch.sqrt((x1.float() - x)**2 + (y - y0.float())**2))
        # # wd = sigmoid_weight(torch.sqrt((x - x0.float())**2 + (y - y0.float())**2))
        # w=wa+wb+wc+wd
        # wa,wb,wc,wd=wa/w,wb/w,wc/w,wd/w
        # Compute output
        # 给权重增加通道轴，再对四个角的特征做加权和。
        output = wa.unsqueeze(1) * Ia + wb.unsqueeze(1) * Ib + wc.unsqueeze(1) * Ic + wd.unsqueeze(1) * Id
        #print(Ia.shape,output.shape)
        # 返回目标坐标上的插值特征。
        return output

# Example usage:
# Assume `input_tensor` is the image tensor and `coords_tensor` is the coordinate tensor generated somehow
# input_tensor shape should be [batch_size, channels, height, width]
# coords_tensor shape should be [batch_size, out_height, out_width, 2] and contains normalized coords in [0, 1]
class BiCubic(nn.Module):
    """
    【暂时不用深究】原作者保留的手写双三次插值实验层。

    双三次插值会参考周围 4×4 个点，理论上比四点双线性插值更平滑。
    【注意】CPmodel 当前不实例化本类；这段代码仅按原样保留，不代表公式已被主流程验证。
    """

    def __init__(self):
        """【暂时不用深究】初始化无可训练参数的插值层。"""
        # 初始化 nn.Module 基类。
        super(BiCubic, self).__init__()

    def forward(self, input, coords):
        """
        【暂时不用深究】用 4×4 邻域特征估计目标坐标的值。

        ``input`` 形状为 ``[N, C, Hin, Win]``，``coords`` 为 ``[N, Hout, Wout, 2]``。
        坐标从 [-1, 1] 换算为网格索引，输出是对应坐标的 C 通道特征。
        """
        #print(input.shape, coords.shape,coords)
        # 拆出输入张量的批量、通道、高和宽。
        N, C, Hin, Win = input.shape
        # 读取目标坐标网格大小。
        Hout, Wout = coords.shape[1], coords.shape[2]
        # 把第 0 个归一化坐标从 [-1, 1] 换到 [0, 1]。
        y = coords[..., 0]/2+0.5  # Corrected indexing
        # 把第 1 个归一化坐标从 [-1, 1] 换到 [0, 1]。
        x = coords[..., 1]/2+0.5  # Corrected indexing
        #print(x.min(),x.max(),y.min(),y.max())
        # Scale coordinates from [0, 1] to [0, H-1] for y and [0, W-1] for x
        # 换算成连续行索引。
        y = y * (Hin - 1)
        # 换算成连续列索引。
        x = x * (Win - 1)

        # Calculate the coordinates of the 4 pixels to interpolate from
        # 以 floor(y) 为中心附近索引之一。
        y0 = torch.floor(y).long()
        # y1 比 y0 小 1，是外圈邻居。
        y1 = y0 - 1
        # y2 比 y0 大 1。
        y2 = y0 + 1
        # y3 比 y0 大 2，是另一侧外圈邻居。
        y3 = y0 + 2
        # x 方向同样构造四个邻近索引。
        x0 = torch.floor(x).long()
        x1 = x0 - 1
        x2 = x0 + 1
        x3 = x0 + 2
        # Clip to range [0, H-1] for y and [0, W-1] for x to not go out of the image boundaries
        # y0 = torch.clamp(y0, 0, Hin - 1)
        # y1 = torch.clamp(y1, 0, Hin - 1)
        # x0 = torch.clamp(x0, 0, Win - 1)
        # x1 = torch.clamp(x1, 0, Win - 1)
        # y0 = torch.clamp(y0, 0, Hin - 1)
        # y1 = torch.clamp(y1, 0, Hin - 1)
        # y2 = torch.clamp(y2, 0, Hin - 1)
        # y3 = torch.clamp(y3, 0, Hin - 1)
        # x0 = torch.clamp(x0, 0, Win - 1)
        # x1 = torch.clamp(x1, 0, Win - 1)
        # x2 = torch.clamp(x2, 0, Win - 1)
        # x3 = torch.clamp(x3, 0, Win - 1)
        # Get pixel values#0，2小圈，1，3大圈
        # 【注意】下面依次取 4×4 邻域的 16 份特征；原代码的边界 clamp 处于注释状态。
        Ia = input[:, :, y0, x0]#- -
        Ib = input[:, :, y0, x2]#- +
        Ic = input[:, :, y2, x0]#+ -
        Id = input[:, :, y2, x2]#+ +
        Ia1=input[:, :, y0, x1]#- -
        Ia2=input[:, :, y0, x3]#- +
        Ib1=input[:, :, y1, x0]#- -
        Ib2=input[:, :, y1, x1]#- -
        Ib3=input[:, :, y1, x2]#- +
        Ib4=input[:, :, y1, x3]#- +
        Ic1=input[:, :, y2, x1]#+ -
        Ic2=input[:, :, y2, x3]#+ +
        Id1=input[:, :, y3, x0]#+ -
        Id2=input[:, :, y3, x1]#+ -
        Id3=input[:, :, y3, x2]#+ +
        Id4=input[:, :, y3, x3]#+ +
        #print(y.shape,y0.shape)
        # 【数学直觉】计算 y 方向四个邻居对目标点的三次多项式权重。
        wy0=1.5*(y-y0)**3-2.5*(y-y0)**2+1
        wy1=-0.5*(y-y1)**3+2.5*(y-y1)**2-4*(y-y1)+2
        wy2=1.5*(y2-y)**3-2.5*(y2-y)**2+1
        wy3=-0.5*(y3-y)**3+2.5*(y3-y)**2-4*(y3-y)+2
        # 计算 x 方向四个邻居的三次多项式权重。
        wx0=1.5*(x-x0)**3-2.5*(x-x0)**2+1
        wx1=-0.5*(x-x1)**3+2.5*(x-x1)**2-4*(x-x1)+2
        wx2=1.5*(x2-x)**3-2.5*(x2-x)**2+1
        wx3=-0.5*(x3-x)**3+2.5*(x3-x)**2-4*(x3-x)+2

        # 将 16 份邻域特征与各自 x/y 权重组合，得到目标点输出。
        # 【注意】完全保留原作者此处的表达式，未对其中 ``Ib+wy0*wx2`` 做改写。
        output = Ia*wy0*wx0+Ib+wy0*wx2+Ic*wy2*wx0+Id*wy2*wx2+\
                Ia1*wy0*wx1+Ia2*wy0*wx3+Ib1*wy1*wx0+Ib2*wy1*wx1+Ib3*wy1*wx2+Ib4*wy1*wx3+\
                Ic1*wy2*wx1+Ic2*wy2*wx3+Id1*wy3*wx0+Id2*wy3*wx1+Id3*wy3*wx2+Id4*wy3*wx3
        #print((y).mean(),(y).max(),(y).min())
        # 返回手写插值结果。
        return output



class CPmodel(TD_Base):
    """
    【主线】用四组一维向量（x/y/z/t）表示时变三维场的 CP 分解模型。

    每组特征实际形状为 ``[1, C, L]``：1 是批量，C 是分解分量数，
    L 是对应 x/y/z 网格长度或时间网格长度。查询 N 个 ``(x,y,z,t)`` 点后，
    得到 ``[4, C, N]`` 特征，再融合并投影到密度维度 ``density_dim`` 或外观维度 ``app_dim``。

    【数学直觉】如果选 multiply，一个时空点的第 c 个隐特征类似
    ``X_c(x) * Y_c(y) * Z_c(z) * T_c(t)``；这就是低秩 CP 表示的核心。
    """

    def __init__(self, aabb, gridSize, device, time_grid, near_far, **kargs):
        """
        【主线】将所有配置交给 TD_Base，基类会再回调本类的 ``init_planes``。

        ``aabb`` 是 ``[2,3]`` 包围盒最小/最大坐标；``gridSize`` 是 x/y/z 体素分辨率；
        ``time_grid`` 是时间轴采样数；``near_far`` 是沿射线采样的近/远边界。
        """
        # 执行公共基类的初始化逻辑。
        super().__init__(aabb, gridSize, device, time_grid, near_far, **kargs)

    def init_planes(self, res, device):
        """
        【主线】创建密度和外观两套 x/y/z/t 一维特征，并创建特征投影层。

        ``res`` 为三维网格分辨率，``device`` 指定 CPU/GPU。此方法主要修改模型内部状态，
        不直接返回张量。
        """
        # 【注意】原代码将备选层记为 F.grid_sample；CP 主特征查询在后文手写一维插值。
        self.bilinear_layer =F.grid_sample#BilinearInterpolationLayer()#BiCubic() #F.grid_sample
        # 【主线】创建用来重建密度/折射率的四根可训练特征线。
        self.density_plane = self.init_one_hexplane(
            self.density_n_comp, self.gridSize, device
        )
        # 创建用来重建外观或其他物理量的另一套特征线。
        self.app_plane = self.init_one_hexplane(
            self.app_n_comp, self.gridSize, device
        )



        # We use density_basis_mat and app_basis_mat to project extracted features from HexPlane to density_dim/app_dim.
        # density_basis_mat and app_basis_mat are linear layers, whose input dims are calculated based on the fusion methods.
        # 【主线】根据融合方式计算投影层需要接收多少维特征。
        if self.fusion_two == "concat":
            # 两级都拼接时，x/y/z/t 四份 C 维特征保留为 4C 维。
            if self.fusion_one == "concat":
                # 密度投影：4C -> density_dim，不使用偏置。
                self.density_basis_mat = torch.nn.Linear(
                    self.density_n_comp[0] * 4, self.density_dim, bias=False
                ).to(device)
                # 外观投影：4C -> app_dim。
                self.app_basis_mat = torch.nn.Linear(
                    self.app_n_comp[0] * 4, self.app_dim, bias=False
                ).to(device)
            else:
                # 若第一级不拼接，输入保持 C 维。
                self.density_basis_mat = torch.nn.Linear(
                    self.density_n_comp[0], self.density_dim, bias=False
                ).to(device)
                # 创建 C -> app_dim 的外观投影。
                self.app_basis_mat = torch.nn.Linear(
                    self.app_n_comp[0], self.app_dim, bias=False
                ).to(device)
        else:
            # 第二级不拼接时，三/四份特征已被相乘等操作融成 C 维。
            self.density_basis_mat = torch.nn.Linear(
                self.density_n_comp[0], self.density_dim, bias=False
            ).to(device)
            # 创建 C -> app_dim 的外观投影。
            self.app_basis_mat = torch.nn.Linear(
                self.app_n_comp[0], self.app_dim, bias=False
            ).to(device)

        # Initialize the basis matrices
        # 【暂时不用深究】初始化不应记入梯度图，因此使用 no_grad。
        with torch.no_grad():
            # 先生成与密度投影权重同形状的全 1 张量，再除以 density_dim。
            weights = torch.ones_like(self.density_basis_mat.weight) / float(
                self.density_dim
            )
            # 把初始权重复制进线性层；外观投影仍使用 PyTorch 默认初始化。
            self.density_basis_mat.weight.copy_(weights)

    def init_one_hexplane(self, n_component, gridSize, device):
        """
        【主线】初始化一套 x/y/z/t 四根可训练一维特征线。

        ``n_component[0]`` 是 CP 秩/特征通道 C，``gridSize`` 提供 x/y/z 三轴长度，
        ``self.time_grid`` 是 t 轴长度。返回 ParameterList，内含形状为
        ``[1,C,X]``、``[1,C,Y]``、``[1,C,Z]``、``[1,C,T]`` 的四个参数张量。

        【三维重建意义】这四根特征线是 CP 模型压缩存储整个动态体素场的核心参数。
        """
        # 先创建普通 Python 列表，稍后把每根特征线放进去。
        plane_coef = []

        # for i in range(len(self.vecMode)):
        #     vec_id = self.vecMode[i]
        #     mat_id_0, mat_id_1 = self.matMode[i]

        #     plane_coef.append(
        #         torch.nn.Parameter(
        #             self.init_scale
        #             * torch.randn(
        #                 (1, n_component[i], gridSize[mat_id_0], gridSize[mat_id_1])
        #             )
        #             + self.init_shift
        #         )
        #     )
        #     line_time_coef.append(
        #         torch.nn.Parameter(
        #             self.init_scale
        #             * torch.randn((1, n_component[i], gridSize[vec_id], self.time_grid))
        #             + self.init_shift
        #         )
        #     )
        # 【主线】遍历 x/y/z 三个空间轴的网格长度。
        for i in range(len(gridSize)):
            # 为第 i 个空间轴追加一个可训练张量。
            plane_coef.append(
                # Parameter 告诉 PyTorch：优化器需要更新这些数值。
                torch.nn.Parameter(
                    # 【数学直觉】从小幅度高斯噪声开始，再加上统一偏移。
                    self.init_scale
                    * torch.randn(
                        # 形状 [1, C, Li]：Li 是当前空间轴的网格数。
                        (1, n_component[0], gridSize[i])
                    )
                    + self.init_shift
                )
            )
        # 【主线】空间三轴之后，再追加时间 t 轴的可训练特征线。
        plane_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn(
                        # 形状 [1, C, T]，T 是 self.time_grid。
                        (1, n_component[0], self.time_grid)
                    )
                    + self.init_shift
                )
            )
        # 将列表包装成能正确登记参数的 ParameterList，并移到指定设备。
        return torch.nn.ParameterList(plane_coef).to(device)

    def get_optparam_groupsrho(self, cfg, lr_scale=1.0):
        """
        【主线】组装“只训练密度/折射率分支”的优化器参数组。

        ``cfg`` 保存网格与小网络的学习率；``lr_scale`` 对当前学习率统一缩放。
        返回的列表可直接交给 PyTorch 优化器。
        """
        # 创建密度 CP 特征线和密度投影层两组参数。
        grad_vars = [
            {
                # 参数对象：x/y/z/t 密度特征线。
                "params": self.density_plane,
                # 实际学习率可被 lr_scale 缩放。
                "lr": lr_scale * cfg.lr_density_grid,
                # 保留原始学习率，供训练调度代码查看。
                "lr_org": cfg.lr_density_grid,
            },
            {
                # 密度基矩阵（线性投影层）的可训练参数。
                "params": self.density_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_density_nn,
                "lr_org": cfg.lr_density_nn,
            },
        ]

        # 【注意】plain 模式的 density_regressor 只是函数，没有可训练参数；MLP/KAN 才追加。
        if isinstance(self.density_regressor, torch.nn.Module):
            # 追加密度解码网络参数组。
            grad_vars += [
                {
                    "params": self.density_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_density_nn,
                    "lr_org": cfg.lr_density_nn,
                }
            ]
        
        # 返回密度分支的完整优化参数列表。
        return grad_vars
    
    def get_optparam_groupsgrad(self, cfg, lr_scale=1.0):
        """
        【主线】组装“外观/梯度分支”的优化器参数组。

        返回 app_plane、app_basis_mat，以及可选 app_regressor 的学习率配置。
        """
        # 先收集外观 CP 特征线和外观投影层。
        grad_vars = [
    
            {
                # 外观/物理量的 x/y/z/t 特征线。
                "params": self.app_plane,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                # 外观基矩阵的所有参数。
                "params": self.app_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_app_nn,
                "lr_org": cfg.lr_app_nn,
            },
        ]

        # 只有 app_regressor 是真正 nn.Module 时才有 parameters() 可以训练。
        if isinstance(self.app_regressor, torch.nn.Module):
            # 追加外观解码网络参数组。
            grad_vars += [
                {
                    "params": self.app_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_app_nn,
                    "lr_org": cfg.lr_app_nn,
                }
            ]
        # 返回外观/梯度分支参数列表。
        return grad_vars

    def get_optparam_groups(self, cfg, lr_scale=1.0):
        """
        【主线】组装主训练阶段的参数组：密度、外观特征、投影层和部分解码器。

        每个字典都有 ``params``、当前 ``lr`` 和原始 ``lr_org``。
        """
        # 主优化器同时收集密度与外观两套 CP 特征和投影层。
        grad_vars = [
            
            {
                # 密度特征线使用密度网格学习率。
                "params": self.density_plane,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            
            {
                # 外观特征线使用外观网格学习率。
                "params": self.app_plane,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                # 密度投影层使用密度网络学习率。
                "params": self.density_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_density_nn,
                "lr_org": cfg.lr_density_nn,
            },
            {
                # 【注意】外观投影的当前 lr 被原代码额外除以 10，lr_org 仍保留未缩小值。
                "params": self.app_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_app_nn/10,
                "lr_org": cfg.lr_app_nn,
            },
        ]

        # if isinstance(self.app_regressor, torch.nn.Module):
        #     grad_vars += [
        #         {
        #             "params": self.app_regressor.parameters(),
        #             "lr": lr_scale * cfg.lr_app_nn,
        #             "lr_org": cfg.lr_app_nn,
        #         }
        #     ]

        # 密度解码器是 MLP/KAN 时，将它加入主优化器。
        if isinstance(self.density_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.density_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_density_nn,
                    "lr_org": cfg.lr_density_nn,
                }
            ]
        # 畸变回归器在基类中定义为 MLP，因此通常会追加其参数。
        if isinstance(self.distortion_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.distortion_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_density_nn,
                    "lr_org": cfg.lr_density_nn,
                }
            ]
        # 返回主训练所需的全部参数组。
        return grad_vars  

    def get_optparam_groupsapp(self, cfg, lr_scale=1.0):
        """
        【主线】只收集 app_regressor 解码网络的参数，用于单独微调。

        如果 app_regressor 是 plain 函数，则返回空列表。
        """
        # 从空参数组开始。
        grad_vars = []

        # 只有神经网模块才包含可训练参数。
        if isinstance(self.app_regressor, torch.nn.Module):
            # 追加外观解码器参数及其学习率。
            grad_vars += [
                {
                    "params": self.app_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_app_nn,
                    "lr_org": cfg.lr_app_nn,
                }
            ]

        # 返回只包含 app_regressor 的参数列表。
        return grad_vars 

    def compute_densityfeature(
        self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor
    ) -> torch.Tensor:
        """
        【主线】在 CP 特征线上查询 N 个时空点，并生成密度解码特征。

        ``xyz_sampled``：``[N,3]`` 的归一化坐标，各分量通常在 [-1,1]。
        ``frame_time``：``[N,1]`` 的归一化时间。
        返回：``[N,density_dim]`` 的密度潜在特征，之后由 density_regressor 转成物理密度。
        """
        # Prepare coordinates for grid sampling.
        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        # 【主线】将 x/y/z/t 四个分量堆成 ``[4,N]``，每行对应一根特征线。
        plane_coords = (
            torch.stack(
                (
                    # 所有采样点的 x 坐标。
                    xyz_sampled[..., 0],
                    # 所有采样点的 y 坐标。
                    xyz_sampled[..., 1],
                    # 所有采样点的 z 坐标。
                    xyz_sampled[..., 2],
                    # 所有采样点的 t 坐标。
                    frame_time[..., 0],
                )
            )
            #.detach()
            .view(4, -1)
        )

        # 用列表依次收集 x/y/z/t 四根线的插值特征。
        plane_feat_list= []
        # Extract features from six feature planes.
        # 【主线】遍历四根密度特征线。
        for idx in range(len(self.density_plane)):

            # 去掉大小为 1 的批量轴：[1,C,L] -> [C,L]。
            plane = self.density_plane[idx].squeeze(0)  # (C, L)
            # 读出特征通道数 C 和当前轴的网格长度 L。
            C, L = plane.shape
            # 取与当前特征线对应的 N 个归一化坐标。
            plane_coord  = plane_coords [idx][:]  # (N,2) 取前两分量或根据实际决定
            # 如果每 plane 对应的是单维度坐标，取第一列为索引
            # 【数学直觉】先限制到 [-1,1]，再线性映射到连续索引 [0,L-1]。
            idxf = (torch.clamp(plane_coord[:], -1.0, 1.0)+1)/2 * (L - 1)  # (N,)
            # 向下取整得到左侧网格索引，并限制在边界内。
            idx0 = torch.floor(idxf).long().clamp(0, L - 1)
            # 右侧网格是 idx0+1；最右边界处仍限制为 L-1。
            idx1 = (idx0 + 1).clamp(0, L - 1)
            # 小数部分 w 表示目标点离左网格有多远，形状变为 [1,N]。
            w = (idxf - idx0.to(idxf.dtype)).unsqueeze(0)  # (1, N)
            # 把 N 个索引复制到 C 个通道，形状 [C,N]。
            idx0_exp = idx0.unsqueeze(0).expand(C, -1)  # (C, N)
            # 对右侧索引做同样的通道扩展。
            idx1_exp = idx1.unsqueeze(0).expand(C, -1)
            # gather 按每个通道的 idx0 取左网格值，得 [C,N]。
            v0 = plane.gather(1, idx0_exp)  # (C, N)
            # 取右网格值。
            v1 = plane.gather(1, idx1_exp)  # (C, N)
            # 【数学直觉】一维线性插值：左值权重 1-w，右值权重 w。
            sampled_vec = (1.0 - w) * v0 + w * v1  # (C, N)
            # 收集当前 x/y/z/t 轴查到的特征。
            plane_feat_list.append(sampled_vec)
        # 将四份 [C,N] 堆叠为 [P,C,N]，CP 模型中 P=4。
        plane_feat = torch.stack(plane_feat_list, dim=0)  # (P, C, N)
            

        # Fusion One
        # 【注意】这一级两个支路当前都只将 plane_feat 原样交给下一级。
        if self.fusion_one == "multiply":
            inter = plane_feat 
        elif self.fusion_one == "concat":
            inter = plane_feat
        else:
            # 对未实现的融合名称立即报错。
            raise NotImplementedError("no such fusion type")

        # Fusion Two
        # 【主线】第二级真正将 x/y/z/t 四份特征融合。
        if self.fusion_two == "multiply":
            # 沿 P 轴相乘：[P,C,N] -> [C,N]，对应 CP 的外积/可分离直觉。
            inter = torch.prod(inter, dim=0)
        elif self.fusion_two == "concat":
            # 将 P 和 C 展平为一个特征轴：[P,C,N] -> [P*C,N]。
            inter = inter.view(-1, inter.shape[-1])
        else:
            raise NotImplementedError("no such fusion type")

        # 先转置为 [N,特征数]，再用基矩阵投影到 [N,density_dim]。
        inter = self.density_basis_mat(inter.T)  # Feature Projection

        # 返回交给密度解码器的潜在特征。
        return inter

    def compute_appfeature(
        self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor
    ) -> torch.Tensor:
        """
        【主线】在外观 CP 特征线上查询 N 个时空点并投影。

        ``xyz_sampled`` 为 ``[N,3]``，``frame_time`` 为 ``[N,1]``，两者通常已归一化到 [-1,1]。
        返回 ``[N,app_dim]``，供 app_regressor 解码为外观、位移或其他三维物理量。
        """
        # Prepare coordinates for grid sampling.
        # 将 x/y/z/t 四个坐标分量堆叠为 [4,N]。
        plane_coords = (
            torch.stack(
                (
                    xyz_sampled[..., 0],
                    xyz_sampled[..., 1],
                    xyz_sampled[..., 2],
                    frame_time[..., 0],
                )
            )
            #.detach()
            .view(4, -1)
        )

        # 创建用于收集四根外观特征线查询结果的列表。
        plane_feat_list= []
        # 【主线】下面的一维插值与 compute_densityfeature 完全同理，只是换成 app_plane。
        for idx in range(len(self.app_plane)):
            # [1,C,L] -> [C,L]。
            plane = self.app_plane[idx].squeeze(0)  # (C, L)
            # 读取通道数 C 和当前轴长度 L。
            C, L = plane.shape
            # 取当前 x/y/z/t 轴对应的 N 个坐标。
            plane_coord  = plane_coords [idx][:]  # (N,2) 取前两分量或根据实际决定
            # 如果每 plane 对应的是单维度坐标，取第一列为索引
            # 将 [-1,1] 坐标换算成 [0,L-1] 的连续索引。
            idxf = (torch.clamp(plane_coord[:], -1.0, 1.0)+1)/2 * (L - 1)  # (N,)
            # 计算左邻居整数索引。
            idx0 = torch.floor(idxf).long().clamp(0, L - 1)
            # 计算右邻居整数索引。
            idx1 = (idx0 + 1).clamp(0, L - 1)
            # 计算相对左邻居的小数距离 w，并增加通道轴。
            w = (idxf - idx0.to(idxf.dtype)).unsqueeze(0)  # (1, N)
            # 将左索引扩展到每个特征通道。
            idx0_exp = idx0.unsqueeze(0).expand(C, -1)  # (C, N)
            # 将右索引扩展到每个特征通道。
            idx1_exp = idx1.unsqueeze(0).expand(C, -1)
            # 取左网格特征。
            v0 = plane.gather(1, idx0_exp)  # (C, N)
            # 取右网格特征。
            v1 = plane.gather(1, idx1_exp)  # (C, N)
            # 对左右特征做线性插值。
            sampled_vec = (1.0 - w) * v0 + w * v1  # (C, N)
            # 收集当前轴的特征。
            plane_feat_list.append(sampled_vec)
        # 四份 [C,N] 堆叠为 [P,C,N]，P=4。
        plane_feat = torch.stack(plane_feat_list, dim=0)  # (P, C, N)
            

        # Fusion One
        # 【注意】第一级当前两个允许分支都不改变张量。
        if self.fusion_one == "multiply":
            inter = plane_feat 
        elif self.fusion_one == "concat":
            inter = plane_feat
        else:
            raise NotImplementedError("no such fusion type")

        # Fusion Two
        # 【主线】第二级选择沿 x/y/z/t 轴相乘或拼接。
        if self.fusion_two == "multiply":
            # [P,C,N] -> [C,N]。
            inter = torch.prod(inter, dim=0)
        elif self.fusion_two == "concat":
            # [P,C,N] -> [P*C,N]。
            inter = inter.view(-1, inter.shape[-1])
        else:
            raise NotImplementedError("no such fusion type")

        # 转置为样本在前的 [N,特征数]，再投影成 [N,app_dim]。
        inter = self.app_basis_mat(inter.T)  # Feature Projection

        # 返回外观/物理解码器的输入特征。
        return inter

    def TV_loss_density(self, reg, reg2=None):
        """
        【主线】累加密度特征线的 TV（总变分）正则项。

        ``reg`` 是外部传入的正则函数；TV 鼓励相邻网格变化平滑，
        可减少三维重建中的噪点。``reg2`` 在此 CP 实现中保留但未用。
        """
        # 用 Python 数值 0 作为累加起点。
        total = 0
        # 未传 reg2 时将它指向 reg，但下文并未调用 reg2。
        if reg2 is None:
            reg2 = reg
        # 遍历 x/y/z/t 四根密度特征线。
        for idx in range(len(self.density_plane)):
            # 计算当前特征线的正则值并加入总和。
            total = (
                total + reg(self.density_plane[idx])
            )
        # 返回标量正则损失。
        return total

    def TV_loss_app(self, reg, reg2=None):
        """
        【主线】累加外观特征线的 TV 正则项。

        【注意】原代码此方法末尾没有 ``return total``，因此 Python 会返回 None。
        本注释版严格保留该行为，不在导读中修 bug。
        """
        # 初始化累加器。
        total = 0
        # 未提供第二正则器时复用 reg；此处后续也未使用 reg2。
        if reg2 is None:
            reg2 = reg
        # 遍历外观特征线。
        for idx in range(len(self.app_plane)):
            # 把每根特征线的 TV 正则加入 total。
            total = total + reg(self.app_plane[idx]) 

    def L1_loss_density(self):
        """
        【主线】计算密度参数的 L1 正则，鼓励更多参数接近 0。

        【数学直觉】L1 是绝对值平均，常用来鼓励稀疏表示。
        【注意】CPmodel 的 init_planes 未定义 ``density_line_time``；保留原代码引用，不修改行为。
        """
        # 初始化累加器。
        total = 0
        # 遍历密度特征线。
        for idx in range(len(self.density_plane)):
            # 累加当前 plane 及原代码所引用 line_time 的绝对值平均。
            total = (
                total
                + torch.mean(torch.abs(self.density_plane[idx]))
                + torch.mean(torch.abs(self.density_line_time[idx]))
            )
        # 返回 L1 正则总和。
        return total

    def L1_loss_app(self):
        """
        【主线】计算外观参数的 L1 正则。

        【注意】原代码遍历 ``density_plane`` 长度，并引用 CPmodel 中未初始化的
        ``app_line_time``；此处只做导读标注，不修正。
        """
        # 初始化累加器。
        total = 0
        # 按原代码遍历 density_plane 的数量。
        for idx in range(len(self.density_plane)):
            # 累加 app_plane 和 app_line_time 的绝对值平均。
            total = (
                total
                + torch.mean(torch.abs(self.app_plane[idx]))
                + torch.mean(torch.abs(self.app_line_time[idx]))
            )
        # 返回 L1 正则总和。
        return total

    @torch.no_grad()
    def up_sampling_planes(self, plane_coef, res_target, time_grid):
        """
        【主线】将一维 CP 特征线插值到更高网格分辨率。

        ``plane_coef`` 含 x/y/z/t 四根特征线；``res_target[0]`` 在原代码中作为
        三根空间线的统一目标长度，``time_grid`` 是时间线目标长度。
        返回原代码中的两份同一 ParameterList。
        """
        # 【暂时不用深究】装饰器 no_grad 表示上采样不建立反向传播图。
        # 依次上采样 x/y/z 三根空间特征线。
        for i in range(3):
            # 将插值结果重新包装为可训练 Parameter。
            plane_coef[i] = torch.nn.Parameter(
                # linear 模式对一维序列做线性插值。
                F.interpolate(
                    # .data 取当前参数数值，不追踪旧梯度图。
                    plane_coef[i].data,
                    # 【注意】原实现对三个空间轴都使用 res_target[0]。
                    size=res_target[0],
                    mode="linear",
                    # 决定输入输出两端网格点是否严格对齐。
                    align_corners=self.align_corners,
                )
            )
        # 单独上采样第 4 根（时间 t）特征线。
        plane_coef[3] = torch.nn.Parameter(
            F.interpolate(
                plane_coef[3].data,
                # 时间轴目标长度由 time_grid 指定。
                size=time_grid,
                mode="linear",
                align_corners=self.align_corners,
            )
        )    

        # 【注意】原代码返回两次同一对象，供下游分别赋给 plane 和 line_time 名称。
        return plane_coef, plane_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target, time_grid):
        """
        【主线】同时提高外观与密度 CP 特征的分辨率，并更新射线采样步长。

        这通常用于“先粗后细”训练：先在低分辨率学大体结构，再扩大网格学细节。
        """
        # 上采样外观特征；原函数返回两份同一对象并分别赋值。
        self.app_plane, self.app_line_time = self.up_sampling_planes(
            self.app_plane, res_target, time_grid
        )
        # 上采样密度特征。
        self.density_plane, self.density_line_time = self.up_sampling_planes(
            self.density_plane, res_target, time_grid
        )

        # 【主线】网格分辨率变化后，重新计算体素尺寸、射线步长和采样数。
        self.update_stepSize(res_target)
        # 在终端打印新分辨率，便于监控训练阶段。
        print(f"upsamping to {res_target}")
