"""
【主线】MMmodel：用“平面 × 线-时间平面”的矩阵-矩阵（Matrix-Matrix）分解表示动态三维场。

零基础可以把它理解成：不直接保存完整的 ``(x,y,z,t)`` 四维大表，而是拆成
``xy + zt``、``xz + yt``、``yz + xt`` 三对二维特征平面。查询一个时空点时，
在六个平面上做双线性插值，再将每对特征以相乘、相加或拼接的方式融合。

【三维重建意义】训练会调整这六张低维特征平面，使通过它们重建出的三维折射率/
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
    本类是备选实现；MMmodel 当前主线实际将 ``F.grid_sample`` 赋给插值器。
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
    【注意】MMmodel 当前不实例化本类；这段代码仅按原样保留，不代表公式已被主流程验证。
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



class MMmodel(TD_Base):
    """
    【主线】用六张二维特征平面表示时变三维场的 Matrix-Matrix 分解模型。

    三个空间平面是 ``xy``、``xz``、``yz``，三个配对的空间-时间平面是
    ``tz``、``ty``、``tx``。每对平面查询后先做 fusion_one，三对结果再做 fusion_two。

    输入 N 个归一化时空点 ``(x,y,z,t)``，模型可以输出 ``[N,density_dim]``
    密度特征或 ``[N,app_dim]`` 外观特征。

    【数学直觉】完整四维体素表很大，而六张二维表小得多；
    网络用这些低维因子组合出高维场，这就是“张量分解”在三维重建中的作用。
    """

    def __init__(self, aabb, gridSize, device, time_grid, near_far, **kargs):
        """
        【主线】将网格、时间和射线配置交给 TD_Base 完成公共初始化。

        ``aabb`` 是 ``[2,3]`` 包围盒边界；``gridSize`` 是 x/y/z 分辨率；
        ``time_grid`` 是时间轴分辨率；``near_far`` 是沿光线采样的近远边界。
        """
        # 基类会设置坐标系、采样步长、解码网络，并回调本类 init_planes。
        super().__init__(aabb, gridSize, device, time_grid, near_far, **kargs)

    def init_planes(self, res, device):
        """
        【主线】创建密度和外观两套六平面特征，以及它们的线性投影层。

        ``density_plane/app_plane`` 保存 xy/xz/yz 空间平面；
        ``density_line_time/app_line_time`` 保存 tz/ty/tx 空间-时间平面。
        """
        # 【主线】当前实际插值器是 PyTorch 的 grid_sample，后面两个名称是原作者保留的备选。
        self.bilinear_layer =F.grid_sample#BilinearInterpolationLayer()#BiCubic() #F.grid_sample
        # 创建密度/折射率分支的三对特征平面。
        self.density_plane, self.density_line_time = self.init_one_hexplane(
            self.density_n_comp, self.gridSize, device
        )
        # 创建外观/其他物理量分支的三对特征平面。
        self.app_plane, self.app_line_time = self.init_one_hexplane(
            self.app_n_comp, self.gridSize, device
        )

        # 【注意】三对特征要相乘/相加时，各对的通道数必须相同才能逐元素运算。
        if (
            self.fusion_two != "concat"
        ):  # if fusion_two is not concat, then we need dimensions from each paired planes are the same.
            # 检查第 1 对与第 2 对的 app 通道数一致。
            assert self.app_n_comp[0] == self.app_n_comp[1]
            # 检查第 1 对与第 3 对的 app 通道数一致。
            assert self.app_n_comp[0] == self.app_n_comp[2]

        # We use density_basis_mat and app_basis_mat to project extracted features from HexPlane to density_dim/app_dim.
        # density_basis_mat and app_basis_mat are linear layers, whose input dims are calculated based on the fusion methods.
        # 【主线】根据两级融合方式，确定基矩阵的输入维数。
        if self.fusion_two == "concat":
            # 三对结果拼接时，各对通道数会相加。
            if self.fusion_one == "concat":
                # 【数学直觉】每对两平面也拼接，因此输入是 2 * sum(C_i)。
                self.density_basis_mat = torch.nn.Linear(
                    sum(self.density_n_comp) * 2, self.density_dim, bias=False
                ).to(device)
                # 外观基矩阵做 2*sum(app_C_i) -> app_dim 投影。
                self.app_basis_mat = torch.nn.Linear(
                    sum(self.app_n_comp) * 2, self.app_dim, bias=False
                ).to(device)
            else:
                # 每对先相乘/相加后仍是 C_i 维，三对拼接后是 sum(C_i)。
                self.density_basis_mat = torch.nn.Linear(
                    sum(self.density_n_comp), self.density_dim, bias=False
                ).to(device)
                # 创建 sum(app_C_i) -> app_dim 投影。
                self.app_basis_mat = torch.nn.Linear(
                    sum(self.app_n_comp), self.app_dim, bias=False
                ).to(device)
        else:
            # 第二级相乘/相加会把三对融成单份 C 维特征。
            self.density_basis_mat = torch.nn.Linear(
                self.density_n_comp[0], self.density_dim, bias=False
            ).to(device)
            # 对外观特征做 C -> app_dim 投影。
            self.app_basis_mat = torch.nn.Linear(
                self.app_n_comp[0], self.app_dim, bias=False
            ).to(device)

        # Initialize the basis matrices
        # 【暂时不用深究】参数初始化不应进入反向传播图。
        with torch.no_grad():
            # 创建与密度基矩阵同形状的常数初值。
            weights = torch.ones_like(self.density_basis_mat.weight) / float(
                self.density_dim
            )
            # 将这些初值复制到密度投影权重；app 投影仍使用默认初始化。
            self.density_basis_mat.weight.copy_(weights)

    def init_one_hexplane(self, n_component, gridSize, device):
        """
        【主线】初始化一套三个空间平面和三个空间-时间平面。

        第 i 对形状为 ``[1,C_i,S_a,S_b]`` 和 ``[1,C_i,S_c,T]``，其中 ``a,b,c``
        是互补的空间轴。返回两个 ParameterList，分别存放三张空间平面和三张时空平面。
        """
        # 创建两个普通列表，用来分别收集空间平面和时空平面。
        plane_coef, line_time_coef = [], []

        # 【主线】vecMode=[2,1,0]，因此共遍历 z、y、x 三个互补轴。
        for i in range(len(self.vecMode)):
            # 第 i 对时空平面使用的单一空间轴：依次为 z/y/x。
            vec_id = self.vecMode[i]
            # 对应空间平面的两个轴：依次是 xy/xz/yz。
            mat_id_0, mat_id_1 = self.matMode[i]

            # 追加第 i 张空间特征平面。
            plane_coef.append(
                # Parameter 将这张平面登记为可训练参数。
                torch.nn.Parameter(
                    # 从小幅度随机值加固定偏移开始训练。
                    self.init_scale
                    * torch.randn(
                        # [1,C_i,第一空间轴长度,第二空间轴长度]。
                        (1, n_component[i], gridSize[mat_id_0], gridSize[mat_id_1])
                    )
                    + self.init_shift
                )
            )
            # 追加与上述空间平面配对的空间-时间平面。
            line_time_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    # 形状 [1,C_i,互补空间轴长度,T]。
                    * torch.randn((1, n_component[i], gridSize[vec_id], self.time_grid))
                    + self.init_shift
                )
            )

        # 将两个列表包装成 ParameterList，并移到指定 CPU/GPU。
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(
            line_time_coef
        ).to(device)

    def get_optparam_groupsrho(self, cfg, lr_scale=1.0):
        """
        【主线】组装只训练密度/折射率分支的优化器参数组。

        包含密度时空平面、空间平面、基矩阵，以及可选的密度解码网络。
        """
        # 创建密度分支的优化参数字典列表。
        grad_vars = [
            {
                # tz/ty/tx 密度时空平面。
                "params": self.density_line_time,
                # 当前学习率等于配置值乘统一缩放。
                "lr": lr_scale * cfg.lr_density_grid,
                # 保留未缩放的原始学习率。
                "lr_org": cfg.lr_density_grid,
            },
            {
                # xy/xz/yz 密度空间平面。
                "params": self.density_plane,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                # 密度特征投影基矩阵的参数。
                "params": self.density_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_density_nn,
                "lr_org": cfg.lr_density_nn,
            },
        ]

        # plain 模式是无参数函数；只有 MLP/KAN 模式才需要加入优化器。
        if isinstance(self.density_regressor, torch.nn.Module):
            # 追加密度解码网络参数。
            grad_vars += [
                {
                    "params": self.density_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_density_nn,
                    "lr_org": cfg.lr_density_nn,
                }
            ]
        
        # 返回密度分支的完整参数组。
        return grad_vars
    
    def get_optparam_groupsgrad(self, cfg, lr_scale=1.0):
        """
        【主线】组装外观/梯度分支的优化器参数组。

        结构与 ``get_optparam_groupsrho`` 对称，但使用 ``lr_app_grid/lr_app_nn``。
        """
        # 收集外观时空平面、空间平面和投影层。
        grad_vars = [
            {
                # 外观的 tz/ty/tx 时空平面。
                "params": self.app_line_time,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                # 外观的 xy/xz/yz 空间平面。
                "params": self.app_plane,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                # 外观基矩阵参数。
                "params": self.app_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_app_nn,
                "lr_org": cfg.lr_app_nn,
            },
        ]

        # 解码器是真正神经网时才有参数可训练。
        if isinstance(self.app_regressor, torch.nn.Module):
            # 追加 app_regressor 的参数组。
            grad_vars += [
                {
                    "params": self.app_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_app_nn,
                    "lr_org": cfg.lr_app_nn,
                }
            ]
        # 返回外观/梯度分支参数。
        return grad_vars

    def get_optparam_groups(self, cfg, lr_scale=1.0):
        """
        【主线】组装主训练使用的密度、外观与畸变参数组。

        列表中每个字典的 ``params`` 是一组参数，``lr`` 是当前学习率，
        ``lr_org`` 保留原始学习率供调度器使用。
        """
        # 主优化器先收集四组平面参数和两个投影层。
        grad_vars = [
            {
                # 密度时空平面。
                "params": self.density_line_time,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                # 密度空间平面。
                "params": self.density_plane,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                # 外观时空平面。
                "params": self.app_line_time,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                # 外观空间平面。
                "params": self.app_plane,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                # 密度基矩阵。
                "params": self.density_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_density_nn,
                "lr_org": cfg.lr_density_nn,
            },
            {
                # 【注意】app 投影层的当前学习率被原代码额外除以 10。
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

        # 密度解码器是 MLP/KAN 时，将它的参数加入主优化器。
        if isinstance(self.density_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.density_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_density_nn,
                    "lr_org": cfg.lr_density_nn,
                }
            ]
        # 基类定义的畸变回归器是 MLP，通常也加入主优化器。
        if isinstance(self.distortion_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.distortion_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_density_nn,
                    "lr_org": cfg.lr_density_nn,
                }
            ]
        # 返回全部主训练参数组。
        return grad_vars  

    def get_optparam_groupsapp(self, cfg, lr_scale=1.0):
        """
        【主线】只收集 app_regressor 解码网络参数，用于单独微调。

        app_regressor 若是 plain 无参数函数，则返回空列表。
        """
        # 从空参数列表开始。
        grad_vars = []

        # 仅 nn.Module 解码器包含可训练 parameters()。
        if isinstance(self.app_regressor, torch.nn.Module):
            # 追加外观解码器参数及学习率。
            grad_vars += [
                {
                    "params": self.app_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_app_nn,
                    "lr_org": cfg.lr_app_nn,
                }
            ]

        # 返回单独微调所需参数组。
        return grad_vars 
    def compute_densityfeature(
        self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor
    ) -> torch.Tensor:
        """
        【主线】在六张密度特征平面上查询 N 个时空点，融合并投影特征。

        ``xyz_sampled``：``[N,3]`` 归一化坐标；``frame_time``：``[N,1]`` 归一化时间。
        返回：``[N,density_dim]`` 密度潜在特征，后续交给 density_regressor。

        【数学直觉】对每个 ``(x,y,z,t)``，同时查 ``xy`` 和 ``tz``、
        ``xz`` 和 ``ty``、``yz`` 和 ``tx``，才能组合出该四维点的信息。
        """
        # Prepare coordinates for grid sampling.
        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        # 【主线】构造三组空间平面坐标：[xy,xz,yz]。
        plane_coord = (
            torch.stack(
                (
                    # self.matMode[0]=[0,1]，取每个点的 (x,y)。
                    xyz_sampled[..., self.matMode[0]],
                    # self.matMode[1]=[0,2]，取 (x,z)。
                    xyz_sampled[..., self.matMode[1]],
                    # self.matMode[2]=[1,2]，取 (y,z)。
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            #.detach()
            # grid_sample 要求坐标形式：此处整理为 [3,N,1,2]。
            .view(3, -1, 1, 2)
        )
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        # 【主线】先取三个互补空间坐标：[z,y,x]，形状 [3,N]。
        line_time_coord = torch.stack(
            (
                # 与 xy 平面配对的 z。
                xyz_sampled[..., self.vecMode[0]],
                # 与 xz 平面配对的 y。
                xyz_sampled[..., self.vecMode[1]],
                # 与 yz 平面配对的 x。
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        # 把时间 t 与 z/y/x 分别配对，得到 [tz,ty,tx] 坐标。
        line_time_coord = (
            torch.stack(
                # frame_time 复制三份后与 [z,y,x] 沿最后一维配成二维坐标。
                (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            #.detach()
            # 整理为 grid_sample 所需的 [3,N,1,2]。
            .view(3, -1, 1, 2)
        )

        # 创建两个列表，分别收集空间平面与时空平面的查询特征。
        plane_feat, line_time_feat = [], []
        # Extract features from six feature planes.
        # 【主线】遍历三对密度特征平面。
        for idx_plane in range(len(self.density_plane)):
            # Spatial Plane Feature: Grid sampling on density plane[idx_plane] given coordinates plane_coord[idx_plane].
            #print(self.density_plane[idx_plane].shape)
            # 将当前空间平面的插值特征追加到 plane_feat。
            plane_feat.append(
                #F.grid_sample
                # grid_sample 在归一化坐标上做双线性插值。
                self.bilinear_layer(
                    # 输入平面形状 [1,C_i,H,W]。
                    self.density_plane[idx_plane],
                    # 只取第 idx_plane 组坐标，保留长度为 1 的首轴。
                    plane_coord[[idx_plane]],
                    # 边角对齐并选择双线性插值。
                    align_corners=True, mode='bilinear',
                # 将 grid_sample 输出整理为 [C_i,N]。
                ).view(-1, *xyz_sampled.shape[:1])
            )
            # Spatial-Temoral Feature: Grid sampling on density line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
            # 同样查询与当前空间平面配对的时空平面。
            line_time_feat.append(
                self.bilinear_layer(
                    # 时空平面形状 [1,C_i,S,T]。
                    self.density_line_time[idx_plane],
                    # 对应的 [tz]、[ty] 或 [tx] 坐标。
                    line_time_coord[[idx_plane]],
                    align_corners=True, mode='bilinear',
                # 同样整理为 [C_i,N]。
                ).view(-1, *xyz_sampled.shape[:1])
            )
        # 将三份空间特征和三份时空特征分别堆叠，通常得 [3,C,N]。
        plane_feat, line_time_feat = torch.stack(plane_feat, dim=0), torch.stack(
            line_time_feat, dim=0
        )
        #plane_feat, line_time_feat=torch.relu(plane_feat),torch.relu(line_time_feat)
        # Fusion One
        # 【主线】第一级融合：在每一对“空间 + 时空”平面之间操作。
        if self.fusion_one == "multiply":
            # 逐元素相乘：一个特征只有在两张配对平面上都强时才强。
            inter = plane_feat * line_time_feat
        elif self.fusion_one == "sum":
            # 逐元素相加。
            inter = plane_feat + line_time_feat
        elif self.fusion_one == "concat":
            # 保留两份特征，沿“平面组”轴拼接：3 组变 6 组。
            inter = torch.cat([plane_feat, line_time_feat], dim=0)
        else:
            # 未实现的融合名称立即报错。
            raise NotImplementedError("no such fusion type")

        # Fusion Two
        # 【主线】第二级融合：在三对（或拼接后六组）特征之间操作。
        if self.fusion_two == "multiply":
            # 沿第 0 轴连乘，得 [C,N]。
            inter = torch.prod(inter, dim=0)
        elif self.fusion_two == "sum":
            # 沿第 0 轴求和，得 [C,N]。
            inter = torch.sum(inter, dim=0)
        elif self.fusion_two == "concat":
            # 将前两个特征轴展平，保留最后的 N 样本轴。
            inter = inter.view(-1, inter.shape[-1])
        else:
            raise NotImplementedError("no such fusion type")

        # 转置为 [N,融合特征数]，再用基矩阵投影为 [N,density_dim]。
        inter = self.density_basis_mat(inter.T)  # Feature Projection

        # 返回交给 density_regressor 的密度特征。
        return inter

    def compute_appfeature(
        self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor
    ) -> torch.Tensor:
        """
        【主线】在六张外观特征平面上查询 N 个时空点，融合并投影。

        ``xyz_sampled`` 形状 ``[N,3]``，``frame_time`` 形状 ``[N,1]``。
        返回 ``[N,app_dim]``，后续可解码为 RGB、位移或其他物理量。

        【注意】算法与 ``compute_densityfeature`` 对称，但使用 app 平面和 app 基矩阵。
        """
        # Prepare coordinates for grid sampling.
        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        # 构造 [xy,xz,yz] 三组空间平面坐标，最终形状 [3,N,1,2]。
        plane_coord = (
            torch.stack(
                (
                    # (x,y)。
                    xyz_sampled[..., self.matMode[0]],
                    # (x,z)。
                    xyz_sampled[..., self.matMode[1]],
                    # (y,z)。
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            #.detach()
            .view(3, -1, 1, 2)
        )
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        # 取三个互补空间坐标 [z,y,x]。
        line_time_coord = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        # 把时间与 [z,y,x] 配成 [tz,ty,tx]，整理为 [3,N,1,2]。
        line_time_coord = (
            torch.stack(
                # 将 [N,1] 时间复制三份并去掉末尾单位轴。
                (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            #.detach()
            .view(3, -1, 1, 2)
        )

        # 创建列表收集两类平面的插值特征。
        plane_feat, line_time_feat = [], []
        # 【主线】遍历三对 app 特征平面。
        for idx_plane in range(len(self.app_plane)):
            # Spatial Plane Feature: Grid sampling on app plane[idx_plane] given coordinates plane_coord[idx_plane].
            # 查询第 idx_plane 张空间平面。
            plane_feat.append(
                self.bilinear_layer(
                    # 输入 app 空间平面 [1,C_i,H,W]。
                    self.app_plane[idx_plane],
                    # 输入对应 [xy]/[xz]/[yz] 坐标。
                    plane_coord[[idx_plane]],
                    #align_corners=self.align_corners, mode='bilinear',
                # 【注意】此调用未显式传 align_corners/mode，使用 grid_sample 默认值；保留原行为。
                ).view(-1, *xyz_sampled.shape[:1])
            )
            # Spatial-Temoral Feature: Grid sampling on app line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
            # 查询对应的 app 时空平面。
            line_time_feat.append(
                self.bilinear_layer(
                    # 输入 app 时空平面 [1,C_i,S,T]。
                    self.app_line_time[idx_plane],
                    # 输入对应 [tz]/[ty]/[tx] 坐标。
                    line_time_coord[[idx_plane]],
                    #align_corners=self.align_corners, mode='bilinear',
                # 整理为 [C_i,N]。
                ).view(-1, *xyz_sampled.shape[:1])
            )

        # 将三份空间特征和三份时空特征分别堆叠。
        plane_feat, line_time_feat = torch.stack(plane_feat), torch.stack(
            line_time_feat
        )
        #print(plane_feat.shape, line_time_feat.shape )
        # Fusion One
        # 【主线】第一级：在每对空间/时空特征之间融合。
        if self.fusion_one == "multiply":
            # 逐元素相乘。
            inter = plane_feat * line_time_feat
        elif self.fusion_one == "sum":
            # 逐元素相加。
            inter = plane_feat + line_time_feat
        elif self.fusion_one == "concat":
            # 沿第 0 轴拼接两类平面特征。
            inter = torch.cat([plane_feat, line_time_feat], dim=0)
        else:
            raise NotImplementedError("no such fusion type")

        # Fusion Two
        # 【主线】第二级：在三对/六组特征之间融合。
        if self.fusion_two == "multiply":
            # 沿第 0 轴求乘积。
            inter = torch.prod(inter, dim=0)
        elif self.fusion_two == "sum":
            # 沿第 0 轴求和。
            inter = torch.sum(inter, dim=0)
        elif self.fusion_two == "concat":
            # 展平所有特征组与通道，保留 N 轴。
            inter = inter.view(-1, inter.shape[-1])
        else:
            raise NotImplementedError("no such fusion type")

        # 转置为 [N,特征数] 并投影为 [N,app_dim]。
        inter = self.app_basis_mat(inter.T)  # Feature Projection

        # 返回交给 app_regressor 的外观/物理特征。
        return inter

    def TV_loss_density(self, reg, reg2=None):
        """
        【主线】累加密度空间平面和时空平面的 TV（总变分）正则损失。

        ``reg`` 作用于 xy/xz/yz 平面，``reg2`` 作用于 tz/ty/tx 平面；
        未传 ``reg2`` 时复用 ``reg``。TV 鼓励相邻体素特征平滑，抑制三维重建噪点。
        """
        # 用数值 0 初始化累加器。
        total = 0
        # 未传第二正则函数时，对两类平面使用同一函数。
        if reg2 is None:
            reg2 = reg
        # 遍历三对密度特征平面。
        for idx in range(len(self.density_plane)):
            # 对第 idx 张空间平面用 reg，对配对时空平面用 reg2，并累加。
            total = (
                total + reg(self.density_plane[idx]) + reg2(self.density_line_time[idx])
            )
        # 返回标量 TV 损失总和。
        return total

    def TV_loss_app(self, reg, reg2=None):
        """
        【主线】累加外观空间平面和时空平面的 TV 正则损失。

        参数含义与 ``TV_loss_density`` 相同，但操作 app 分支。
        """
        # 初始化累加器。
        total = 0
        # 未指定 reg2 时使它等于 reg。
        if reg2 is None:
            reg2 = reg
        # 遍历三对 app 特征平面。
        for idx in range(len(self.app_plane)):
            # 累加空间平面与配对时空平面的正则值。
            total = total + reg(self.app_plane[idx]) + reg2(self.app_line_time[idx])
        # 返回 app 分支 TV 损失。
        return total

    def L1_loss_density(self):
        """
        【主线】计算六张密度特征平面的 L1 正则。

        L1 是特征绝对值的平均，会鼓励更多参数靠近 0，得到稀疏紧凑的分解。
        """
        # 初始化累加器。
        total = 0
        # 遍历三对密度特征平面。
        for idx in range(len(self.density_plane)):
            # 对空间平面和时空平面分别取绝对值平均，然后加入总和。
            total = (
                total
                + torch.mean(torch.abs(self.density_plane[idx]))
                + torch.mean(torch.abs(self.density_line_time[idx]))
            )
        # 返回密度平面 L1 损失。
        return total

    def L1_loss_app(self):
        """
        【主线】计算六张外观特征平面的 L1 正则。

        【注意】原实现的循环上限写为 ``len(self.density_plane)``，而不是 app_plane；
        通常两者都是 3，中文注释版保留原代码。
        """
        # 初始化累加器。
        total = 0
        # 按原实现遍历 density_plane 的数量。
        for idx in range(len(self.density_plane)):
            # 累加第 idx 张 app 空间平面和时空平面的绝对值平均。
            total = (
                total
                + torch.mean(torch.abs(self.app_plane[idx]))
                + torch.mean(torch.abs(self.app_line_time[idx]))
            )
        # 返回 app 平面 L1 损失。
        return total

    @torch.no_grad()
    def up_sampling_planes(self, plane_coef, line_time_coef, res_target, time_grid):
        """
        【主线】将三对空间/时空特征平面双线性上采样到更高分辨率。

        ``plane_coef`` 是 xy/xz/yz，``line_time_coef`` 是 tz/ty/tx；
        ``res_target`` 是新 [X,Y,Z]，``time_grid`` 是新 T。返更新后的两个 ParameterList。
        """
        # 【暂时不用深究】no_grad 使上采样不连回旧参数的反向传播图。
        # 遍历三对特征平面。
        for i in range(len(self.vecMode)):
            # 读取当前时空平面的空间轴 z/y/x。
            vec_id = self.vecMode[i]
            # 读取当前空间平面的两个轴 xy/xz/yz。
            mat_id_0, mat_id_1 = self.matMode[i]
            # 将上采样结果重新包装为可训练 Parameter。
            plane_coef[i] = torch.nn.Parameter(
                # 对 [1,C,H,W] 空间特征平面做双线性插值。
                F.interpolate(
                    # .data 取当前参数值而不保留旧梯度图。
                    plane_coef[i].data,
                    # 【注意】F.interpolate 的 size 顺序是 (H,W)，原实现按 (mat_id_1,mat_id_0) 传入。
                    size=(res_target[mat_id_1], res_target[mat_id_0]),
                    mode="bilinear",
                    # 沿用模型配置的边角对齐方式。
                    align_corners=self.align_corners,
                )
            )
            # 同样将时空特征平面上采样并包装为 Parameter。
            line_time_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    line_time_coef[i].data,
                    # 目标形状的两个空间轴是 (互补空间分辨率, 时间分辨率)。
                    size=(res_target[vec_id], time_grid),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )

        # 返回上采样后的三张空间平面和三张时空平面。
        return plane_coef, line_time_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target, time_grid):
        """
        【主线】同时提高 app 与 density 全部特征平面的分辨率，然后更新射线步长。

        常用于“从粗到细”训练：先在小网格学低频结构，再上采样学三维场细节。
        """
        # 上采样 app 的三对特征平面。
        self.app_plane, self.app_line_time = self.up_sampling_planes(
            self.app_plane, self.app_line_time, res_target, time_grid
        )
        # 上采样 density 的三对特征平面。
        self.density_plane, self.density_line_time = self.up_sampling_planes(
            self.density_plane, self.density_line_time, res_target, time_grid
        )

        # 【主线】新网格体素尺寸变小，因此重新计算射线步长与默认采样数。
        self.update_stepSize(res_target)
        # 向终端打印新空间分辨率。
        print(f"upsamping to {res_target}")
