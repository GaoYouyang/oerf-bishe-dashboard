"""
TDBOST 的训练调度器：连接数据、可微渲染、损失函数和优化器。

零基础建议阅读顺序：
1. 【主线】SimpleSampler.nextids：如何抽出一批训练数据。
2. 【主线】Trainer.sample_data：一批数据包含什么。
3. 【主线】Trainer.train：BOS 观测怎样通过渲染器变成损失，再更新三维场。
4. 【进阶】Trainer.train_PDE / train_PDE_FVM：再加入流体物理方程和 PIV 数据。

本注释版保留原程序的变量名、函数签名、计算公式和执行顺序，只增加中文导读。
"""

# Python 标准库 math：这里主要用 cos 和 pi 生成余弦学习率。
import math
# sys：把 tqdm 进度条的输出明确写到标准输出。
import sys
# os：检查预处理数据文件是否已存在。
import os
# NumPy：读写数组、生成网格、检查 NaN（非数）。
import numpy as np
# PyTorch：张量计算、自动求导、GPU 加速和优化器。
import torch
# tqdm：在终端显示训练进度条。
from tqdm.auto import tqdm
# torchviz：可把 PyTorch 计算图绘出来；当前只在被注释的调试代码中出现。
from torchviz import make_dot
# ZeroMQ 库导出的 device 名称；【注意】后面 Trainer.__init__ 的 device 参数会遮蔽它。
from zmq import device
# 项目内部的快速三线性体渲染器：把三维场预测成 BOS 观测。
from render.render import OctreeRender_trilinear_fast as renderer
# 项目内部的评估函数：定期在测试集上可视化/计算指标。
from render.render import evaluation
# TVLoss（总变差损失）：鼓励相邻网格值平滑，减少噪声。
from render.util.Reg import TVLoss
# cal_n_samples 根据网格分辨率估计每条光线的采样点数；GM_Resi 当前未在本文件中执行。
from render.util.Sampling import GM_Resi, cal_n_samples
# N_to_reso：把目标体素总数转换为 x/y/z 三个方向的分辨率。
from render.util.util import N_to_reso
# 【注意】原文件重复 import torch，这不影响结果；为保留原执行顺序，此处不删除。
import torch
# PyTorch 数据工具：TensorDataset 组合输入/标签，DataLoader 分批，Subset 按索引取子集。
from torch.utils.data import TensorDataset, DataLoader, Subset
# loadmat：读取 MATLAB 的 .mat 实验数据文件。
from scipy.io import loadmat
# lhs（Latin Hypercube Sampling，拉丁超立方采样）：在时空区域中较均匀地选残差点。
from pyDOE import lhs
# autocast 和 GradScaler：混合精度训练，在 GPU 上节省显存并防止小梯度下溢。
from torch.amp import autocast, GradScaler
# RegularGridInterpolator：对规则三维网格上的折射率/密度值做线性插值。
from scipy.interpolate import RegularGridInterpolator
# SciPy 顶层包：后面通过 scipy.ndimage.generic_filter 填补 NaN。
import scipy


def ssim_my(x, y, c1=0.01, c2=0.03):
    """
    【评估工具】计算两个同形状张量的简化全局 SSIM（结构相似性）。

    输入：
        x, y: 形状相同的 PyTorch 张量，可以是 [D,H,W] 三维场。
        c1, c2: 防止分母过小的稳定常数，函数内会平方。
    输出：
        Python 浮点数，通常越接近 1 表示两个场越相似。
    注意：这里对整个张量求一组均值/方差，不是常见的滑动窗口 SSIM。
    """

    # 分别计算 x 和 y 所有元素的全局均值。
    mu_x = torch.mean(x)
    mu_y = torch.mean(y)
    # 分别计算 x 和 y 的全局标准差，衡量数值起伏。
    sigma_x = torch.std(x)
    sigma_y = torch.std(y)
    # 计算交叉协方差：衡量 x 和 y 是否同步变化。
    sigma_xy = torch.mean((x - mu_x) * (y - mu_y))

    # SSIM 公式使用 C1=(c1)^2 和 C2=(c2)^2。
    c1 = c1**2
    c2 = c2**2

    # 分子同时比较均值（亮度）和协方差（结构）。
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    # 分母用两个场各自的均值平方和方差做归一化。
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2)

    # 得到标量 SSIM 张量。
    ssim_value = numerator / denominator
    # item() 把只有一个值的 PyTorch 张量转为普通 Python 浮点数。
    return ssim_value.item()  # 
#np.save('/home/hyz/Project/Data_16t/hyz/AdjointNonlinearRayTracing/fuelrecon_64_KPlane.npy', nf1.detach().cpu().numpy())

# 【注意：导入即执行】下面这一段不在函数里：只要 import trainer.py，Python 就会立即读取师兄电脑上的 .mat 文件。
# 设定时间帧索引 k=9，下一行因而读取第 k+1=10 ms 的数据。
k=9#-1~~1
# f-string 中 {k+1:02} 表示两位数、10 毫秒数据；data 是一个按变量名取值的字典。
data = loadmat(f'/home/hyz/Project/Data/hyz/Sprayflame/jetflameinterpolated_data_{k+1:02}ms.mat')

# 从 MATLAB 字典中取出三维网格的 x/y/z 坐标，形状由原 .mat 文件决定。
xq = data['xq']
yq = data['yq']
zq = data['zq']
# 取出待重建/插值的三维标量场 Vq，后面将它当作折射率相关场。
Vq = data['Vq']
# 【暂时不用深究】计算忽略 NaN 后的最大值；返回值未赋给变量，主要是原作者交互调试留痕。
np.nanmax(Vq)


def fill_nan(values):
    """
    【数据清洗】为 scipy.ndimage.generic_filter 提供一个局部 NaN 填补规则。

    输入 values: 当前 6×6×6 立方邻域展平后的一维数组，长度 216。
    输出: 一个数；中心位置为 NaN 时返回邻域非 NaN 均值，否则保留中心值。
    数据意义：先修复空洞，才能进行后续三维插值和梯度计算。
    """
    # 【注意】原代码固定检查 values[108]，这是该 216 元素邻域中选定的中心位置。
    if np.isnan(values[108]):  # 检查中心点是否为 NaN
        # nanmean 在求均值时忽略 NaN，用邻域有效值填补中心空洞。
        return np.nanmean(values)  # 计算非 NaN 值的均值并返回
    # 中心值本来就有效时，不做平滑，直接原样返回。
    return values[108]  # 如果中心点不是 NaN，返回原始值


# 先统计并打印填补前的 NaN 总数，用于人工核对数据质量。
print("Original NaN positions:", np.isnan(Vq).sum())
# generic_filter 在 Vq 上滑动 6×6×6 窗口，每个位置调用上面的 fill_nan；mirror 表示边界外用镜像值补齐。
Vq = scipy.ndimage.generic_filter(Vq, fill_nan, size=(6, 6, 6), mode='mirror')
# 再打印填补后的 NaN 数，理想情况应为 0。
print("Filled NaN positions:", np.isnan(Vq).sum())
# PyTorch 的函数式接口；【注意】当前执行路径没有使用 F，但保留原 import。
import torch.nn.functional as F
# 把 NumPy 三维数组 Vq 复制成 PyTorch 张量 RI，默认为 float32。
RI=torch.Tensor(Vq)
# 分别在 [-2,2] 上创建 x/y/z 规则坐标轴；x 有 151 点，y/z 各 101 点。
x = torch.linspace(-2, 2, 151)
y = torch.linspace(-2, 2, 101)
z = torch.linspace(-2, 2, 101)
# SciPy 插值器使用 NumPy float64，因此将三根坐标轴从 PyTorch 转换为 NumPy 双精度数组。
x_points, y_points, z_points=x.numpy().astype(np.float64), y.numpy().astype(np.float64), z.numpy().astype(np.float64)
# 【主线】建立三维线性插值器；网格轴顺序是 (z,y,x)，数据也按这个顺序解读。
interpolator = RegularGridInterpolator(
    (z_points, y_points, x_points), Vq[:,:,:151], method='linear', bounds_error=False, fill_value=None)

# 打印 RI 中是否还有 NaN 或无穷大，是训练前的数值健康检查。
print(torch.isnan(RI).any(), torch.isinf(RI).any())


def refractive_index_field(X, Y, Z,t=0):
    """
    【物理数据接口】用上面建立的 SciPy 插值器查询任意坐标处的折射率场值。

    输入：
        X, Y, Z: 形状相同的 PyTorch 坐标张量，可以是 [N] 或更高维网格。
        t: 时间参数，为了与其他场函数保持调用格式一致；本函数内未实际使用。
    输出：
        与广播后 X/Y/Z 形状一致的 torch.float64 张量，位于 CPU。
    三维重建意义：把离散体素数据变成可在连续空间中查询的场函数。
    """
    # 【注意】原代码在这里交换 X/Y 的赋值顺序，以匹配数据文件和插值器的坐标约定。
    # cpu() 将数据移回 CPU，numpy() 转成 NumPy，astype 升为 float64。
    X, Y, Z=Y.cpu().numpy().astype(np.float64), X.cpu().numpy().astype(np.float64), Z.cpu().numpy().astype(np.float64)
    # stack(..., axis=-1) 在最后新增一维，把三个坐标合成 (...,3) 查询点。
    query_points = np.stack((X, Y, Z), axis=-1)  # 转换为 [batch_size, n, 3] 维度

    # 对 query_points 中的每个三维坐标做规则网格线性插值。
    interpolated_values = interpolator(query_points)

    # 把 SciPy 输出重新包装为 PyTorch float64 张量，便于后续光线追迹使用。
    return torch.tensor(interpolated_values, dtype=torch.float64)
# 【暂时不用深究】下面整块都以 # 开头，是读取 64³ RAW 数据的旧备选方案，当前不执行。
# # 设置文件路径
# file_path = '/home/hyz/Project/IRtrace/fuel_64x64x64_uint8.raw'

# # 使用 numpy 读取原始二进制文件
# # 'uint8' 代表无符号 8 位整数
# data = np.fromfile(file_path, dtype=np.uint8)

# # 重新调整数据形状为 (64, 64, 64)
# data = data.reshape((64, 64, 64))

# # 将 numpy 数组转换为 PyTorch 张量
# tensor_data = torch.from_numpy(data).float()
# tensor_data=(-tensor_data+tensor_data.max())/(tensor_data.max()-tensor_data.min())
# bd=2
# x = torch.linspace(-bd, bd, 64)
# y = torch.linspace(-bd, bd, 64)
# z = torch.linspace(-bd, bd, 64)
# x_points, y_points, z_points=x.numpy(), y.numpy(), z.numpy()
# tensor_data =tensor_data.numpy()
# # 检查读取的张量
# from scipy.interpolate import RegularGridInterpolator
# # 创建插值器，基于 x_points, y_points, z_points 和对应的 tensor_data 进行插值
# interpolator = RegularGridInterpolator(
#     (z_points, y_points, x_points), tensor_data, method='linear', bounds_error=False, fill_value=None)
# print(tensor_data.shape)  # 输出张量的形状 (64, 64, 64)
# print(tensor_data.dtype)  # 输出数据类型 torch.uint8
# print(tensor_data.max(),tensor_data.min(),tensor_data.mean())
# def refractive_index_field(X, Y, Z,t=0):
#         """
#         使用 SciPy 进行三维插值计算折射率场
#         Y, X, Z: 批量坐标
#         tensor_data: 三维张量数据 (D, H, W)
#         x_points, y_points, z_points: 数据网格坐标
#         """
#         Y, X, Z=Y.cpu().numpy(), X.cpu().numpy(), Z.cpu().numpy()
#         # 将数据转换为 numpy 数组，以便与 scipy 配合
 
#         # 批量插值 (Y, X, Z) 是批量坐标，形状为 [batch_size, n] 
#         query_points = np.stack((X, Y, Z), axis=-1)  # 转换为 [batch_size, n, 3] 维度
        
#         # 对每个查询点进行插值
#         interpolated_values = interpolator(query_points)#*2.56/100
        
#         # 返回插值后的结果
#         return torch.tensor(interpolated_values)
class SimpleSampler:
    """
    【主线】简单的随机索引采样器：每轮先打乱全部样本索引，再连续切出一个 batch。

    输入（初始化时）：
        total: 数据集总样本数 N。
        batch: 每次需要返回的索引数 B。
    输出（nextids）：
        [B] 的 torch.long 张量，可用来从数据集取出一批数据。
    训练意义：每次只用小批数据估计梯度，既节省显存，也引入有益的随机性。
    """

    def __init__(self, total, batch):
        """
        保存数据总量和批量大小，并把内部指针设成“需要首次打乱”的状态。

        输入 total/batch 是整数；无显式返回值。
        """
        # 保存数据集总数 N。
        self.total = total
        # 保存每批样本数 B。
        self.batch = batch
        # 把指针初始化为 total，使第一次 nextids() 进入重新打乱分支。
        self.curr = total
        # ids 将在第一次采样时变成 [N] 的随机排列；此刻先设为空。
        self.ids = None

    def nextids(self):
        """
        【主线】取出下一批随机但不重复的样本索引。

        无输入；返回 [batch] 的 LongTensor。
        当当前随机排列快用完时，重新打乱 0..total-1 并从头开始。
        """
        # 按一个 batch 向前移动指针；【注意】这是原代码的指针更新顺序。
        self.curr += self.batch
        # 如果再取一批将超出数据集，就开始新的随机轮次。
        if self.curr + self.batch > self.total:
            # np.random.permutation(total) 生成 0..N-1 的无重复随机排列，再转为 PyTorch 长整型索引。
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            # 新一轮从第 0 个索引开始。
            self.curr = 0
        # 用 Python 切片取出 [curr, curr+batch) 这 B 个索引。
        return self.ids[self.curr : self.curr + self.batch]


class Trainer:
    """
    TDBOST 的训练控制器。

    【主线】它自身不定义神经网络，而是保存已建好的 model、数据集和配置，然后组织：
    采样 -> 渲染/前向计算 -> 损失 -> 反向传播 -> 更新参数 -> 定期评估/升分辨率。
    """

    def __init__(
        self,
        model,
        modelcfg,
        optimcfg,
        cfgdata,
        cfgsystem,
        reso_cur,
        train_dataset,
        test_dataset,
        summary_writer,
        logfolder,
        device,
    ):
        """
        把训练需要的对象和配置保存为 Trainer 的属性。

        输入：
            model: 待优化的四维/三维场模型，输入通常是空间坐标 [N,3] 和时间 [N,1]。
            modelcfg: 模型配置，含网格分辨率、TV/L1 权重、升采样时刻等。
            optimcfg: 优化配置，含学习率、迭代数、batch size 和 Adam 参数。
            cfgdata: 数据配置，含采样方式、可视化数量等。
            cfgsystem: 系统/日志配置，含进度刷新和评估频率。
            reso_cur: 当前体素网格分辨率 [Rx,Ry,Rz]。
            train_dataset/test_dataset: 训练集和测试集。
            summary_writer: TensorBoard 日志写入器。
            logfolder: 图像和日志输出目录。
            device: torch.device 或等价设备标识，如 cuda:0/cpu。
        输出：无显式返回；只建立 self.* 属性。
        """
        # 保存待训练模型。
        self.model = model
        # 保存模型结构/网格相关配置。
        self.modelcfg = modelcfg
        # 保存优化器和迭代配置。
        self.optimcfg=optimcfg
        # 保存数据采样/可视化配置。
        self.cfgdata=cfgdata
        # 保存系统配置；【注意】属性名为 cfgsystems（复数），遵照原代码。
        self.cfgsystems=cfgsystem
        # 保存当前空间网格分辨率。
        self.reso_cur = reso_cur
        # 保存训练数据集。
        self.train_dataset = train_dataset
        # 保存测试/可视化数据集。
        self.test_dataset = test_dataset
        # 保存 TensorBoard 日志写入器。
        self.summary_writer = summary_writer
        # 保存图像和日志目录。
        self.logfolder = logfolder
        # 保存训练设备。
        self.device = device

    def get_lr_decay_factor(self, step):
        """
        【主线】计算当前学习率相对初始学习率的缩放因子。

        输入 step: 当前迭代次数（整数）。
        输出 lr_factor: 标量，后面使 current_lr = initial_lr * lr_factor。
        训练意义：训练初期大步学习，后期小步精修，通常更容易稳定收敛。
        """
        # -1 表示未单独指定衰减步数，因而用总迭代数 n_iters 作为衰减区间。
        if self.optimcfg.lr_decay_step == -1:
            # 【注意】这行会就地修改 optimcfg.lr_decay_step，保留原行为。
            self.optimcfg.lr_decay_step = self.optimcfg.n_iters

        # 指数衰减：比率的幂随 step 线性增加，使学习率按乘法平滑变小。
        if self.optimcfg.lr_decay_type == "exp":  # exponential decay
            # 在 step=0 时因子为 1，step=lr_decay_step 时为目标比率。
            lr_factor = self.optimcfg.lr_decay_target_ratio ** (
                step / self.optimcfg.lr_decay_step
            )
        # 线性衰减：从 1 匀速走向目标比率。
        elif self.optimcfg.lr_decay_type == "linear":  # linear decay
            # (1-step/decay_step) 表示“还剩多少衰减路程”。
            lr_factor = self.optimcfg.lr_decay_target_ratio + (
                1 - self.optimcfg.lr_decay_target_ratio
            ) * (1 - step / self.optimcfg.lr_decay_step)
        # 余弦衰减：开头和结尾变化慢，中间变化快。
        elif self.optimcfg.lr_decay_type == "cosine":  # consine decay
            # 0.5*(1+cos(π·progress)) 会从 1 平滑降到 0。
            lr_factor = self.optimcfg.lr_decay_target_ratio + (
                1 - self.optimcfg.lr_decay_target_ratio
            ) * 0.5 * (1 + math.cos(math.pi * step / self.optimcfg.lr_decay_step))

        # 返回缩放因子，实际学习率在训练循环中设置。
        return lr_factor

    def get_voxel_upsample_list(self):
        """
        【网格由粗到细】预先计算每次升采样后的空间体素总数和时间网格数。

        输入：无显式参数，从 modelcfg 读取 upsample_list、N_voxel_init/final、time_grid_init/final。
        输出：无显式返回；写入 self.N_voxel_list 和 self.Time_grid_list。
        训练意义：先在粗网格学大结构，再扩大网格学细节，可减少计算量并帮助收敛。
        """
        # upsample_list 存放“在第几次迭代做升采样”；其长度就是升采样次数。
        upsample_list = self.modelcfg.upsample_list
        # unaligned 模式下，需要预计算每一阶段的目标体素总数。
        if (
            self.modelcfg.upsampling_type == "unaligned"
        ):  # logaritmic upsampling. See explation of "unaligned" in model/__init__.py.
            # 在 log(N_init) 到 log(N_final) 之间等间隔取点，再 exp 回原尺度，得到几何级数式增长。
            N_voxel_list = (
                torch.round(
                    torch.exp(
                        torch.linspace(
                            np.log(self.modelcfg.N_voxel_init),
                            np.log(self.modelcfg.N_voxel_final),
                            len(upsample_list) + 1,
                        )
                    )
                ).long()
            # tolist()[1:] 转成 Python 整数列表并丢掉已在使用的初始网格值。
            ).tolist()[1:]
        # aligned 模式直接按各轴“2R-1”扩大，不需要预先存总体素数。
        elif (
            self.modelcfg.upsampling_type == "aligned"
        ):  # aligned upsampling doesn't need precompute N_voxel_list.
            # 用 None 明确表示后面不会 pop 这个列表。
            N_voxel_list = None
        # 时间网格无论空间是哪种模式，都用对数均匀/几何级数增长。
        Time_grid_list = (
            torch.round(
                torch.exp(
                    torch.linspace(
                        np.log(self.modelcfg.time_grid_init),
                        np.log(self.modelcfg.time_grid_final),
                        len(upsample_list) + 1,
                    )
                )
            ).long()
        # 转为 Python 列表，并丢掉初始时间网格值。
        ).tolist()[1:]
        # 保存空间升采样计划；训练时每触发一次就 pop(0) 取下一个值。
        self.N_voxel_list = N_voxel_list
        # 保存时间升采样计划。
        self.Time_grid_list = Time_grid_list

    def sample_data(self, train_dataset, iteration):
        """
        【主线】根据数据采样类型，从训练集取出一批 BOS 光线和标签。

        输入：
            train_dataset: 支持按索引取数据的训练集。
            iteration: 当前迭代数；原函数签名保留它，但函数体内未使用。
        输出：
            rays_train: 光线参数，批量形状由数据集定义。
            rgb_train: BOS 位移/观测标签，后续变量名又写作 deta_train，通常最后一维至少含 x/y 两分量。
            frame_time: 光线所属时间帧。
            frame_w2c: 世界到相机的位姿矩阵，通常末两维为 [4,4]。
            train_depth: 当前始终为 None，作为深度监督的预留接口。
        """
        # 当前路径不使用深度标签，先放一个空值保持统一返回格式。
        train_depth = None
        # rays 模式：把所有图像的光线打散后，每次随机取一批单独光线。
        if self.cfgdata.datasampler_type == "rays":
            # 从 SimpleSampler 得到 [batch_size] 的随机索引。
            ray_idx = self.sampler.nextids()
            # 用这批索引从数据集取出一个字典。
            data = train_dataset[ray_idx]
            # 按键名解包输入光线、观测标签、时间和相机位姿。
            rays_train, rgb_train, frame_time,frame_w2c = (
                # rays 的设备位置由数据集/渲染器协议决定，原代码没在此 .to(device)。
                data["rays"],
                # 标签会参与 GPU 损失计算，因此明确移到 self.device。
                data["rgbs"].to(self.device),
                # 每条光线对应的时间。
                data["time"],
                # 每条光线/图像对应的 world-to-camera 矩阵。
                data["w2c"]
            )
        # images 模式：每次取一整幅图像的全部光线。
        elif self.cfgdata.datasampler_type == "images":
            # 该模式的 sampler batch 为 1，因此返回一个图像索引。
            img_i = self.sampler.nextids()
            # 按图像索引取数据字典。
            data = train_dataset[img_i]
            # squeeze() 删掉 batch=1 引入的长度为 1 的维度，便于渲染器使用。
            rays_train, rgb_train, frame_time,frame_w2c  = (
                data["rays"].squeeze(),
                data["rgbs"].to(self.device).squeeze(),
                data["time"].squeeze(),
                data["w2c"].squeeze()
            )

        # 【注意】如果配置不是 rays/images，这些变量不会被赋值；这里保留原代码行为。
        return rays_train, rgb_train, frame_time,frame_w2c, train_depth
    
    # def sample_dataPINNs(self, train_dataset, iteration):
    #     """
    #     Sample a batch of data from the dataset.
    #     """
    #     ray_idx = self.sampler.nextids()
    #     data = train_dataset[ray_idx]
    #     Xs,Ys,Zs,Ts,RHOs = (
    #         data["Xs"],
    #         data["Ys"],
    #         data["Zs"],
    #         data["Ts"],
    #         data["RHOs"].to(self.device),
    #     )
    #     return Xs,Ys,Zs,Ts,RHOs#,Us,Vs,Ws,Ps,YO2s,YCH4s


    def init_sampler(self, train_dataset):
        """
        【主线】根据训练数据的采样类型，初始化随机采样器或层级采样所需统计量。

        输入 train_dataset: 训练数据集。
        输出: 无显式返回；设置 self.sampler 或 self.global_mean。
        """
        # 单光线模式每次抽 optimcfg.batch_size 条光线。
        if self.cfgdata.datasampler_type == "rays":
            # len(train_dataset) 是可采样光线总数。
            self.sampler = SimpleSampler(len(train_dataset), self.optimcfg.batch_size)
        # 整图模式每次只取 1 幅图。
        elif self.cfgdata.datasampler_type == "images":
            self.sampler = SimpleSampler(len(train_dataset), 1)
        # 层级采样模式保存训练集的全局 RGB/观测均值到训练设备。
        elif self.cfgdata.datasampler_type == "hierach":
            self.global_mean = train_dataset.global_mean_rgb.to(self.device)
    # def init_samplerPDE(self, train_dataset):
    #     """
    #     Initialize the sampler for the training dataset.
    #     """
    #     if self.cfgdata.datasampler_type == "rays":
    #         #定义采用索引
    #         print("totaldata:",train_dataset.lenPDE())
    #         self.sampler = SimpleSampler(train_dataset.lenPDE(), self.optimcfg.batch_size)
    #     elif self.cfgdata.datasampler_type == "images":
    #         self.sampler = SimpleSampler(train_dataset.lenPDE(), 1)
    #     elif self.cfgdata.datasampler_type == "hierach":
    #         self.global_mean = train_dataset.global_mean_rgb.to(self.device)

    def sampleBCres(self,model):
        """
        【进阶：边界约束】在四维时空区域中抽样，约束密度/折射率场靠近背景值且梯度不要过大。

        输入 model: 场模型，需提供 scene_bbox_min/max、rho0，并能输出 (n,dn)。
        输出 lossBCother: 标量张量，是梯度平方和背景值偏差平方的加权平均。
        形状：residual_points 为 [50000,4]，每行是 (x,y,z,t)；n 通常为 [50000,1]，dn 为相关梯度。
        """
        # 先建立 [x,y,z,t] 的下界列表，时间下界保持 0。
        lower_bounds=[0,0,0,0]
        # 用模型的三维场景包围盒最小值覆盖前三项。
        lower_bounds[:3]=model.scene_bbox_min
        # 先建立上界列表，时间上界为 1。
        upper_bounds=[1,1,1,1]
        # 用场景包围盒最大值覆盖 x/y/z 上界。
        upper_bounds[:3]=model.scene_bbox_max

        #print(lower_bounds,upper_bounds)
        # lhs(4, samples=50000) 在单位四维超立方 [0,1]^4 中较均匀地生成 50000 个点。
        sample = lhs(4, samples=50000)  # Generate normalized LHS samples in [0,1]^3
        # 创建与 sample 同形状的全 0 数组，用于存放缩放后的真实时空坐标。
        residual_points = np.zeros_like(sample)
        # 逐个维度将 [0,1] 线性映射到 [lower_bounds[i], upper_bounds[i]]。
        for i in range(4):
            # 线性映射公式：lower + unit_sample*(upper-lower)。
            residual_points[:, i] = lower_bounds[i] + sample[:, i] * (upper_bounds[i] - lower_bounds[i])
        # 将 NumPy [50000,4] 数组转成 float32 PyTorch 张量；原代码此处仍留在 CPU。
        residual_points=torch.from_numpy(residual_points).to(torch.float32)
        #mask=((residual_points[:,2]**2+(residual_points[:,1]-10)**2)>30**2)&(residual_points[:,0]>20)
        # 将前三列整理为 [50000,3]、时间列整理为 [50000,1]，并调用模型的非渲染路径。
        n, dn = model(residual_points[...,:3].reshape(-1, 3).to(self.device), residual_points[...,3].reshape(-1, 1).to(self.device), residual_points, is_rendear=False)
        # 计算本批预测 n 的全局均值，用于筛选较高的区域。
        nmean=torch.mean(n)
        # 【注意】用布尔条件筛选 n；由于是“或”，保留高于 rho0 或高于均值的元素。
        n=n[(n>model.rho0)|(n>nmean)]  
        #print(n)
        # 第一项惩罚场梯度 dn，第二项惩罚筛选后的 n 偏离背景值 rho0。
        lossBCother=1/100*torch.mean(dn**2)+1/1*torch.mean((n-model.rho0)**2)#torch.mean((n[n>nmean+0.1]-model.rho0)**2)
        # 返回单个标量损失，供总损失加权使用。
        return lossBCother

    def pde_res(self,model,modelU):
        """
        【进阶：PDE 残差】用自动求导构造密度连续性等物理约束损失。

        输入：
            model: 密度/折射率模型，返回 n 及 dn。
            modelU: 速度场模型，返回 UVW=[u,v,w]。
        输出：标量 PDE 损失张量。
        形状：residual_points=[50000,4]，UVW=[50000,3]，U/V/W_xyzt 和 rho_xyzt 通常为 [50000,4]。
        物理意义：让学到的密度场不仅拟合 BOS 数据，还尽量满足质量守恒/运输关系。
        """
        # 建立 [x,y,z,t] 下界；这里的空间边界是原实验手动设定值。
        lower_bounds=[0,0,0,0]
        # x∈[0,50]、y∈[-10,40]、z∈[-25,25]。
        lower_bounds[:3]=[0,-10,-25]#model.scene_bbox_min
        # 时间上界保持 1。
        upper_bounds=[1,1,1,1]
        # 覆盖前三项的空间上界。
        upper_bounds[:3]=[50,40,25]#model.scene_bbox_max

        #print(lower_bounds,upper_bounds)
        # 在 [0,1]^4 中拉丁超立方采样 50000 个点。
        sample = lhs(4, samples=50000)  # Generate normalized LHS samples in [0,1]^3
        # 创建实际坐标容器。
        residual_points = np.zeros_like(sample)
        # 将每一维从 [0,1] 映射到对应真实范围。
        for i in range(4):
            residual_points[:, i] = lower_bounds[i] + sample[:, i] * (upper_bounds[i] - lower_bounds[i])
        # 转为 GPU/CPU 上的 float32 张量，requires_grad_(True) 允许后面对 x/y/z/t 求导。
        residual_points=torch.from_numpy(residual_points).to(torch.float32).to(self.device).requires_grad_(True)
        # 前三列是 [N,3] 空间坐标，第四列是 [N,1] 时间；返回密度 n 和附加梯度 dn。
        n, dn = model(residual_points[...,:3].reshape(-1, 3), residual_points[...,3].reshape(-1, 1), residual_points, is_rendear=False)
        # 用第二个模型预测三维速度 UVW，通常形状 [N,3]。
        UVW = modelU(residual_points[...,:3].reshape(-1, 3), residual_points[...,3].reshape(-1, 1))
        # 对 u 相对 (x,y,z,t) 自动求导；create_graph=True 保留高阶求导能力。
        U_xyzt  = torch.autograd.grad(UVW[...,0], residual_points, grad_outputs=torch.ones_like(UVW[...,0]), create_graph = True, retain_graph = True, only_inputs=True)[0]
        # 对 v 相对 (x,y,z,t) 求一阶偏导。
        V_xyzt  = torch.autograd.grad(UVW[...,1], residual_points, grad_outputs=torch.ones_like(UVW[...,1]), create_graph = True, retain_graph = True, only_inputs=True)[0]
        # 对 w 相对 (x,y,z,t) 求一阶偏导。
        W_xyzt  = torch.autograd.grad(UVW[...,2], residual_points, grad_outputs=torch.ones_like(UVW[...,2]), create_graph = True, retain_graph = True, only_inputs=True)[0]
        # 对密度 n 相对 (x,y,z,t) 求梯度，各列依次是 ∂n/∂x,∂n/∂y,∂n/∂z,∂n/∂t。
        rho_xyzt  = torch.autograd.grad(n, residual_points, grad_outputs=torch.ones_like(n), create_graph = True, retain_graph = True, only_inputs=True)[0]#.detach()
        # dxrho_xyzt  = torch.autograd.grad(dn[...,0], residual_points, grad_outputs=torch.ones_like(dn[...,0]), create_graph = True, retain_graph = True, only_inputs=True)[0]#.detach()
        # dyrho_xyzt  = torch.autograd.grad(dn[...,1], residual_points, grad_outputs=torch.ones_like(dn[...,1]), create_graph = True, retain_graph = True, only_inputs=True)[0]#.detach()
        # dzrho_xyzt  = torch.autograd.grad(dn[...,2], residual_points, grad_outputs=torch.ones_like(dn[...,2]), create_graph = True, retain_graph = True, only_inputs=True)[0]#.detach()
        # 用密度范围和 y-z 平面上的圆形区域筛选需要额外约束的点。
        maskn=(n.squeeze()<0.5)&(n.squeeze()>0.1)&((residual_points[:,2]**2+(residual_points[:,1]-16)**2)<20**2)
        # 【主线】l1 是可压缩密度连续性形式：∂ρ/∂t + u·∇ρ + ρ∇·u，其中含原实验的时间尺度 1000/310。
        l1=rho_xyzt[...,3]*1000/310+rho_xyzt[...,0]*UVW[...,0]+rho_xyzt[...,1]*UVW[...,1]+rho_xyzt[...,2]*UVW[...,2]+n.squeeze()*(U_xyzt[...,0]+V_xyzt[...,1]+W_xyzt[...,2])
        # l2 是另一个运输/梯度约束；detach() 阻止梯度通过 rho_xyzt 分支回传，但仍可约束 UVW。
        l2=rho_xyzt[...,3].detach()*1000/310+rho_xyzt[...,0].detach()*UVW[...,0]+rho_xyzt[...,1].detach()*UVW[...,1]+rho_xyzt[...,2].detach()*UVW[...,2]+1200*torch.norm(rho_xyzt[...,:3].detach(),dim=-1)
        #print(l2.shape)
        # l2=dxrho_xyzt[...,3]+dxrho_xyzt[...,0]*UVW[...,0]+dxrho_xyzt[...,1]*UVW[...,1]+dxrho_xyzt[...,2]*UVW[...,2]
        # l3=dyrho_xyzt[...,3]+dyrho_xyzt[...,0]*UVW[...,0]+dyrho_xyzt[...,1]*UVW[...,1]+dyrho_xyzt[...,2]*UVW[...,2]
        # l4=dzrho_xyzt[...,3]+dzrho_xyzt[...,0]*UVW[...,0]+dzrho_xyzt[...,1]*UVW[...,1]+dzrho_xyzt[...,2]*UVW[...,2]
        #print(maskn.shape,l1.shape,l2.shape)
        # 只有当 maskn 选到至少一个点时，才对筛选后的 l2 求均值，避免空张量 mean 产生 NaN。
        if maskn.sum() > 0:
            # l1 先除以 100、l2 先除以 10 再平方，用于调整不同物理量的尺度。
            lossBCother=torch.mean((l1/100)**2)+torch.mean((l2[maskn]/10)**2)#+(l2/10)**2+(l3/10)**2+(l4/10)**2
        else:
            # 无有效筛选点时只使用连续性残差 l1。
            lossBCother=torch.mean((l1/100)**2)
        # 总 PDE 损失：连续性/运输残差 + 速度时空导数正则 + 速度幅值约束。
        return lossBCother/100+torch.mean((U_xyzt/100)**2+(V_xyzt/100)**2+(W_xyzt/100)**2)/100+torch.mean(((torch.sum((UVW[...,0]/1000)**2,dim=-1)-1.2**2))**2)
    def train(self,modelU=None):
        """
        【主线：BOS 反演训练】用可微渲染得到的 BOS 位移与真实观测的误差，优化时变三维场。

        输入：
            modelU: 可选速度场模型。None 表示主要做 BOS 数据监督；非 None 时原代码试图联合 PDE 约束。
        输出：
            无显式返回；会就地更新 self.model（及可选 modelU）的参数，并写 TensorBoard/可视化结果。
        核心形状：
            一批光线数记为 B；deta_train 通常为 [B,2]；渲染器返回每条光线的 x/y 位移预测及中间物理量。
        学习意义：
            BOS 相机只观测到折射产生的图像位移；通过可微渲染，位移误差能反向传回三维场参数。
        """
        # 释放 PyTorch 当前未使用的 CUDA 缓存，尽量为训练留出 GPU 显存。
        torch.cuda.empty_cache()
        # 把常用属性取成局部变量，后面代码更简洁。
        train_dataset = self.train_dataset
        test_dataset = self.test_dataset
        # 读取测试图像宽 W 和高 H；当前函数后续未直接使用这两个局部变量。
        W, H = test_dataset.img_wh
        # 待优化的密度/折射率场模型。
        model = self.model
        # 保存数据集是否提供深度监督。
        self.depth_data = test_dataset.depth_data
        # TensorBoard 日志写入器。
        summary_writer = self.summary_writer
        # 当前空间网格分辨率 [Rx,Ry,Rz]。
        reso_cur = self.reso_cur

        # ndc_ray 表示光线是否已映射到 NDC（归一化设备坐标）。
        ndc_ray = train_dataset.ndc_ray  # if the rays are in NDC
        # white_bg 表示渲染时是否使用白色背景。
        white_bg = test_dataset.white_bg  # if the background is white

        # 根据当前网格分辨率估计每条光线需要的采样点数。
        # min 确保不超过配置中的硬上限 modelcfg.nSamples。
        nSamples = min(
            self.modelcfg.nSamples,
            cal_n_samples(reso_cur, self.modelcfg.step_ratio),
        )

        # 只有在“单光线采样 + 非 NDC”时，才预先用场景包围盒过滤光线。
        if (self.cfgdata.datasampler_type == "rays") and (ndc_ray is False):
            # 从数据集中取出全部光线、标签、时间和世界到相机矩阵。
            allrays, allrgbs, alltimes ,allw2c= (
                train_dataset.all_rays,
                train_dataset.all_rgbs,
                train_dataset.all_times,
                train_dataset.all_W2C,
            )
            # 如果数据集有深度数据，就同步取出；否则用 None。
            if self.depth_data:
                alldepths = train_dataset.all_depths
            else:
                alldepths = None
            # 这个内层判断与外层条件重复，但保留原程序结构。
            if self.cfgdata.datasampler_type == "rays":
                # filtering_rays 删掉不与包围盒相交的光线；bbox_only=True 表示只做包围盒粗筛。
                allrays, allrgbs, alltimes, allw2c,alldepths = model.filtering_rays(
                    allrays, allrgbs, alltimes,allw2c, alldepths, bbox_only=True
                )
            # 用过滤后的各张量覆盖数据集内的全量数据。
            train_dataset.all_rays = allrays
            train_dataset.all_rgbs = allrgbs
            train_dataset.all_times = alltimes
            # 【注意】原数据属性读取时是 all_W2C，写回时是 all_w2c；本注释版不改动原名称。
            train_dataset.all_w2c=allw2c
            train_dataset.all_depths = alldepths

        # 根据 rays/images/hierach 模式创建采样器。
        self.init_sampler(train_dataset)
        # 预计算空间和时间网格由粗到细的升采样目标。
        self.get_voxel_upsample_list()

        # 创建纯空间平面上的 TV 正则器。
        tvreg_s = TVLoss()  # TV loss on the spatial planes
        # 创建时空平面的 TV 正则器，TV_t_s_ratio 调节时间与空间平滑强度。
        tvreg_s_t = TVLoss(
            1.0, self.modelcfg.TV_t_s_ratio
        )  # TV loss on the spatial-temporal planes     
        # 创建从 0 到 n_iters-1 的训练进度条；miniters 控制至少间隔多少步刷新。
        pbar = tqdm(
            range(self.optimcfg.n_iters),
            miniters=self.cfgsystems.progress_refresh_rate,
            file=sys.stdout,
        )

        # 创建优化器前再次尝试释放缓存显存。
        torch.cuda.empty_cache()
        # 默认路径：只训练主场模型，分别获取密度和 appearance/梯度参数组。
        if modelU is None:
            # 获取主密度分支的优化参数组，每组通常含 params、lr 和 lr_org。
            grad_varsrho = model.get_optparam_groups(self.optimcfg)
            # 获取 appearance/附加分支参数组。
            grad_varsgrad = model.get_optparam_groupsapp(self.optimcfg) 
            # 创建只含密度参数的 Adam；【注意】后面当前执行路径并未用它 step。
            optimizerrho = torch.optim.Adam(
                grad_varsrho, betas=(self.optimcfg.beta1, self.optimcfg.beta2)
            )
            # 创建同时更新 appearance 和密度参数的 Adam，它是默认路径实际 step 的优化器。
            optimizergapp= torch.optim.Adam(
                grad_varsgrad+grad_varsrho, betas=(self.optimcfg.beta1, self.optimcfg.beta2)
            )
        # 可选的联合速度模型路径。
        else:
            # 获取主场模型参数。
            grad_varsrho = model.get_optparam_groups(self.optimcfg)
            # 获取速度模型 modelU 参数。
            grad_varsU = modelU.get_optparam_groups(self.optimcfg) 
            # 将两个模型参数交给同一个 Adam 优化器。
            optimizerrho = torch.optim.Adam(
                grad_varsrho+grad_varsU, betas=(self.optimcfg.beta1, self.optimcfg.beta2)
            )
        # 【注意】原代码后面无条件调用 optimizergapp，但它只在 modelU is None 分支建立；这里只标注，不修改。
        # 创建混合精度的梯度缩放器，减少 float16/bfloat16 下的梯度下溢风险。
        scaler = GradScaler()   
        # 【主线】开始主训练循环：iteration 从 0 逐步增至 n_iters-1。
        for iteration in pbar:
            #lossBCother=self.sampleBCres(model)
            # 从数据集取一批光线、BOS 观测、时间和相机位姿。
            rays_train, deta_train, frame_time,frame_w2c, depth = self.sample_data(
                train_dataset, iteration
            )
            #print('trainer,framew2c',frame_w2c.shape)
            # 【注意】这是一个自赋值，不改变数值；右边的旧缩放公式已被注释。
            deta_train=deta_train#*2.2/320/2.48*1e4
            #optimizerrho.zero_grad()
            # 清空上一次迭代留在 optimizergapp 所管参数上的梯度。
            optimizergapp.zero_grad()
            # 开启自动混合精度上下文；原代码将 self.device 转为字符串传入。
            with autocast(str(self.device)):
                # 【主线】可微渲染：沿光线采样当前模型，预测 BOS 位移及相关中间量。
                detaxND, detayND,detax, detay,dxd1,dyd1,dzd1,rhobc=renderer(
                    # 本批光线几何参数。
                    rays_train,
                    # 本批光线对应时间。
                    frame_time,
                    # 本批的世界到相机矩阵。
                    frame_w2c,
                    # 要被查询的当前三维/四维场模型。
                    model,
                    # 一次送入渲染器的最大光线数，防止显存溢出。
                    chunk=self.optimcfg.batch_size,
                    # 每条光线沿线的采样点数。
                    N_samples=nSamples,#+int(torch.randint(-10, 11, (1,)))
                    # 是否使用白色背景。
                    white_bg=white_bg,
                    # 光线是否处在 NDC 坐标中。
                    ndc_ray=ndc_ray,
                    # 张量计算所在设备。
                    device=self.device,
                    # 明确通知渲染器当前在训练，需要保留可微路径/训练特性。
                    is_train=True,
                )
                #print('detax:',detax.shape)
                # 【输出导读】detaxND/detayND 是用于主监督的 x/y 位移；detax/detay 是其他尺度/路径的位移；dxd1/dyd1/dzd1 是中间导数；rhobc 是密度边界相关量。
                # 下面以 # 开头的几种 loss 是原作者留下的备选实验，不执行。
                # lossL1=(torch.norm(detax, p=1)+torch.norm(detay, p=2))*0.0001
                # summary_writer.add_scalar(
                #         "train/loss_L1",
                #         lossL1.detach().item(),
                #         global_step=iteration,
                #     )
                #lossgrad= torch.mean((dxd1 - deta_train[...,0]) ** 2)+torch.mean((dyd1- deta_train[...,1]) ** 2)
                # 【主线：数据项】对 x 位移和 y 位移分别计算 MSE，y 方向的误差权重为 2。
                loss0 =torch.mean((detaxND/1 - deta_train[...,0]/1) ** 2)+2*torch.mean((detayND/1 - deta_train[...,1]/1) ** 2)#+ torch.mean((detax - deta_train[...,0]) ** 2)+torch.mean((detay - deta_train[...,1]) ** 2)+
                #loss0=loss0+torch.mean(detax**2)
                #print(loss,torch.mean((dxd1**2)),torch.mean((dyd1**2)),torch.mean(rhobc))
                # 在 BOS 数据 MSE 上加入很小的 rhobc 平均值惩罚，权重为 0.01。
                loss = loss0+torch.mean(rhobc)*0.01#+1/100*torch.mean(rhoTV)#+1/100*torch.mean(dxyzd1**2)
                #print( loss0,torch.mean((dyd1**2)),torch.mean((dxd1**2)))
                # total_loss1 用来累加密度分支正则项，先用 Python 数字 0 初始化。
                total_loss1=0
                # total_loss2 用来累加 appearance 分支正则项。
                total_loss2=0
                # 计算本次迭代的学习率/正则权重衰减因子。
                lr_factor = self.get_lr_decay_factor(iteration)
                # 额外惩罚 dxd1/dyd1 的平方幅值，权重 0.1；后面与 total_lossrho 合并反传。
                lossapp=0.1*(torch.mean((dxd1**2))+torch.mean((dyd1**2)))
                # 【正则化】只有配置权重大于 0 且迭代超过 1000 步时，才开启密度平面 TV 平滑。
                if (self.modelcfg.TV_weight_density > 0)&(iteration>1000):
                    # 让 TV 权重与学习率同步衰减。
                    TV_weight_density = lr_factor * self.modelcfg.TV_weight_density
                    # 调用模型自身的密度 TV 计算，再乘当前权重。
                    loss_tv = model.TV_loss_density(tvreg_s, tvreg_s_t) * TV_weight_density
                    # 把 TV 损失累加到密度分支正则和。
                    total_loss1 = total_loss1 + loss_tv
                    #print(loss_tv)
                    # detach().item() 只把数值写入 TensorBoard，不把日志操作连入反向传播图。
                    summary_writer.add_scalar(
                        "train/reg_tv_density",
                        loss_tv.detach().item(),
                        global_step=iteration,
                    )

                # appearance 平面的 TV 正则：只要配置权重大于 0 就启用。
                if self.modelcfg.TV_weight_app > 0:
                    # 同样使正则权重随训练进程衰减。
                    TV_weight_app = lr_factor * self.modelcfg.TV_weight_app
                    # 计算 appearance 特征平面的 TV 损失。
                    loss_tv = model.TV_loss_app(tvreg_s, tvreg_s_t) * TV_weight_app
                    # 累加到 appearance 分支正则和。
                    total_loss2 = total_loss2 + loss_tv
                    # 记录 appearance TV 数值到 TensorBoard。
                    summary_writer.add_scalar(
                        "train/reg_tv_app", loss_tv.detach().item(), global_step=iteration
                    )

                # 密度平面的 L1 正则：鼓励大量参数靠近 0，得到更稀疏的表示。
                if self.modelcfg.L1_weight_density > 0:
                    # 当前 L1 权重 = 初始配置权重 × 衰减因子。
                    L1_weight_density = lr_factor * self.modelcfg.L1_weight_density
                    # 调用模型内密度分支的 L1 损失并加权。
                    loss_l1 = model.L1_loss_density() * L1_weight_density
                    # 累加到 total_loss1。
                    total_loss1 = total_loss1 + loss_l1
                    #print(loss_l1)
                    # 记录密度 L1 损失的标量数值。
                    summary_writer.add_scalar(
                        "train/reg_l1_density",
                        loss_l1.detach().item(),
                        global_step=iteration,
                    )

                # appearance 分支的 L1 正则。
                if self.modelcfg.L1_weight_app > 0:
                    # 计算衰减后的 appearance L1 权重。
                    L1_weight_app = lr_factor * self.modelcfg.L1_weight_app
                    # 对 appearance 参数计算加权 L1 损失。
                    loss_l1 = model.L1_loss_app() * L1_weight_app
                    # 累加到 total_loss2。
                    total_loss2 = total_loss2 + loss_l1
                    # 记录 appearance L1 损失。
                    summary_writer.add_scalar(
                        "train/reg_l1_app", loss_l1.detach().item(), global_step=iteration
                    )
                
                # # Loss on the rendered and gt depth maps.
                # if self.modelcfg.depth_loss and self.modelcfg.depth_loss_weight > 0:
                #     depth_loss = (depth_map.unsqueeze(-1) - depth) ** 2
                #     mask = depth != 0
                #     depth_loss = (
                #         torch.mean(depth_loss[mask]) * self.modelcfg.depth_loss_weight
                #     )
                #     total_loss += depth_loss
                #     summary_writer.add_scalar(
                #         "train/depth_loss",
                #         depth_loss.detach().item(),
                #         global_step=iteration,
                #     )
                # 默认无速度模型时：密度主损失 = 密度正则 + BOS/边界数据项。
                if modelU is None:
                    total_lossrho=total_loss1+loss#+lossBCother
                # 联合 modelU 时：原代码试图再加边界损失和 PDE 残差。
                else:
                    # 用密度模型和速度模型计算 PDE 损失。
                    lossPDE=self.pde_res(model,modelU)
                    # 【注意】lossBCother 的赋值在循环开头被注释；这是原实验分支的现状，本注释版不修复。
                    total_lossrho=total_loss1+loss+lossBCother+lossPDE
                #total_lossgrad=total_loss2+lossgrad

                # 将本次总密度损失写入 TensorBoard；add_scalar 可接收标量张量。
                summary_writer.add_scalar("train/mse", total_lossrho, global_step=iteration)

            # 下面注释掉的几行是旧的普通精度反传/优化路径，当前使用 GradScaler 的混合精度路径。
            # optimizergrad.zero_grad()
            # total_lossgrad.backward(retain_graph=True)
            # optimizergrad.step()
            # total_lossrho.backward()
            # optimizerrho.step()
            # 下面两行是另一种分开 step optimizerrho 的混合精度尝试，当前不执行。
            # scaler.scale(total_lossrho).backward(retain_graph=True)
            # # 使用缩放器来更新权重
            # scaler.step(optimizerrho)
            # 【主线：反向传播】先用 scaler 放大损失，再对 lossapp + total_lossrho 计算所有参数梯度。
            scaler.scale(lossapp+total_lossrho).backward()
            # 在梯度数值有效时调用 optimizergapp.step()，用 Adam 更新 appearance+密度参数。
            scaler.step(optimizergapp)
            # 根据本次是否发生上溢，动态调整下次的梯度缩放倍数。
            scaler.update()

            # detach() 从计算图中分离，item() 转成 Python 数字，供进度条显示。
            loss = total_lossrho.detach().item()
            # 【暂时不用深究】下面被注释的大段用于每 500 步在 64³ 网格上与真值比较相对 L2/SSIM，当前不执行。
            # if iteration%500==0:
            #     with torch.no_grad():
            #         from scipy.interpolate import RegularGridInterpolator
            #         bd=2
            #         x = torch.linspace(-bd, bd, 64)
            #         y = torch.linspace(-bd, bd, 64)
            #         z = torch.linspace(-bd, bd, 64)
            #         X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
            #         xyz=torch.stack((X,Y,Z),-1).to(self.device)
            #         k=9
            #         ngt=refractive_index_field(xyz[...,0], xyz[...,1], xyz[...,2],t=0)
            #         #print(ngt.shape,ngt.max(),ngt.min(),ngt.shape)
            #         t=torch.ones_like(X).to(self.device)*k
            #         with torch.no_grad():
            #             n,_=model(xyz.reshape(-1,3),t.reshape(-1,1),t,is_rendear=False)
            #         nf=n.reshape(64,64,64).to('cpu')
            #         nf1=(nf-nf.mean())+ngt.mean()#
            #         print(torch.norm((ngt-nf1))/torch.norm((ngt)),ssim_my(ngt, nf1),'lossdeta:',lossapp.item())#
            # graph = make_dot(loss, params=dict([('input', rays_train)]))
            #     # Save the graph to a file
            # graph.render(f"computation_graph2", format='png', cleanup=True) 
            # 每到设定的 progress_refresh_rate 步数倍，更新进度条文本。
            if iteration % self.cfgsystems.progress_refresh_rate == 0:
                # 格式化显示 5 位补零的迭代号和 6 位小数的损失。
                pbar.set_description(
                    f"Iteration {iteration:05d}:"
                    + f" mse = {loss:.6f}"
                )

            # 遍历 optimizerrho 中的每个参数组，更新当前学习率。
            # 【注意】默认路径实际 step 的是 optimizergapp，原代码却在此更新 optimizerrho；只标注不修改。
            for param_group in optimizerrho.param_groups:
                # lr_org 是模型参数组保存的初始学习率，乘衰减因子得到本步 lr。
                param_group["lr"] = param_group["lr_org"] * lr_factor
            # for param_group in optimizergrad.param_groups:
            #     param_group["lr"] = param_group["lr_org"] * lr_factor

            # 在每个 vis_every 周期的最后一步做测试集评估，且 N_vis 不能为 0。
            if (
                iteration % self.cfgsystems.vis_every == self.cfgsystems.vis_every - 1
                and self.cfgdata.N_vis != 0
            ):
                # 评估只做前向计算，禁止自动求导以节省显存和时间。
                with torch.no_grad():
                    # evaluation 在测试数据上渲染，把图像写入 imgs_vis，并返回 PSNR 列表/指标。
                    PSNRs_test = evaluation(
                        # 测试数据集。
                        test_dataset,
                        # 当前训练中的模型。
                        model,
                        # 数据/可视化配置。
                        self.cfgdata,
                        # 可视化图像输出目录。
                        f"{self.logfolder}/imgs_vis/",
                        # 用当前迭代号作为输出文件名前缀。
                        prefix=f"{iteration:06d}_",
                        # 背景、光线采样数和坐标模式与训练保持一致。
                        white_bg=white_bg,
                        N_samples=nSamples,
                        ndc_ray=ndc_ray,
                        # 评估计算所在设备。
                        device=self.device,
                        # 只计算基础指标，不额外计算更昂贵的扩展指标。
                        compute_extra_metrics=False,
                    )

                # 等待前面排队的 CUDA 操作全部完成，再继续下一步。
                torch.cuda.synchronize()

            # 如果当前迭代数在空体素掩码更新列表中，就准备更新分辨率。
            if iteration in self.modelcfg.update_emptymask_list:
                # 只在网格总体素数小于 256³ 时使用当前分辨率作为掩码分辨率。
                if (
                    reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3
                ):  # update volume resolution
                    # 保存准备给 EmptyMask 的 [Rx,Ry,Rz]。
                    reso_mask = reso_cur
                # 【注意】真正的 updateEmptyMask 调用在原代码中被注释，所以当前这个分支只会可能赋值 reso_mask。
                #model.updateEmptyMask(tuple(reso_mask))

            # 【主线：由粗到细】当迭代数命中 upsample_list 时，提升时空网格分辨率。
            if iteration in self.modelcfg.upsample_list:
                # aligned 方式对每根轴使用 R_new=2*R_old-1，旧网格点与新网格点对齐。
                if self.modelcfg.upsampling_type == "aligned":
                    # 列表推导式依次更新 x/y/z 三个分辨率。
                    reso_cur = [reso_cur[i] * 2 - 1 for i in range(len(reso_cur))]
                # unaligned 方式根据预计算的总体素数反推三轴分辨率。
                else:
                    # pop(0) 取出并删除下一个目标体素总数。
                    N_voxel = self.N_voxel_list.pop(0)
                    # 结合模型包围盒 aabb 和是否允许非立方体素，计算 [Rx,Ry,Rz]。
                    reso_cur = N_to_reso(
                        N_voxel, model.aabb, self.modelcfg.nonsquare_voxel
                    )
                # 同样取出下一阶段的时间网格数。
                time_grid = self.Time_grid_list.pop(0)
                # 网格变细后，重新估计每条光线的采样数，仍不超过配置上限。
                nSamples = min(
                    self.modelcfg.nSamples,
                    cal_n_samples(reso_cur, self.modelcfg.step_ratio),
                )
                # 调用模型内部插值/扩容逻辑，把空间网格和时间网格升到新分辨率。
                model.upsample_volume_grid(reso_cur, time_grid)
                # 升采样会创建新的网格参数，所以需要重新向模型查询可优化密度参数组。
                grad_varsrho = model.get_optparam_groupsrho(self.optimcfg, 1.0)
                # 同理，重新获取梯度/appearance 参数组；当前下面的优化器并未将它加入。
                grad_varsgrad = model.get_optparam_groupsgrad(self.optimcfg, 1.0)
                # 为新网格参数重建 Adam 优化器，旧优化器中的动量状态不会继承。
                optimizerrho = torch.optim.Adam(
                    grad_varsrho, betas=(self.optimcfg.beta1, self.optimcfg.beta2)
                )
                # 原代码保留了为 grad_varsgrad 建立独立 optimizergrad 的尝试，当前被注释。
                # optimizergrad = torch.optim.Adam(
                #     grad_varsgrad, betas=(self.optimcfg.beta1, self.optimcfg.beta2)
                # )

    def generate_4d_grid(self,db, size, time_start, time_end, time_size):
        """
        【PDE 采样工具】生成一组 (x,y,z,t) 四维网格点并随机打乱顺序。

        输入：
            db: x 轴两个采样面的半间距，x 实际只取 -db 和 db。
            size: y 和 z 每根轴的点数。
            time_start/time_end: 时间区间下界/上界。
            time_size: 时间采样点数。
        输出 grid_shuffled: [2*size*size*time_size, 4]，每行是 (x,y,z,t)。
        物理意义：在两个 x 截面上构造时空点，可用于 PIV/边界或 PDE 残差评估。
        """
        # 【注意】原注释说“size 个点”，但原代码实际只生成 x=[-db,db] 两个点。
        x = torch.Tensor([-db, db])
        # y 轴在 [-2.5,2.5] 中等间隔取 size 个点。
        y = torch.linspace(-2.5, 2.5, size)
        # z 轴同样在 [-2.5,2.5] 中取 size 个点。
        z = torch.linspace(-2.5, 2.5, size)
        # 在 time_start 到 time_end 之间等间隔生成 time_size 个时间点。
        t = torch.linspace(time_start, time_end, time_size)

        # meshgrid 穷举 x/y/z/t 的所有组合；ij 表示维度顺序与输入轴顺序一致。
        grid_x, grid_y, grid_z, grid_t = torch.meshgrid(x, y, z, t, indexing='ij')

        # 在最后一维把四个坐标叠成 (...,4)，再展平为二维点表。
        grid = torch.stack([grid_x, grid_y, grid_z, grid_t], dim=-1).reshape(-1, 4)
        # randperm 生成 0..M-1 的随机排列，M 是网格点总数。
        shuffled_indices = torch.randperm(grid.size(0))
        # 按随机索引重排所有四维点，数值集合不变，只改顺序。
        grid_shuffled = grid[shuffled_indices]
        # 返回打乱后的四维点表。
        return grid_shuffled

    def prepare_data(self,processed_file='processed_PINNs_dataLES.pt'):
        """
        【PDE/PIV 数据预处理】从 DNS 二进制数据或 LES MATLAB 数据中组装 (x,y,z,t)->(ρ,u,v,w,p) 数据集。

        输入 processed_file: 缓存的 PyTorch 数据文件路径。
        输出 dataset: TensorDataset，单个样本含空间 [3]、时间 [1]、ρ/u/v/w/p 五个标量。
        数据意义：为后面的密度拟合、PIV 速度监督和 PDE 残差提供场真值。
        注意：原代码在函数内固定 dataset='LES'，因此默认实际只走 LES 分支。
        """
        # 硬编码选择 LES 数据源；下面 DNS 分支保留作为备选实验路径。
        dataset='LES'
        # DNS（直接数值模拟）原始二进制文件读取分支。
        if dataset=='DNS':
            # 保存缓存文件路径。
            self.processed_file = processed_file
            # 师兄机器上 DNS 网格坐标文件目录。
            self.gridPath='/home/hyz/Project/Data/hyz/CH4O2_HIT_DNS/grid'
            # 师兄机器上 DNS 物理场文件目录。
            self.dataPath='/home/hyz/Project/Data/hyz/CH4O2_HIT_DNS/data'
            # 如果已有预处理缓存，就不再重新读取大量原始文件。
            if os.path.exists(self.processed_file):
                # 打印正在加载的缓存路径。
                print(f"Loading processed data from {self.processed_file}")
                # torch.load 反序列化之前保存的 Dataset/Subset 对象。
                dataset = torch.load(self.processed_file)
                # 命中缓存时提前结束函数。
                return dataset#DataLoader(dataset, batch_size=self.optimcfg.batch_size, shuffle=True)

            # 为 20 个时间帧的坐标、时间和五个物理量建立空列表。
            Xs, Ys, Zs, Times, RHOs, Us, Vs, Ws, Ps = [], [], [], [], [], [], [], [], []

            # 依次处理 index=0..19 共 20 帧 DNS 数据。
            for index in range(20):
                # 以小端 float32（<f4）格式读取 x/y/z 网格坐标展平数组。
                X = np.fromfile(f'{self.gridPath}/X_m.dat', dtype='<f4')
                Y = np.fromfile(f'{self.gridPath}/Y_m.dat', dtype='<f4')
                Z = np.fromfile(f'{self.gridPath}/Z_m.dat', dtype='<f4')

                # unique 取出每根轴不重复的规则网格点。
                x1d, y1d, z1d = np.unique(X), np.unique(Y), np.unique(Z)
                # 重建三维坐标网格，形状为 [Nx,Ny,Nz]。
                X, Y, Z = np.meshgrid(x1d, y1d, z1d, indexing='ij')

                # 读取本帧密度 ρ，并按网格形状重排。
                RHO = np.fromfile(f'{self.dataPath}/RHO_kgm-3_id0{index:02}.dat', dtype='<f4').reshape(X.shape)
                # 读取 x/y/z 三个速度分量。
                U = np.fromfile(f'{self.dataPath}/UX_ms-1_id0{index:02}.dat', dtype='<f4').reshape(X.shape)
                V = np.fromfile(f'{self.dataPath}/UY_ms-1_id0{index:02}.dat', dtype='<f4').reshape(X.shape)
                W = np.fromfile(f'{self.dataPath}/UZ_ms-1_id0{index:02}.dat', dtype='<f4').reshape(X.shape)
                # 读取压力 P。
                P = np.fromfile(f'{self.dataPath}/P_Pa_id0{index:02}.dat', dtype='<f4').reshape(X.shape)

                # 为本帧所有网格点生成相同时间，相邻帧间隔为 0.2462930526。
                Time = torch.ones_like(torch.Tensor(X)) * index *0.2462930526
                # 将坐标展平为一维并乘 10000 做单位/数值尺度转换，再追加到跨时间列表。
                Xs.append(torch.Tensor(X.ravel()) * 10000)
                Ys.append(torch.Tensor(Y.ravel()) * 10000)
                Zs.append(torch.Tensor(Z.ravel()) * 10000)
                # 时间和各物理场也展平为每网格点一个样本。
                Times.append(Time.reshape(-1))
                RHOs.append(torch.Tensor(RHO).reshape(-1))
                Us.append(torch.Tensor(U).reshape(-1))
                Vs.append(torch.Tensor(V).reshape(-1))
                Ws.append(torch.Tensor(W).reshape(-1))
                Ps.append(torch.Tensor(P).reshape(-1))
        # LES（大涡模拟）MATLAB 数据读取分支；由于函数开头硬编码 LES，默认会进入这里。
        elif dataset=='LES' :
            # 保存预处理缓存文件路径。
            self.processed_file = processed_file
            # 优先使用已存在的缓存，避免再扫描 17 个大型 .mat 文件。
            if os.path.exists(self.processed_file):
                # 输出实际加载路径。
                print(f"Loading processed data from {self.processed_file}")
                # 加载之前序列化的 Dataset/Subset。
                dataset = torch.load(self.processed_file)
                # 命中缓存时直接返回。
                return dataset#DataLoader(dataset, batch_size=self.optimcfg.batch_size, shuffle=True)

            # 为全部时间帧建立累积列表。
            Xs, Ys, Zs, Times, RHOs, Us, Vs, Ws, Ps = [], [], [], [], [], [], [], [], []

            # range(4,21) 依次处理 04ms..20ms，共 17 帧。
            for idx in range(4,21):
                # 从师兄机器上的绝对路径读取当前毫秒帧的 MATLAB 数据。
                data = loadmat(f'/home/hyz/Project/Data/hyz/Sprayflame/jetflameinterpolated_data_{idx:02}ms.mat')

                # 取出三维坐标网格。
                xq = data['xq']
                yq = data['yq']
                zq = data['zq']
                # Vq 是密度/折射率相关标量场。
                Vq = data['Vq']
                # Vuq/Vvq/Vwq 分别是 u/v/w 三个速度分量场。
                Vuq = data['Vuq']
                Vvq = data['Vvq']
                Vwq = data['Vwq']
                # 导入 SciPy 的 N 维图像/体数据滤波模块；导入放在循环内是原代码顺序。
                import scipy.ndimage

                def fill_nan(values):
                    """
                    对当前 6×6×6 局部窗口做 NaN 填补；输入长 216，输出一个标量。

                    中心 values[108] 为 NaN 时用邻域非 NaN 均值，否则保留原值。
                    """
                    # 检查窗口中心值是否缺失。
                    if np.isnan(values[108]):  # 检查中心点是否为 NaN
                        # 忽略 NaN 计算局部均值用作填补。
                        return np.nanmean(values)  # 计算非 NaN 值的均值并返回
                    # 中心值有效时原样返回。
                    return values[108]  # 如果中心点不是 NaN，返回原始值
                # 打印密度场填补前的 NaN 数量。
                print("Original NaN positions:", np.isnan(Vq).sum())
                # 对密度和三个速度场分别滑动 6×6×6 窗口填补缺失值，边界用镜像拓展。
                Vq = scipy.ndimage.generic_filter(Vq, fill_nan, size=(6, 6, 6), mode='mirror')
                Vuq = scipy.ndimage.generic_filter(Vuq, fill_nan, size=(6, 6, 6), mode='mirror')
                Vvq = scipy.ndimage.generic_filter(Vvq, fill_nan, size=(6, 6, 6), mode='mirror')
                Vwq = scipy.ndimage.generic_filter(Vwq, fill_nan, size=(6, 6, 6), mode='mirror')
                # 打印填补后密度场的 NaN 数，检查预处理是否成功。
                print("Filled NaN positions:", np.isnan(Vq).sum())
                # 导入 PyTorch 函数式接口；当前这个 F 名称未在分支内使用，保留原 import。
                import torch.nn.functional as F
                # 将四个 NumPy 三维场转成 PyTorch float32 张量。
                RHO=torch.Tensor(Vq)
                U=torch.Tensor(Vuq)
                V=torch.Tensor(Vvq)
                W=torch.Tensor(Vwq)
                # 【注意】LES 文件未读取独立压力，原代码直接让 P 引用 W 张量。
                P=W
                # 打印密度场形状，例如 [Nz,Ny,Nx]。
                print(RHO.shape)
                # 检查密度张量是否还含 NaN 或 Inf。
                print(torch.isnan(RHO).any(), torch.isinf(RHO).any())
                # 坐标乘 100 做单位/数值尺度转换，z 还整体减 3.5 对齐场景原点。
                X=torch.Tensor(xq)*100
                Y=torch.Tensor(yq)*100
                Z=torch.Tensor(zq)*100-3.5
                # 04ms 映射为时间 0，每后续 1ms 数据帧的模型时间增加 4。
                Time = torch.ones_like(X) * (idx-4) *4
                # 把当前帧的坐标、时间和场值全部展平为一维样本列，再加入跨帧列表。
                Xs.append(X.reshape(-1))
                Ys.append(Y.reshape(-1))
                Zs.append(Z.reshape(-1))
                Times.append(Time.reshape(-1))
                RHOs.append(RHO.reshape(-1))
                Us.append(U.reshape(-1))
                Vs.append(V.reshape(-1))
                Ws.append(W.reshape(-1))
                Ps.append(P.reshape(-1))

        # 先分别把所有帧的 X/Y/Z 一维列拼起来，再加一个列维并沿 dim=1 组成 [N,3] 坐标。
        inputsxyz = torch.cat((torch.cat(Xs).unsqueeze(1), torch.cat(Ys).unsqueeze(1),
                            torch.cat(Zs).unsqueeze(1), ), dim=1)
        # 合并所有帧时间并转成 [N,1]。
        inputsT=torch.cat(Times).unsqueeze(1)
        # 合并所有帧的密度标签，得到 [N]。
        outputsRHO = torch.cat(RHOs)
        # 合并三个速度分量并除以 400 归一化，降低数值量级。
        outputsU = torch.cat(Us)/400
        outputsV = torch.cat(Vs)/400
        outputsW = torch.cat(Ws)/400
        # 合并压力列；LES 分支中它实际与 W 相同。
        outputsP = torch.cat(Ps)
        # 打印各速度分量和 P 的最大绝对值，用于检查归一化尺度。
        print(max(abs(outputsU)),max(abs(outputsV)),max(abs(outputsW)),max(abs(outputsP)))
        # 将七个等长张量绑成数据集，按一个索引就能同时取出一个点的全部物理量。
        dataset = TensorDataset(inputsxyz,inputsT, outputsRHO,outputsU,outputsV,outputsW,outputsP)
        # 计算全数据集 2% 的样本数，用于缓存子集。
        subset_size = int(0.02 * len(dataset))
        # 生成全部索引的随机排列，只取前 subset_size 个。
        indices = torch.randperm(len(dataset))[:subset_size]
        # 用这些索引创建原 dataset 的 2% 子集视图。
        subset_dataset = Subset(dataset, indices)
        # 【注意】原代码保存的是 2% subset_dataset，但下一步返回的是完整 dataset。
        torch.save(subset_dataset, self.processed_file)
        # 输出缓存已保存的提示。
        print(f"Processed data saved to {self.processed_file}")
        # 返回本次内存中构建的完整数据集。
        return dataset
    def train_PDE(self):
        """
        【进阶：密度 + PIV + PDE 联合训练】用场真值、速度截面数据和 Navier-Stokes/连续性残差约束模型。

        输入：无显式参数，使用 self.model 和 Trainer 配置。
        输出：无显式返回；就地更新模型参数。
        数据形状：DataLoader 每批提供 XYZs1=[B,3]、Ts1=[B,1]、RHOs/Us/Vs/Ws/Ps=[B]；PDE 网格有 15×15×15×5=16875 点。
        物理意义：除了让密度预测符合数据，还尝试让速度/压力/密度共同满足质量和动量守恒。
        【注意】原代码将 PDE 损失乘了 0，因此当前实际总损失主要由密度/PIV/正则项组成。
        """
        # 清理未使用的 CUDA 缓存。
        torch.cuda.empty_cache()

        # # load the training and testing dataset and other settings.
        # train_dataset = self.train_dataset
        # test_dataset = self.test_dataset
        # 取出待训练模型。
        model = self.model
        #self.depth_data = test_dataset.depth_data
        # 取出 TensorBoard 日志写入器。
        summary_writer = self.summary_writer
        # 复制当前空间网格分辨率到局部变量。
        reso_cur = self.reso_cur

        # 读取/预处理 LES 或 DNS 数据，得到 (xyz,t,ρ,u,v,w,p) 数据集。
        dataset = self.prepare_data()
        # 如果之前已生成 PIV 截面数据缓存，直接加载。
        if os.path.exists('pivdataload.pt'):
            # 输出缓存命中提示。
            print(f"Loading processed data from {'pivdataload.pt'}")
            # pivdatasetall 每行预期为 [x,y,z,t,u,v,w]。
            pivdatasetall = torch.load('pivdataload.pt')
            # 打印缓存形状用于核对。
            print(pivdatasetall.shape)
        # 缓存不存在时，从完整数据集抽取 x 截面上的 PIV 样本并保存。
        else:
            # 为坐标、时间和三个速度分量分别建立分批累积列表。
            xyzpiv,tpiv,uspiv,vspiv,wspiv=[],[],[],[],[]
            # 【注意】int(1 * len(dataset)) 实际选取 100% 数据，与旧注释的 1% 不同；保留原代码。
            subset_size = int(1 * len(dataset))

            # 随机打乱数据集索引并取前 subset_size 个。
            indices = torch.randperm(len(dataset))[:subset_size]

            # 按索引创建数据子集。
            subset_dataset = Subset(dataset, indices)
            # 用最多 4 个工作进程、每批 400000 样本读取，并每轮打乱。
            subdataloader=DataLoader( subset_dataset,num_workers=4, batch_size=400000, shuffle=True)
            # 遍历大批数据，每个样本按顺序解包七个物理量。
            for XYZs, Ts, RHOs,Us,Vs,Ws,Ps in subdataloader:
                # 【注意】这里使用“或”：x<0.01 或 x>-0.01 对几乎所有实数 x 都为真；只标注不修改。
                mask=((XYZs[...,0]<0.01)|(XYZs[...,0]>-0.01))#|((XYZs[...,1]<0.01)&(XYZs[...,1]>-0.01))|((XYZs[...,1]<0.01)&(XYZs[...,1]>-0.01))
                # 用布尔掩码同步筛选 xyz、t、u、v、w 并加入列表。
                xyzpiv.append(XYZs[mask])
                tpiv.append(Ts[mask])
                uspiv.append(Us[mask])
                vspiv.append(Vs[mask])
                wspiv.append(Ws[mask])
                # 打印本批筛选后的坐标、时间和 u 形状。
                print(XYZs[mask].shape,Ts[mask].shape,Us[mask].shape)
            # 先跨批次 cat 每一列，再沿最后一维合成 [Npiv,7]=[x,y,z,t,u,v,w]。
            pivdatasetall=torch.cat((torch.cat(xyzpiv),
                        torch.cat(tpiv),
                        torch.cat(uspiv).unsqueeze(-1),
                        torch.cat(vspiv).unsqueeze(-1),
                        torch.cat(wspiv).unsqueeze(-1)),-1)
            # 输出组合后 PIV 数据形状。
            print(pivdatasetall.shape)

            # 将 PIV 数据张量序列化到当前工作目录。
            torch.save(pivdatasetall, 'pivdataload.pt')

        # 将全部 PIV 数据一次性移到训练设备。
        pivdatasetall=pivdatasetall.to(self.device)
        # 预计算时空网格升采样计划。
        self.get_voxel_upsample_list()
        # 创建空间平面 TV 正则器。
        tvreg_s = TVLoss()  # TV loss on the spatial planes
        # 创建时空平面 TV 正则器。
        tvreg_s_t = TVLoss(
            1.0, self.modelcfg.TV_t_s_ratio
        )  # TV loss on the spatial-temporal planes     
        # 创建总步数为 n_iters 的进度条。
        pbar = tqdm(
            range(self.optimcfg.n_iters),
            miniters=self.cfgsystems.progress_refresh_rate,
            file=sys.stdout,
        )

        # 向模型查询密度分支可训练参数组。
        grad_varsrho = model.get_optparam_groupsrho(self.optimcfg)
        #grad_varsgrad = model.get_optparam_groupsgrad(self.optimcfg)

        # 用密度参数组创建 Adam 优化器。
        optimizerrho = torch.optim.Adam(
            grad_varsrho, betas=(self.optimcfg.beta1, self.optimcfg.beta2)
        )
        # optimizergrad = torch.optim.Adam(
        #     grad_varsgrad, betas=(self.optimcfg.beta1, self.optimcfg.beta2)
        # )
        # 创建完优化器后再清理一次 CUDA 缓存。
        torch.cuda.empty_cache()
        # 手动迭代计数器，因为下面使用 while+for 双层循环而不是直接 for pbar。
        iteration = 0
        # db 是截面/边界范围的初始半宽。
        db=0.05
        # 将完整数据集分批；16 工作进程、锁页内存、每工作进程预取 100 批。
        dataloader=DataLoader(dataset,num_workers=16,pin_memory=True,prefetch_factor=100, batch_size=self.optimcfg.batch_size, shuffle=True)
        # 为设备上的 PIV 数据建立顺序索引 [0..Npiv-1]。
        indicespiv = torch.arange(len(pivdatasetall))
        # pointer 记录下一批 PIV 索引的起点。
        pointer = 0  # 指针，用于记录当前索引位置
        # PIV 每批使用主 batch_size 的 3 倍。
        batch_size=self.optimcfg.batch_size*3
        # 只要手动迭代数未达总步数，就重复遍历 DataLoader。
        while iteration < self.optimcfg.n_iters:
            # 从 DataLoader 中取一批密度/速度/压力真值。
            for XYZs1, Ts1, RHOs,Us,Vs,Ws,Ps in dataloader:
                # 将密度标签移到设备，并除以 23.5 做数值归一化。
                RHOs=RHOs.to(self.device)/23.5
                # Your training code here
                # Here, handle the training logic, loss computation, etc.

                # 手动将进度条向前推进 1 步。
                pbar.update(1)
                # 手动迭代计数加 1。
                iteration += 1
                # 将时间和坐标张量移到训练设备。
                Ts1=Ts1.to(self.device)
                XYZs1=XYZs1.to(self.device)
                
                # 【暂时不用深究】下面整段是“定期用当前模型生成新 PIV 伪标签”的旧实验，当前全部被注释。
                # 【暂时不用深究】下面被注释的区域是定期生成 PIV 伪标签的旧实验。
                # if ((iteration+1)%200==0)&(iteration>1000):
                #     #设置残差点范围
                #     size = 30         # 30*30*30 的空间网格
                #     time_start = 0.0  # 时间开始点
                #     time_end = 4.679568  # 时间结束点
                #     time_size = 20    # 10 个时间点

                #     # 生成四维网格
                #     grid_4d = self.generate_4d_grid(db-0.01, size, time_start, time_end, time_size)
                #     with torch.no_grad():
                #         velocity = model.PIVdatacal(grid_4d[...,:3].to(self.device), grid_4d[...,3].to(self.device))
                    
                #     pivdataset1=torch.cat((grid_4d.to(self.device),velocity[...,:3]),-1)#.to(self.device)

                #     #print(pivdataset.shape,pivdataset1.shape,velocity.shape)
                #     pivdataset=torch.cat((pivdataset1,pivdataset),0)
                #     db=2/10000*(iteration-1000)
                # 打印当前截面边界 [-db,db]。
                print('boundry:[',-db,',',db,']')
                # 【PIV 分批】如果当前排列中剩余索引不够一批，就用“旧轮剩余 + 新轮开头”拼成完整 batch。
                if pointer + batch_size > len(indicespiv):  # 如果剩余的数据不足一个batch
                    # 先保留当前排列未用完的尾部索引。
                    remaining_indices = indicespiv[pointer:]
                    # 生成一个新的 0..Npiv-1 随机排列。
                    indicespiv = torch.randperm(len(pivdatasetall))  # 重新洗牌
                    # 新排列的指针从 0 开始。
                    pointer = 0  # 重置指针
                    # 计算拼满本 batch 还需要从新排列取多少个索引。
                    needed = batch_size - len(remaining_indices)
                    # 把剩余索引与新排列前 needed 个索引串接。
                    current_indices = torch.cat((remaining_indices, indicespiv[pointer:pointer + needed]))
                    # 新排列的指针跳过刚用掉的 needed 个位置。
                    pointer += needed
                # 如果剩余数据足够，就直接取连续 batch_size 个索引。
                else:
                    # 从 pointer 开始切出本批索引。
                    current_indices = indicespiv[pointer:pointer + batch_size]
                    # 指针向后移动一个 batch。
                    pointer += batch_size
                # 用本批索引从 [Npiv,7] 中取出 [Bpiv,7] PIV 数据。
                pivdataset = pivdatasetall[current_indices]

                # 如果指针到达/超过数据末尾，为下一批准备全新随机排列。
                if pointer >= len(pivdatasetall):  # 如果所有数据都被抽取完毕
                    # 重新打乱 PIV 索引。
                    indicespiv = torch.randperm(len(pivdatasetall))
                    # 下轮从头开始。
                    pointer = 0
                # 【注意】这个条件同样使用“或”，对几乎所有 x 为真；且 maskPIV 后面实际未用。
                # 计算截面掩码；【注意】使用“或”使条件几乎总为真，且 maskPIV 在后面未实际使用。
                maskPIV=((XYZs1[...,0]<0.3+db)|(XYZs1[...,0]>-0.3-db))
                # 调用体渲染过程Render the rgb values of rays
                #print('trainer,framew2c',fra
                
                #print(Ts.max(),Ts.min())
                # Unpack the batch and move the data to the device
                #Xs, Ys, Zs, Ts, RHOs, Us, Vs, Ws, Ps, YO2s, YCH4s = (item for item in batch)
                # 计算当前学习率/正则权重衰减因子。
                lr_factor = self.get_lr_decay_factor(iteration)

                # 调用模型的 BOSdata 路径预测本批坐标/时间上的密度。
                preRHO1=model.BOSdata(XYZs1.squeeze(), Ts1.squeeze())
                # 【数据项】密度预测与归一化真值 RHOs 之间的均方误差。
                loss=torch.mean((preRHO1.squeeze()-RHOs)**2)

                # 【PDE 残差点】设定 x/y/z/t 坐标范围和每根轴点数。
                x_start, x_end, x_points = -2, 2, 15
                y_start, y_end, y_points = -2, 2, 15
                z_start, z_end, z_points = -3, 3, 15
                t_start, t_end, t_points = 0, 80, 5

                # 在等间隔 x 轴上加入小幅随机抖动，避免每次只在完全相同网格点计算 PDE。
                x = torch.linspace(x_start, x_end, x_points)+torch.rand(15)/5-1/2/5
                # y 轴加入同量级随机抖动。
                y = torch.linspace(y_start, y_end, y_points)+torch.rand(15)/5-1/2/5
                # z 轴范围更大，随机抖动幅度相应乘 3。
                z = torch.linspace(z_start, z_end, z_points)+torch.rand(15)/10*3-1/2/10*3
                # 在 5 个等间隔时间点上各加 [0,16) 随机偏移。
                t = torch.linspace(t_start, t_end, t_points)+torch.rand(5)*16

                # 穷举 15×15×15×5 个 x/y/z/t 组合。
                X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing='ij')
                # 将三个空间网格展平并叠为 [16875,3]，移到训练设备。
                XYZs=torch.stack((X.reshape(-1),Y.reshape(-1),Z.reshape(-1)),-1).to(self.device)
                # 将时间网格展平为 [16875]并移到设备。
                Ts=T.reshape(-1).to(self.device)
                # XYZs=XYZs[maskPIV]
                # Ts=Ts[maskPIV]
                # 允许对空间坐标求导，用于构造速度/密度空间偏导。
                XYZs.requires_grad = True
                # 允许对时间求导。
                Ts.requires_grad = True
                #print(XYZs.shape,Ts.shape)
                # 调用模型的非渲染路径，返回密度 preRHO 和 [u,v,w,p] 场 velocity。
                preRHO, velocity = model(XYZs, Ts.unsqueeze(-1), Ts, is_rendear=False)
                # 设定动力黏度常数；【注意】10e-5 在 Python 中等于 1e-4，保留原表达式。
                miu=3.9*10e-5
                # 根据原实验特征尺度/速度和黏度计算雷诺数 Re。
                Re=22.84*400*10e-2/miu
                # 【自动求导】密度对 x/y/z 的一阶偏导，形状 [16875,3]。
                RHO_xyz  = torch.autograd.grad(preRHO, XYZs, grad_outputs=torch.ones_like(preRHO), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # 密度对时间的一阶偏导，形状 [16875]。
                RHO_t  = torch.autograd.grad(preRHO, Ts, grad_outputs=torch.ones_like(preRHO), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # u 对 x/y/z 的一阶偏导 [u_x,u_y,u_z]。
                U_xyz  = torch.autograd.grad(velocity[...,0], XYZs, grad_outputs=torch.ones_like(velocity[...,0]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # 对 u_x 再对 x/y/z 求导；其第 0 列是 u_xx。
                U_x_xyz  = torch.autograd.grad(U_xyz[...,0], XYZs, grad_outputs=torch.ones_like(U_xyz[...,0]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # 对 u_y 再求空间导数；第 1 列是 u_yy。
                U_yy=torch.autograd.grad(U_xyz[...,1], XYZs, grad_outputs=torch.ones_like(U_xyz[...,1]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # 对 u_z 再求空间导数；第 2 列是 u_zz。
                U_zz=torch.autograd.grad(U_xyz[...,2], XYZs, grad_outputs=torch.ones_like(U_xyz[...,2]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # v 对 x/y/z 的一阶偏导。
                V_xyz  = torch.autograd.grad(velocity[...,1], XYZs, grad_outputs=torch.ones_like(velocity[...,1]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # 对 v_y 再求空间导数；第 1 列是 v_yy。
                V_y_xyz  = torch.autograd.grad(V_xyz[...,1], XYZs, grad_outputs=torch.ones_like(V_xyz[...,1]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # 对 v_x 再求导；第 0 列是 v_xx。
                V_xx=torch.autograd.grad(V_xyz[...,0], XYZs, grad_outputs=torch.ones_like(V_xyz[...,0]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # 对 v_z 再求导；第 2 列是 v_zz。
                V_zz=torch.autograd.grad(V_xyz[...,2], XYZs, grad_outputs=torch.ones_like(V_xyz[...,2]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # w 对 x/y/z 的一阶偏导。
                W_xyz  = torch.autograd.grad(velocity[...,2], XYZs, grad_outputs=torch.ones_like(velocity[...,2]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # 对 w_z 再求导；第 2 列是 w_zz。
                W_z_xyz  = torch.autograd.grad(W_xyz[...,2], XYZs, grad_outputs=torch.ones_like(W_xyz[...,2]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # 对 w_x 再求导；第 0 列是 w_xx。
                W_xx=torch.autograd.grad(W_xyz[...,0], XYZs, grad_outputs=torch.ones_like(W_xyz[...,0]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # 对 w_y 再求导；第 1 列是 w_yy。
                W_yy=torch.autograd.grad(W_xyz[...,1], XYZs, grad_outputs=torch.ones_like(W_xyz[...,1]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # 压力 p=velocity[...,3] 对 x/y/z 的一阶梯度。
                P_xyz  = torch.autograd.grad(velocity[...,3], XYZs, grad_outputs=torch.ones_like(velocity[...,2]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # u/v/w 三个速度分量分别对时间求导。
                U_gradt  = torch.autograd.grad(velocity[...,0], Ts, grad_outputs=torch.ones_like(velocity[...,0]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                V_gradt  = torch.autograd.grad(velocity[...,1], Ts, grad_outputs=torch.ones_like(velocity[...,1]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                W_gradt  = torch.autograd.grad(velocity[...,2], Ts, grad_outputs=torch.ones_like(velocity[...,2]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # 【连续性方程】L1 = ρ_t + ρ∇·u + u·∇ρ，理想物理解应让它接近 0。
                L1 =(RHO_t+preRHO*(U_xyz[...,0]+V_xyz[...,1]+W_xyz[...,2])+\
                        RHO_xyz[...,0]*velocity[...,0]+\
                        RHO_xyz[...,1]*velocity[...,1]+\
                        RHO_xyz[...,2]*velocity[...,2])
                # 【x 动量方程】L2 组合对流加速度、压力梯度和黏性/体积黏性项。
                L2 =((U_gradt+velocity[...,0]*U_xyz[...,0]+velocity[...,1]*U_xyz[...,1]+velocity[...,2]*U_xyz[...,2])*preRHO+\
                     (P_xyz[...,0]-1/Re*(U_x_xyz[...,0]+U_yy[...,1]+U_zz[...,2]+1/3*(U_x_xyz[...,0]+V_y_xyz[...,0]+W_z_xyz[...,0]))))#u_hat*rho_t+ rho_hat * (u_t+ + u_hat * u_x + v_hat * u_y) + p_x - (1/Re) * (u_xx + u_yy)
                # 【y 动量方程】L3 是 v 分量的对流-压力-黏性残差。
                L3 =((V_gradt+velocity[...,0]*V_xyz[...,0]+velocity[...,1]*V_xyz[...,1]+velocity[...,2]*V_xyz[...,2])*preRHO+\
                     (P_xyz[...,1]-1/Re*(V_y_xyz[...,1]+V_xx[...,0]+V_zz[...,2]+1/3*(U_x_xyz[...,1]+V_y_xyz[...,1]+W_z_xyz[...,1]))))
                # 【z 动量方程】L4 还包含经尺度换算后的重力项。
                L4 =((W_gradt+velocity[...,0]*W_xyz[...,0]+velocity[...,1]*W_xyz[...,1]+velocity[...,2]*W_xyz[...,2])*preRHO+\
                     (P_xyz[...,2]-1/Re*(W_z_xyz[...,2]+W_xx[...,0]+W_yy[...,1]+1/3*(U_x_xyz[...,2]+V_y_xyz[...,2]+W_z_xyz[...,2]))))-preRHO*9.8*100/400**2
                # L5 是预留的第五个物理残差，当前固定为 0 且后面未使用。
                L5 = 0
                #print(W_z_xyz[...,2],W_xx[...,0],W_yy[...,1],U_x_xyz[...,2],V_y_xyz[...,2],W_z_xyz[...,2])
                # 打印四个 PDE 残差的均方值，用于观察物理一致性。
                print(torch.mean(L1**2).item(),torch.mean(L2**2).item(),torch.mean(L3**2).item(),torch.mean(L4**2).item())
                # if RHO_gradxyz is not None:
                #     print("Gradient w.r.t XYZs:", RHO_gradxyz, RHO_gradxyz.shape)
                # else:
                #     print("XYZs do not affect preRHO")

                # if RHO_gradt is not None:
                #     print("Gradient w.r.t Ts:", RHO_gradt, RHO_gradt.shape)
                # else:
                #     print("Ts do not affect preRHO")
                #print(XYZs.shape, Ts.shape, RHOs.shape,preRHO.shape)
                # 密度分支正则损失累加器初始为 0。
                total_loss1=0#+torch.mean(velocity_gradxyz**2)+torch.mean(velocity_gradt**2)
                # 【注意】原代码明确乘 0，因此连续性和动量 PDE 均方误差当前不会进入总损失。
                total_loss2=0*(torch.mean(L1**2)+(torch.mean(L2**2)+torch.mean(L3**2)+torch.mean(L4**2)))/25

                # 用专门的 PIVdatacal 路径在本批 [x,y,z,t] 上预测 [u,v,w,…]。
                Velocity=model.PIVdatacal(pivdataset[...,:3],pivdataset[...,3])
                # 【PIV 数据项】比较预测前三个速度分量和数据最后三列 u/v/w 的 MSE。
                lossPIV=torch.mean((Velocity[...,:3]-pivdataset[...,-3:])**2)
                
                #print(torch.mean((preRHO)**2),torch.mean(preRHO),preRHO.max(),preRHO.min())
                
                # 再次计算当前衰减因子；与本次循环前面的值相同，保留原执行顺序。
                lr_factor = self.get_lr_decay_factor(iteration)

                # 迭代超过 1000 且配置权重为正时，计算密度平面 TV 正则。
                if (self.modelcfg.TV_weight_density > 0)&(iteration>1000):
                    # 计算衰减后的 TV 权重。
                    TV_weight_density = lr_factor * self.modelcfg.TV_weight_density
                    # 对空间/时空密度平面求 TV 并加权。
                    loss_tv = model.TV_loss_density(tvreg_s, tvreg_s_t) * TV_weight_density
                    # 累加进 total_loss1。
                    total_loss1 = total_loss1 + loss_tv
                    # summary_writer.add_scalar(
                    #     "train/reg_tv_density",
                    #     loss_tv.detach().item(),
                    #     global_step=iteration,
                    # )

               

                # 配置权重为正时对密度参数加 L1 稀疏正则。
                if self.modelcfg.L1_weight_density > 0:
                    # 计算衰减后的 L1 权重。
                    L1_weight_density = lr_factor * self.modelcfg.L1_weight_density
                    # 计算模型密度参数的加权 L1 损失。
                    loss_l1 = model.L1_loss_density() * L1_weight_density
                    # 累加进 total_loss1。
                    total_loss1 = total_loss1 + loss_l1
                    # summary_writer.add_scalar(
                    #     "train/reg_l1_density",
                    #     loss_l1.detach().item(),
                    #     global_step=iteration,
                    # )

               
                # # Loss on the rendered and gt depth maps.
                # if self.modelcfg.depth_loss and self.modelcfg.depth_loss_weight > 0:
                #     depth_loss = (depth_map.unsqueeze(-1) - depth) ** 2
                #     mask = depth != 0
                #     depth_loss = (
                #         torch.mean(depth_loss[mask]) * self.modelcfg.depth_loss_weight
                #     )
                #     total_loss += depth_loss
                #     summary_writer.add_scalar(
                #         "train/depth_loss",
                #         depth_loss.detach().item(),
                #         global_step=iteration,
                #     )
                # 密度主损失 = 密度正则 + ρ 数据 MSE。
                total_lossrho=total_loss1+loss
                # 打印 PDE/PIV/BOS（此处 loss 实际是密度数据 MSE）三类损失。
                print('PDE loss:',total_loss2.item(),'PIV loss:',lossPIV.item(),'BOS loss:',loss.item())
                # 把 PIV MSE 加到原本为 0 的 PDE 损失容器中。
                total_loss2=total_loss2+lossPIV
                #summary_writer.add_scalar("train/mse", total_lossrho, global_step=iteration)
                # 总损失 = 密度数据/正则 + PIV（PDE 当前因乘 0 不起作用）。
                total_lossrho=total_lossrho+total_loss2
                # 清空上一批梯度。
                optimizerrho.zero_grad()
                # optimizergrad.zero_grad()
                # total_loss2.backward(retain_graph=True)
                # optimizergrad.step()
                # 从总损失反向传播，计算模型参数梯度。
                total_lossrho.backward()
                # Adam 根据当前梯度更新模型参数。
                optimizerrho.step()
                # 记录参数信息
                # group=optimizerrho.param_groups[-1]
                # param=group['params']
                # print(param)
                # summary_writer.add_histogram(f"Gradients", param.grad, iteration)
                # summary_writer.add_histogram(f"Weights", param.data, iteration)
                # 把总损失转成 Python 浮点数用于显示。
                loss = total_lossrho.detach().item()
                
                # graph = make_dot(loss, params=dict([('input', rays_train)]))
                #     # Save the graph to a file
                # graph.render(f"computation_graph2", format='png', cleanup=True) 
                # 按配置频率更新进度条的迭代号和 MSE 文本。
                if iteration % self.cfgsystems.progress_refresh_rate == 0:
                    pbar.set_description(
                        f"Iteration {iteration:05d}:"
                        + f" mse = {loss:.6f}"
                    )
                
                # 依次更新 Adam 每个参数组的学习率。
                for param_group in optimizerrho.param_groups:
                    # 当前 lr = 初始 lr_org × 衰减因子。
                    param_group["lr"] = param_group["lr_org"] * lr_factor
                # for param_group in optimizergrad.param_groups:
                #     param_group["lr"] = param_group["lr_org"] * lr_factor

                

                # 在指定迭代步准备空体素掩码的分辨率。
                if iteration in self.modelcfg.update_emptymask_list:
                    # 体素总数小于 256³ 时才赋值 reso_mask。
                    if (
                        reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3
                    ):  # update volume resolution
                        # 使用当前 [Rx,Ry,Rz]。
                        reso_mask = reso_cur
                    # 真正更新掩码的调用被原作者注释。
                    #model.updateEmptyMask(tuple(reso_mask))

                # 当前步命中升采样计划时，提升时空网格分辨率。
                if iteration in self.modelcfg.upsample_list:
                    # aligned 模式使旧网格点与新网格点对齐。
                    if self.modelcfg.upsampling_type == "aligned":
                        # 每轴执行 R_new=2R_old-1。
                        reso_cur = [reso_cur[i] * 2 - 1 for i in range(len(reso_cur))]
                    # unaligned 模式从预计算列表取下一个体素总数。
                    else:
                        # pop(0) 消费下一个总体素数。
                        N_voxel = self.N_voxel_list.pop(0)
                        # 根据包围盒转换成三轴分辨率。
                        reso_cur = N_to_reso(
                            N_voxel, model.aabb, self.modelcfg.nonsquare_voxel
                        )
                    # 取下一阶段时间网格数。
                    time_grid = self.Time_grid_list.pop(0)
                    # 根据新分辨率重新计算每光线采样数；该函数后续不渲染，但保留原计算。
                    nSamples = min(
                        self.modelcfg.nSamples,
                        cal_n_samples(reso_cur, self.modelcfg.step_ratio),
                    )
                    # 对模型参数网格做插值/扩容。
                    model.upsample_volume_grid(reso_cur, time_grid)
                    # 重新查询新网格密度参数组。
                    grad_varsrho = model.get_optparam_groupsrho(self.optimcfg, 1.0)
                    #grad_varsgrad = model.get_optparam_groupsgrad(self.optimcfg, 1.0)
                    # 用新参数对象重建 Adam。
                    optimizerrho = torch.optim.Adam(
                        grad_varsrho, betas=(self.optimcfg.beta1, self.optimcfg.beta2)
                    )
                    # optimizergrad = torch.optim.Adam(
                    #     grad_varsgrad, betas=(self.optimcfg.beta1, self.optimcfg.beta2)
                    # )
                # 内层 for 可能在一次 DataLoader 遍历中超过总步数，因此在每批末尾主动检查。
                if iteration >= self.optimcfg.n_iters:
                    # 只跳出内层 for；外层 while 随后也因条件不成立而结束。
                    break

    def train_PDE_FVM(self):
        """
        【进阶：有限体积风格约束】直接在模型的离散时空网格上重建密度/速度场，用相邻单元差分构造 TV 和密度运输残差。

        输入：无显式参数，使用 self.model、数据集和配置。
        输出：无显式返回；就地更新模型参数。
        核心形状：
            gRHO 通常为 [Rx,Ry,Rz,Rt,1]，gUP 通常为 [Rx,Ry,Rz,Rt,C]，其前个通道可解读为 u/v/w/p。
        与 train_PDE 的区别：
            train_PDE 对随机连续坐标用 autograd 求导；本函数从模型的平面-时间线分解中构造整张网格，再用相邻网格差分。
        """
        # 清理未使用的 CUDA 缓存。
        torch.cuda.empty_cache()

        # 取出训练集；【注意】本函数后续未直接使用 train_dataset。
        train_dataset = self.train_dataset
        # 取出测试集，用于读取 depth_data 标志。
        test_dataset = self.test_dataset
        # 取出待训练模型。
        model = self.model
        # 保存是否有深度数据；后续深度损失代码被注释。
        self.depth_data = test_dataset.depth_data
        # 取出 TensorBoard 日志写入器。
        summary_writer = self.summary_writer
        # 复制当前三维网格分辨率。
        reso_cur = self.reso_cur

        # 加载/预处理 (xyz,t,ρ,u,v,w,p) 数据集。
        dataset = self.prepare_data()
        # 如果已有 PIV 缓存就直接加载。
        if os.path.exists('pivdataload.pt'):
            # 输出缓存加载提示。
            print(f"Loading processed data from {'pivdataload.pt'}")
            # pivdataset 每行预期是 [x,y,z,t,u,v,w]。
            pivdataset = torch.load('pivdataload.pt')
            # 打印数据形状。
            print(pivdataset.shape)
        # 否则从完整数据集构造 PIV 子集。
        else:
            # 为 xyz/t/u/v/w 分别建立累积列表。
            xyzpiv,tpiv,uspiv,vspiv,wspiv=[],[],[],[],[]
            # 随机选取数据集的 20%。
            subset_size = int(0.2 * len(dataset))

            # 打乱全部索引并取前 20%。
            indices = torch.randperm(len(dataset))[:subset_size]

            # 创建对原数据集的索引子集。
            subset_dataset = Subset(dataset, indices)
            # 用 4 工作进程、每批 400000 样本遍历子集。
            subdataloader=DataLoader( subset_dataset,num_workers=4, batch_size=400000, shuffle=True)
            # 逐批解包坐标、时间和五个物理场。
            for XYZs, Ts, RHOs,Us,Vs,Ws,Ps in subdataloader:
                # 筛选坐标截面；【注意】第一个 x 条件用“或”使其几乎总为真，且后两项重复。
                mask=((XYZs[...,0]<0.01)|(XYZs[...,0]>-0.01))|((XYZs[...,1]<0.01)&(XYZs[...,1]>-0.01))|((XYZs[...,1]<0.01)&(XYZs[...,1]>-0.01))
                # 对 xyz/t/u/v/w 使用相同布尔掩码并追加到列表。
                xyzpiv.append(XYZs[mask])
                tpiv.append(Ts[mask])
                uspiv.append(Us[mask])
                vspiv.append(Vs[mask])
                wspiv.append(Ws[mask])
                # 打印本批筛选后的坐标/时间/u 形状。
                print(XYZs[mask].shape,Ts[mask].shape,Us[mask].shape)
            # 先跨批拼接，再沿最后一维组成 [Npiv,7]。
            pivdataset=torch.cat((torch.cat(xyzpiv),
                        torch.cat(tpiv),
                        torch.cat(uspiv).unsqueeze(-1),
                        torch.cat(vspiv).unsqueeze(-1),
                        torch.cat(wspiv).unsqueeze(-1)),-1)
            # 打印组合后的 PIV 张量形状。
            print(pivdataset.shape)

            # 保存 PIV 缓存，供以后直接加载。
            torch.save(pivdataset, 'pivdataload.pt')

        # 将 PIV 数据移到训练设备。
        pivdataset=pivdataset.to(self.device)
        # 预计算网格升采样计划。
        self.get_voxel_upsample_list()
        # 创建空间和时空 TV 正则器。
        tvreg_s = TVLoss()  # TV loss on the spatial planes
        tvreg_s_t = TVLoss(
            1.0, self.modelcfg.TV_t_s_ratio
        )  # TV loss on the spatial-temporal planes     
        # 创建训练进度条。
        pbar = tqdm(
            range(self.optimcfg.n_iters),
            miniters=self.cfgsystems.progress_refresh_rate,
            file=sys.stdout,
        )

        # 获取密度分支参数组。
        grad_varsrho = model.get_optparam_groupsrho(self.optimcfg)
        # 获取速度/appearance 梯度分支参数组。
        grad_varsgrad = model.get_optparam_groupsgrad(self.optimcfg)

        # 用两类参数共同创建 Adam 优化器。
        optimizerrho = torch.optim.Adam(
            grad_varsrho+grad_varsgrad, betas=(self.optimcfg.beta1, self.optimcfg.beta2)
        )
        # optimizergrad = torch.optim.Adam(
        #     grad_varsgrad, betas=(self.optimcfg.beta1, self.optimcfg.beta2)
        # )
        # 释放未使用 CUDA 缓存。
        torch.cuda.empty_cache()
        # 手动迭代计数从 0 开始。
        iteration = 0
        # 截面半宽/边界宽度初值。
        db=0.05
        # 建立完整数据集的高并行分批读取器。
        dataloader=DataLoader(dataset,num_workers=16,pin_memory=True,prefetch_factor=100, batch_size=self.optimcfg.batch_size, shuffle=True)
        # 只要未达总步数，就重复遍历数据集。
        while iteration < self.optimcfg.n_iters:
            # 取出一批坐标、时间、密度、速度和压力。
            for XYZs1, Ts1, RHOs,Us,Vs,Ws,Ps in dataloader:
                # 将密度真值移到设备并除 23.5 归一化。
                RHOs=RHOs.to(self.device)/23.5
                # Your training code here
                # Here, handle the training logic, loss computation, etc.

                # 手动更新进度条和迭代计数。
                pbar.update(1)
                iteration += 1
                # 将时间和坐标移到训练设备。
                Ts1=Ts1.to(self.device)
                XYZs1=XYZs1.to(self.device)
                
                # if ((iteration+1)%200==0)&(iteration>1000):
                #     #设置残差点范围
                #     size = 30         # 30*30*30 的空间网格
                #     time_start = 0.0  # 时间开始点
                #     time_end = 4.679568  # 时间结束点
                #     time_size = 20    # 10 个时间点

                #     # 生成四维网格
                #     grid_4d = self.generate_4d_grid(db-0.01, size, time_start, time_end, time_size)
                #     with torch.no_grad():
                #         velocity = model.PIVdatacal(grid_4d[...,:3].to(self.device), grid_4d[...,3].to(self.device))
                    
                #     pivdataset1=torch.cat((grid_4d.to(self.device),velocity[...,:3]),-1)#.to(self.device)

                #     #print(pivdataset.shape,pivdataset1.shape,velocity.shape)
                #     pivdataset=torch.cat((pivdataset1,pivdataset),0)
                #     db=2/10000*(iteration-1000)
                #print('boundry:[',-db,',',db,']')

                
                maskPIV=((XYZs1[...,0]<0.3+db)|(XYZs1[...,0]>-0.3-db))
                # 调用体渲染过程Render the rgb values of rays
                #print('trainer,framew2c',fra
                
                #print(Ts.max(),Ts.min())
                # Unpack the batch and move the data to the device
                #Xs, Ys, Zs, Ts, RHOs, Us, Vs, Ws, Ps, YO2s, YCH4s = (item for item in batch)
                # 计算当前学习率/正则权重衰减因子。
                lr_factor = self.get_lr_decay_factor(iteration)

                # 在真值坐标/时间上预测密度。
                preRHO1=model.BOSdata(XYZs1.squeeze(), Ts1.squeeze())
                # 【数据损失】计算预测密度与归一化真值的 MSE。
                loss=torch.mean((preRHO1.squeeze()-RHOs)**2)

                # 为连续 PDE 求导备选路径设定时空网格范围/点数。
                x_start, x_end, x_points = -2, 2, 15
                y_start, y_end, y_points = -2, 2, 15
                z_start, z_end, z_points = -3, 3, 15
                t_start, t_end, t_points = 0, 80, 5

                # 生成带随机抖动的 x/y/z/t 坐标轴。
                x = torch.linspace(x_start, x_end, x_points)+torch.rand(15)/5-1/2/5
                y = torch.linspace(y_start, y_end, y_points)+torch.rand(15)/5-1/2/5
                z = torch.linspace(z_start, z_end, z_points)+torch.rand(15)/10*3-1/2/10*3
                t = torch.linspace(t_start, t_end, t_points)+torch.rand(5)*16

                # 生成 15×15×15×5 四维网格。
                X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing='ij')
                # 将空间坐标展平为 [16875,3] 并移到设备。
                XYZs=torch.stack((X.reshape(-1),Y.reshape(-1),Z.reshape(-1)),-1).to(self.device)
                # 将时间展平为 [16875]。
                Ts=T.reshape(-1).to(self.device)
                # XYZs=XYZs[maskPIV]
                # Ts=Ts[maskPIV]
                # 使 XYZs/Ts 可求导；【注意】本 FVM 路径后面未用它们计算模型输出。
                XYZs.requires_grad = True
                Ts.requires_grad = True
                #print(XYZs.shape,Ts.shape)
                #preRHO, velocity = model(XYZs, Ts.unsqueeze(-1), Ts, is_rendear=False)
                # 保留与连续 PDE 版一致的黏度和雷诺数计算；当前后续未使用 Re。
                miu=3.9*10e-5
                Re=(22.84*400*10e-2)/miu
                # 【张量分解导读】密度场由 xy×zt、xz×yt、yz×xt 三组平面-时间线外积表示。
                # einsum 将第 1 组 [n,x,y] 与 [n,z,t] 按分量 n 配对相乘，得 [n,x,y,z,t]。
                density1 = torch.einsum('nxy,nzt->nxyzt', model.density_plane[0].squeeze(),model.density_line_time[0].squeeze())
                # 第 2 组用 xz 平面与 yt 时空线组合。
                density2 = torch.einsum('nxz,nyt->nxyzt', model.density_plane[1].squeeze(),model.density_line_time[1].squeeze())
                # 第 3 组用 yz 平面与 xt 时空线组合。
                density3 = torch.einsum('nyz,nxt->nxyzt', model.density_plane[2].squeeze(),model.density_line_time[2].squeeze())
                #print(density1.shape,density2.shape,density3.shape)
                # 沿分量维将三组五维特征串接。
                inter= torch.cat((density1,density2,density3), dim=0)
                #print(inter.shape)
                # 把时空网格展平，得 [总分量数, Rx*Ry*Rz*Rt]。
                inter=inter.reshape(sum(model.density_n_comp),-1)

                # 转置为 [网格点数,总分量数]，再用线性基矩阵投影到密度特征空间。
                inter = model.density_basis_mat(inter.T)
                #print(inter.shape)
                # 密度回归器接收展平特征、恢复的五维特征网格和再次展平特征。
                # 回归输出乘 (rho_bd[1]-rho_bd[0]) 并加 rho0，从归一化值还原到密度范围。
                gRHO = model.density_regressor(
                                inter,
                                inter.reshape(model.gridSize[0],model.gridSize[1],model.gridSize[2],model.time_grid,-1),
                                inter,
                            )*(model.rho_bd[1]-model.rho_bd[0])+model.rho0
                # 【速度/appearance 场】用同样的三对平面-时间线构造整张五维特征网格。
                density1 = torch.einsum('nxy,nzt->nxyzt', model.app_plane[0].squeeze(),model.app_line_time[0].squeeze())
                density2 = torch.einsum('nxz,nyt->nxyzt', model.app_plane[1].squeeze(),model.app_line_time[1].squeeze())
                density3 = torch.einsum('nyz,nxt->nxyzt', model.app_plane[2].squeeze(),model.app_line_time[2].squeeze())
                #print(density1.shape,density2.shape,density3.shape)
                # 拼接三组 appearance 分量。
                inter= torch.cat((density1,density2,density3), dim=0)
                #print(inter.shape)
                # 将时空网格维展平。
                inter=inter.reshape(sum(model.app_n_comp),-1)

                # 用 appearance 基矩阵做特征投影。
                inter = model.app_basis_mat(inter.T)
                #print(inter.shape)
                # 用 appearance 回归器生成整张速度/压力场 gUP。
                gUP = model.app_regressor(
                                inter,
                                inter.reshape(model.gridSize[0],model.gridSize[1],model.gridSize[2],model.time_grid,-1),
                                inter,
                            )
                # 【时间中心差分】用 t-1 与 t+1 密度差除以原实验时间间隔 8，得到内部网格的 ∂ρ/∂t。
                dgRHO=-(gRHO[1:-1,1:-1,1:-1,:-2,0]-gRHO[1:-1,1:-1,1:-1,2:,0])/8
                #(gUP[2:,1:-1,1:-1,1:-1,:3]+gUP[1:-1,1:-1,1:-1,1:-1,:3])/2
                # 【质量通量散度】对 ρu、ρv、ρw 在 x/y/z 方向做中心差分并相加，再除以网格间距。
                grad_RHOU=(gUP[2:,1:-1,1:-1,1:-1,0]*gRHO[2:,1:-1,1:-1,1:-1,0]/2-gUP[:-2,1:-1,1:-1,1:-1,0]*gRHO[:-2,1:-1,1:-1,1:-1,0]/2+\
                        gUP[1:-1,2:,1:-1,1:-1,1]*gRHO[1:-1,2:,1:-1,1:-1,0]/2-gUP[1:-1,:-2,1:-1,1:-1,1]*gRHO[1:-1,:-2,1:-1,1:-1,0]/2+\
                        gUP[1:-1,1:-1,2:,1:-1,2]*gRHO[1:-1,1:-1,2:,1:-1,0]/2-gUP[1:-1,1:-1,:-2,1:-1,2]*gRHO[1:-1,1:-1,:-2,1:-1,0]/2)/(2*model.scene_bbox_max[0]/model.gridSize[0])
                # 打印密度时间导数与质量通量散度的形状，两者应可对齐比较。
                print('dgRHO:',dgRHO.shape,grad_RHOU.shape)
                #print(density1.shape,model.app_plane[0].shape,model.app_line_time[0].shape)
                # 密度场 TV：分别计算 x/y/z/t 相邻网格差的平方均值并相加。
                LOSSTV=torch.mean((gRHO[1:]-gRHO[:-1])**2)+\
                        torch.mean((gRHO[:,1:]-gRHO[:,:-1])**2)+\
                        torch.mean((gRHO[:,:,1:]-gRHO[:,:,:-1])**2)+\
                        torch.mean((gRHO[:,:,:,1:]-gRHO[:,:,:,:-1])**2)
                # 【连续性约束】惩罚 ∇·(ρu) 与原代码定义的 dgRHO 的差；detach 阻止这一项通过 dgRHO 更新密度分支。
                LOSSTV_PDE=torch.mean((grad_RHOU-dgRHO.detach())**2)
                # 速度/appearance 场在 x/y/z/t 四个方向的 TV 平滑损失。
                LOSSTV_UP= torch.mean((gUP[1:]-gUP[:-1])**2)+\
                        torch.mean((gUP[:,1:]-gUP[:,:-1])**2)+\
                        torch.mean((gUP[:,:,1:]-gUP[:,:,:-1])**2)+\
                        torch.mean((gUP[:,:,:,1:]-gUP[:,:,:,:-1])**2)
                # 【暂时不用深究】下面被注释的大段是其他特征融合方式和 autograd PDE 公式，当前 FVM 路径不执行。
                # # Fusion Two
                # if self.fusion_two == "multiply":
                #     inter = torch.prod(inter, dim=0)
                # elif self.fusion_two == "sum":
                #     inter = torch.sum(inter, dim=0)
                # elif self.fusion_two == "concat":
                #     inter = inter.view(-1, inter.shape[-1])
                # else:
                #     raise NotImplementedError("no such fusion type")
                # interRHO = self.density_basis_mat(interRHO.T)  # Feature Projection
                # interUP = self.app_basis_mat(interUP.T)  # Feature Projection
                # RHO_xyz  = torch.autograd.grad(preRHO, XYZs, grad_outputs=torch.ones_like(preRHO), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # RHO_t  = torch.autograd.grad(preRHO, Ts, grad_outputs=torch.ones_like(preRHO), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # U_xyz  = torch.autograd.grad(velocity[...,0], XYZs, grad_outputs=torch.ones_like(velocity[...,0]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # U_x_xyz  = torch.autograd.grad(U_xyz[...,0], XYZs, grad_outputs=torch.ones_like(U_xyz[...,0]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # U_yy=torch.autograd.grad(U_xyz[...,1], XYZs, grad_outputs=torch.ones_like(U_xyz[...,1]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # U_zz=torch.autograd.grad(U_xyz[...,2], XYZs, grad_outputs=torch.ones_like(U_xyz[...,2]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # V_xyz  = torch.autograd.grad(velocity[...,1], XYZs, grad_outputs=torch.ones_like(velocity[...,1]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # V_y_xyz  = torch.autograd.grad(V_xyz[...,1], XYZs, grad_outputs=torch.ones_like(V_xyz[...,1]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # V_xx=torch.autograd.grad(V_xyz[...,0], XYZs, grad_outputs=torch.ones_like(V_xyz[...,0]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # V_zz=torch.autograd.grad(V_xyz[...,2], XYZs, grad_outputs=torch.ones_like(V_xyz[...,2]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # W_xyz  = torch.autograd.grad(velocity[...,2], XYZs, grad_outputs=torch.ones_like(velocity[...,2]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # W_z_xyz  = torch.autograd.grad(W_xyz[...,2], XYZs, grad_outputs=torch.ones_like(W_xyz[...,2]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # W_xx=torch.autograd.grad(W_xyz[...,0], XYZs, grad_outputs=torch.ones_like(W_xyz[...,0]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # W_yy=torch.autograd.grad(W_xyz[...,1], XYZs, grad_outputs=torch.ones_like(W_xyz[...,1]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # P_xyz  = torch.autograd.grad(velocity[...,3], XYZs, grad_outputs=torch.ones_like(velocity[...,2]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # U_gradt  = torch.autograd.grad(velocity[...,0], Ts, grad_outputs=torch.ones_like(velocity[...,0]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # V_gradt  = torch.autograd.grad(velocity[...,1], Ts, grad_outputs=torch.ones_like(velocity[...,1]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # W_gradt  = torch.autograd.grad(velocity[...,2], Ts, grad_outputs=torch.ones_like(velocity[...,2]), create_graph = True, retain_graph = True, only_inputs=True)[0]
                # L1 =(RHO_t+preRHO*(U_xyz[...,0]+V_xyz[...,1]+W_xyz[...,2])+\
                #         RHO_xyz[...,0]*velocity[...,0]+\
                #         RHO_xyz[...,1]*velocity[...,1]+\
                #         RHO_xyz[...,2]*velocity[...,2])
                # L2 =((U_gradt+velocity[...,0]*U_xyz[...,0]+velocity[...,1]*U_xyz[...,1]+velocity[...,2]*U_xyz[...,2])*preRHO+\
                #      (P_xyz[...,0]-1/Re*(U_x_xyz[...,0]+U_yy[...,1]+U_zz[...,2]+1/3*(U_x_xyz[...,0]+V_y_xyz[...,0]+W_z_xyz[...,0]))))#u_hat*rho_t+ rho_hat * (u_t+ + u_hat * u_x + v_hat * u_y) + p_x - (1/Re) * (u_xx + u_yy)
                # L3 =((V_gradt+velocity[...,0]*V_xyz[...,0]+velocity[...,1]*V_xyz[...,1]+velocity[...,2]*V_xyz[...,2])*preRHO+\
                #      (P_xyz[...,1]-1/Re*(V_y_xyz[...,1]+V_xx[...,0]+V_zz[...,2]+1/3*(U_x_xyz[...,1]+V_y_xyz[...,1]+W_z_xyz[...,1]))))
                # L4 =((W_gradt+velocity[...,0]*W_xyz[...,0]+velocity[...,1]*W_xyz[...,1]+velocity[...,2]*W_xyz[...,2])*preRHO+\
                #      (P_xyz[...,2]-1/Re*(W_z_xyz[...,2]+W_xx[...,0]+W_yy[...,1]+1/3*(U_x_xyz[...,2]+V_y_xyz[...,2]+W_z_xyz[...,2]))))-preRHO*9.8*100/400**2
                # 预留的第五个物理残差，当前固定为 0 且后面未使用。
                L5 = 0
                #print(W_z_xyz[...,2],W_xx[...,0],W_yy[...,1],U_x_xyz[...,2],V_y_xyz[...,2],W_z_xyz[...,2])
                #print(torch.mean(L1**2).item(),torch.mean(L2**2).item(),torch.mean(L3**2).item(),torch.mean(L4**2).item())
                # if RHO_gradxyz is not None:
                #     print("Gradient w.r.t XYZs:", RHO_gradxyz, RHO_gradxyz.shape)
                # else:
                #     print("XYZs do not affect preRHO")

                # if RHO_gradt is not None:
                #     print("Gradient w.r.t Ts:", RHO_gradt, RHO_gradt.shape)
                # else:
                #     print("Ts do not affect preRHO")
                #print(XYZs.shape, Ts.shape, RHOs.shape,preRHO.shape)
                # 密度分支通用正则项累加器。
                total_loss1=0#+torch.mean(velocity_gradxyz**2)+torch.mean(velocity_gradt**2)
                # appearance/速度分支通用正则项累加器。
                total_loss2=0#(torch.mean(L1**2)+(torch.mean(L2**2)+torch.mean(L3**2)+torch.mean(L4**2)))/25

                # 在缓存 PIV 坐标/时间上预测速度。
                Velocity=model.PIVdatacal(pivdataset[...,:3],pivdataset[...,3])
                # 【PIV 损失】预测 u/v/w 与真值最后三列的 MSE。
                lossPIV=torch.mean((Velocity[...,:3]-pivdataset[...,-3:])**2)
                
                #print(torch.mean((preRHO)**2),torch.mean(preRHO),preRHO.max(),preRHO.min())
                
                # 重新计算当前衰减因子，用于正则权重和学习率。
                lr_factor = self.get_lr_decay_factor(iteration)

                # 迭代超过 1000 且密度 TV 权重为正时，启用模型平面 TV 正则。
                if (self.modelcfg.TV_weight_density > 0)&(iteration>1000):
                    # 计算衰减后的密度 TV 权重。
                    TV_weight_density = lr_factor * self.modelcfg.TV_weight_density
                    # 计算模型分解平面上的密度 TV（与上面整网格 LOSSTV 是两个不同项）。
                    loss_tv = model.TV_loss_density(tvreg_s, tvreg_s_t) * TV_weight_density
                    # 累加到密度正则和。
                    total_loss1 = total_loss1 + loss_tv
                    # 写入 TensorBoard。
                    summary_writer.add_scalar(
                        "train/reg_tv_density",
                        loss_tv.detach().item(),
                        global_step=iteration,
                    )

                # appearance 平面 TV 正则。
                if self.modelcfg.TV_weight_app > 0:
                    # 计算衰减后权重。
                    TV_weight_app = lr_factor * self.modelcfg.TV_weight_app
                    # 计算 appearance 分解平面 TV。
                    loss_tv = model.TV_loss_app(tvreg_s, tvreg_s_t) * TV_weight_app
                    # 累加到 total_loss2。
                    total_loss2 = total_loss2 + loss_tv
                    # 记录数值到 TensorBoard。
                    summary_writer.add_scalar(
                        "train/reg_tv_app", loss_tv.detach().item(), global_step=iteration
                    )

                # 密度平面 L1 稀疏正则。
                if self.modelcfg.L1_weight_density > 0:
                    # 计算衰减后 L1 权重。
                    L1_weight_density = lr_factor * self.modelcfg.L1_weight_density
                    # 对模型密度参数求加权 L1。
                    loss_l1 = model.L1_loss_density() * L1_weight_density
                    # 累加并写日志。
                    total_loss1 = total_loss1 + loss_l1
                    summary_writer.add_scalar(
                        "train/reg_l1_density",
                        loss_l1.detach().item(),
                        global_step=iteration,
                    )

                # appearance 平面 L1 稀疏正则。
                if self.modelcfg.L1_weight_app > 0:
                    # 计算衰减后权重。
                    L1_weight_app = lr_factor * self.modelcfg.L1_weight_app
                    # 计算 appearance 加权 L1。
                    loss_l1 = model.L1_loss_app() * L1_weight_app
                    # 累加并写 TensorBoard。
                    total_loss2 = total_loss2 + loss_l1
                    summary_writer.add_scalar(
                        "train/reg_l1_app", loss_l1.detach().item(), global_step=iteration
                    )

                # # Loss on the rendered and gt depth maps.
                # if self.modelcfg.depth_loss and self.modelcfg.depth_loss_weight > 0:
                #     depth_loss = (depth_map.unsqueeze(-1) - depth) ** 2
                #     mask = depth != 0
                #     depth_loss = (
                #         torch.mean(depth_loss[mask]) * self.modelcfg.depth_loss_weight
                #     )
                #     total_loss += depth_loss
                #     summary_writer.add_scalar(
                #         "train/depth_loss",
                #         depth_loss.detach().item(),
                #         global_step=iteration,
                #     )
                # 【密度主损失】通用密度正则 + 数据 MSE + 整网格密度 TV×10e-2 + 连续性差分残差。
                total_lossrho=total_loss1+loss+LOSSTV*10e-2+LOSSTV_PDE#+LOSSTV_UP#*10e-2
                #print('PDE loss:',total_loss2.item(),'PIV loss:',lossPIV.item(),'BOS loss:',loss.item())
                # 打印密度 TV、速度 TV、连续性残差和密度数据损失。
                print('TV loss RHO:',LOSSTV.item(),'TV loss UP:',LOSSTV_UP.item(),LOSSTV_PDE.item(),'BOS loss:',loss.item())
                # 将 PIV 速度 MSE 加入 appearance/速度分支损失。
                total_loss2=total_loss2+lossPIV
                # 记录尚未加 total_loss2 的密度主损失。
                summary_writer.add_scalar("train/mse", total_lossrho, global_step=iteration)
                # 最终总损失再加 appearance/PIV 分支。
                total_lossrho=total_lossrho+total_loss2
                # 清空上一批梯度。
                optimizerrho.zero_grad()
                # optimizergrad.zero_grad()
                # total_loss2.backward(retain_graph=True)
                # optimizergrad.step()
                # 【主线】反向传播总损失，同时为密度和 appearance/速度参数求梯度。
                total_lossrho.backward()
                # Adam 执行一次参数更新。
                optimizerrho.step()
                # 记录参数信息
                # group=optimizerrho.param_groups[-1]
                # param=group['params']
                # print(param)
                # summary_writer.add_histogram(f"Gradients", param.grad, iteration)
                # summary_writer.add_histogram(f"Weights", param.data, iteration)
                # 取出普通 Python 浮点损失用于显示。
                loss = total_lossrho.detach().item()
                
                # graph = make_dot(loss, params=dict([('input', rays_train)]))
                #     # Save the graph to a file
                # graph.render(f"computation_graph2", format='png', cleanup=True) 
                # 按配置频率更新进度条文本。
                if iteration % self.cfgsystems.progress_refresh_rate == 0:
                    pbar.set_description(
                        f"Iteration {iteration:05d}:"
                        + f" mse = {loss:.6f}"
                    )
                
                # 更新 Adam 所有参数组的学习率。
                for param_group in optimizerrho.param_groups:
                    # 当前 lr=初始 lr_org×lr_factor。
                    param_group["lr"] = param_group["lr_org"] * lr_factor
                # for param_group in optimizergrad.param_groups:
                #     param_group["lr"] = param_group["lr_org"] * lr_factor

                

                # 在配置时刻准备空体素掩码分辨率。
                if iteration in self.modelcfg.update_emptymask_list:
                    # 只有总体素数小于 256³ 时赋值。
                    if (
                        reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3
                    ):  # update volume resolution
                        # 使用当前分辨率。
                        reso_mask = reso_cur
                    # 真正更新 EmptyMask 的调用被原代码注释。
                    #model.updateEmptyMask(tuple(reso_mask))

                # 命中升采样时刻时扩大时空参数网格。
                if iteration in self.modelcfg.upsample_list:
                    # aligned 模式每轴按 2R-1 放大。
                    if self.modelcfg.upsampling_type == "aligned":
                        reso_cur = [reso_cur[i] * 2 - 1 for i in range(len(reso_cur))]
                    # unaligned 模式从预计算列表取下一个体素总数。
                    else:
                        # 消费下一个目标体素数。
                        N_voxel = self.N_voxel_list.pop(0)
                        # 结合包围盒反推三轴分辨率。
                        reso_cur = N_to_reso(
                            N_voxel, model.aabb, self.modelcfg.nonsquare_voxel
                        )
                    # 消费下一个时间网格数。
                    time_grid = self.Time_grid_list.pop(0)
                    # 根据新分辨率更新每光线采样数的预计算值。
                    nSamples = min(
                        self.modelcfg.nSamples,
                        cal_n_samples(reso_cur, self.modelcfg.step_ratio),
                    )
                    # 对模型密度和 appearance 网格做插值/扩容。
                    model.upsample_volume_grid(reso_cur, time_grid)
                    # 重新取密度参数组。
                    grad_varsrho = model.get_optparam_groupsrho(self.optimcfg, 1.0)
                    # 重新取 appearance/速度参数组。
                    grad_varsgrad = model.get_optparam_groupsgrad(self.optimcfg, 1.0)
                    # 【注意】升采样前 Adam 包含 grad_varsrho+grad_varsgrad，此处原代码重建后只传 grad_varsrho；只标注不修改。
                    optimizerrho = torch.optim.Adam(
                        grad_varsrho, betas=(self.optimcfg.beta1, self.optimcfg.beta2)
                    )
                    # optimizergrad = torch.optim.Adam(
                    #     grad_varsgrad, betas=(self.optimcfg.beta1, self.optimcfg.beta2)
                    # )
                # 每批结束时检查是否已达总迭代数。
                if iteration >= self.optimcfg.n_iters:
                    # 跳出内层 DataLoader 循环，随后外层 while 也会结束。
                    break
