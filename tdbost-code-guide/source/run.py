# 【文件作用】这是整个 TDBOST 项目的主入口文件。
# 你在终端运行 python run.py 时，Python 会从本文件最下面开始进入 main()。
"""
基于张量分解的四维背景纹影层析成像（TDBOST）训练入口。

给零基础读者的主线：
1. 读取配置；
2. 准备 BOS 数据；
3. 建立用于表示折射率场的 MMmodel；
4. 用 Trainer 反复优化模型；
5. 保存模型，并把重构出的三维场画成切片图。

“四维”通常指三维空间 (x, y, z) 再加一个时间维度 t。
"""

# os 用来处理文件夹和文件路径。
import os
# datetime 用来把当前日期和时间加入实验文件夹名称，避免覆盖旧实验。
import datetime
# torch 是 PyTorch：负责张量计算、神经网络训练和 GPU 加速。
import torch
# numpy 常简称 np：负责 CPU 上的多维数组和数值计算。
import numpy as np
# matplotlib.pyplot 常简称 plt：负责画图和保存图片。
import matplotlib.pyplot as plt
# loadmat 用来读取 MATLAB 的 .mat 数据文件。
from scipy.io import loadmat
# scipy 提供插值、滤波等科学计算工具；下文会用 scipy.ndimage。
import scipy
# RegularGridInterpolator 用来把规则网格上的真值插值到目标网格。
from scipy.interpolate import RegularGridInterpolator
# OmegaConf 把 dataclass 配置转换成可用字典，方便用 **config_dict 传参。
from omegaconf import OmegaConf
# SummaryWriter 把训练损失等信息写入 TensorBoard 日志。
from torch.utils.tensorboard import SummaryWriter

# 从本项目配置文件导入四组配置：模型、数据、优化器和系统。
from configs.config import ModelConfig, DataConfig, OptimConfig, SystemConfig
# N_to_reso 根据体素总数和三维边界，算出 x/y/z 三个方向的网格分辨率。
from render.util.util import N_to_reso
# MMmodel 是本次实际使用的张量分解折射率场模型。
from TDmodel.MMmodel import MMmodel
# 这两个函数分别建立测试集和训练集对象。
from dataloader import get_test_dataset, get_train_dataset
# Trainer 封装了采样、计算损失、反向传播和更新参数的训练循环。
from All_util.trainer import Trainer
# preprocess_bost_data 把原始 BOS 数据整理成训练代码能读取的格式。
from dataloader.data_preprocess import preprocess_bost_data

# gridspec 用来把一张大图划分成规则的小图区域。
import matplotlib.gridspec as gridspec


def plot_multi_slices(n_field, title_prefix, save_path):
    """
    把一个三维场沿 X、Y、Z 三个方向分别切成多张二维切片并保存。

    参数：
        n_field: 三维 NumPy 数组，形状约为 [Nx, Ny, Nz]。
                 每个元素表示该空间位置的预测折射率或相关标量。
        title_prefix: 图片标题前缀，例如 Prediction 或 Ground Truth。
        save_path: 最终 PNG 图片的保存路径。

    返回：
        没有返回值；结果直接写成图片文件。

    【物理直觉】三维体数据很难直接显示，所以像医学 CT 一样从三个方向切片观察。
    """
    # 读取三维数组在 X、Y、Z 三个方向分别有多少个格点。
    shape_x, shape_y, shape_z = n_field.shape
    # 在 X 方向 15% 到 85% 的范围内，等间距选 6 个整数切片位置。
    x_slices = np.linspace(int(shape_x * 0.15), int(shape_x * 0.85), 6, dtype=int)
    # 在 Y 方向也选 6 个切片位置。
    y_slices = np.linspace(int(shape_y * 0.15), int(shape_y * 0.85), 6, dtype=int)
    # 在 Z 方向也选 6 个切片位置。
    z_slices = np.linspace(int(shape_z * 0.15), int(shape_z * 0.85), 6, dtype=int)

    # 创建一张宽 15 英寸、高 25 英寸的画布。
    fig = plt.figure(figsize=(15, 25))
    # 把整张画布的背景设置为白色。
    fig.patch.set_facecolor('white')
    # 把画布划分为 6 行 6 列，后面每两列显示一个方向的切片。
    gs = gridspec.GridSpec(6, 6, figure=fig)

    # 循环 6 次；每一次在同一行画 X、Y、Z 三张切片。
    for i in range(6):
        # 【X 切片】占当前行的第 0、1 两列。
        ax_x = fig.add_subplot(gs[i, 0:2])
        # 固定 X 索引，保留 Y、Z 两个方向，因此得到二维图。
        im_x = ax_x.imshow(n_field[x_slices[i], :, :], cmap='jet')
        # 为这张切片增加颜色条；颜色数值对应场值大小。
        plt.colorbar(im_x, ax=ax_x, fraction=0.046, pad=0.04)
        # 标题中写明是哪个 X 索引处的切片。
        ax_x.set_title(f'[{title_prefix}] X Slice at {x_slices[i]}', fontweight="bold")
        # 隐藏像素坐标轴，使结果图更简洁。
        ax_x.axis('off')

        # 【Y 切片】占当前行的第 2、3 两列。
        ax_y = fig.add_subplot(gs[i, 2:4])
        # 固定 Y 索引，保留 X、Z 两个方向。
        im_y = ax_y.imshow(n_field[:, y_slices[i], :], cmap='jet')
        # 为 Y 切片增加颜色条。
        plt.colorbar(im_y, ax=ax_y, fraction=0.046, pad=0.04)
        # 标题中写明是哪个 Y 索引处的切片。
        ax_y.set_title(f'[{title_prefix}] Y Slice at {y_slices[i]}', fontweight="bold")
        # 隐藏坐标轴。
        ax_y.axis('off')

        # 【Z 切片】占当前行的第 4、5 两列。
        ax_z = fig.add_subplot(gs[i, 4:6])
        # 固定 Z 索引，保留 X、Y 两个方向。
        im_z = ax_z.imshow(n_field[:, :, z_slices[i]], cmap='jet')
        # 为 Z 切片增加颜色条。
        plt.colorbar(im_z, ax=ax_z, fraction=0.046, pad=0.04)
        # 标题中写明是哪个 Z 索引处的切片。
        ax_z.set_title(f'[{title_prefix}] Z Slice at {z_slices[i]}', fontweight="bold")
        # 隐藏坐标轴。
        ax_z.axis('off')

    # 自动调整子图间距，尽量避免标题和颜色条互相遮挡。
    plt.tight_layout()
    # 以 200 dpi 保存图片，并裁掉画布外多余的空白。
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    # 关闭画布，及时释放内存；训练后画大图时这一点很重要。
    plt.close()


def evaluate_and_plot(model, device, logfolder, dataset_target, time_step=0.5):
    """
    在规则三维网格上查询训练好的模型，并保存预测/真值切片图。

    参数：
        model: 已训练的 MMmodel。
        device: 计算设备，例如 cuda:3 或 cpu。
        logfolder: 本次实验输出文件夹。
        dataset_target: 数据集名称。当前函数保留了此参数，但原代码没有实际使用它。
        time_step: 要观察的归一化时间点，默认 0.5。

    返回：
        没有返回值；会在 logfolder 中保存图片。

    【主线】模型不是一次输出完整三维数组，而是接收很多 (x,y,z,t) 查询坐标，
    再给出每个坐标处的场值。这里先造坐标网格，再把预测值拼回三维数组。
    """
    # 在终端显示评估已经开始。
    print("\n[INFO] Starting validation and slice plotting...")
    # 切换到评估模式；这会关闭某些只在训练时启用的层行为。
    model.eval()

    # 三个空间坐标都覆盖 [-2, 2]，因此边界半宽 bd 为 2。
    bd = 2
    # X 方向取 101 个等间距坐标。
    x = torch.linspace(-bd, bd, 101)
    # Y 方向取 101 个等间距坐标。
    y = torch.linspace(-bd, bd, 101)
    # Z 方向取 151 个等间距坐标。
    z = torch.linspace(-bd, bd, 151)
    # meshgrid 把三条一维坐标轴扩展成三个 [101,101,151] 坐标体。
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    # 把 X/Y/Z 叠到最后一维，得到 [101,101,151,3] 的三维坐标表，并搬到 device。
    xyz = torch.stack((X, Y, Z), dim=-1).to(device)

    # 生成与 X 同形状的时间张量，每个空间点都使用同一个 time_step。
    t = torch.ones_like(X).to(device) * time_step
    # no_grad 表示这里只做预测，不保存反向传播所需的梯度图，从而节省显存。
    with torch.no_grad():
        # 把网格拉平成点列表 [N,3] 和时间列表 [N,1]，逐点查询模型。
        # n_pred 是预测场值；下划线接住但忽略模型返回的第二个结果。
        n_pred, _ = model(xyz.reshape(-1, 3), t.reshape(-1, 1), t, is_rendear=False)
    # 把一维预测列表还原为 [101,101,151]，移回 CPU，并转成 NumPy 数组。
    n_pred = n_pred.reshape(101, 101, 151).cpu().numpy()

    # 组合预测切片图的完整保存路径。
    pred_save_path = os.path.join(logfolder, 'validation_slices_Prediction.png')
    # 画出预测三维场的多方向切片。
    plot_multi_slices(n_pred, "Prediction", pred_save_path)
    # 在终端告诉用户图片保存在哪里。
    print(f"[INFO] Prediction slices saved to: {pred_save_path}")

    # 指定用于比较的 MATLAB 真值文件路径。
    gt_path = f'./data/jetflameinterpolated_data_10ms.mat'
    # 先检查真值文件是否真的存在，避免直接读取时报错。
    if os.path.exists(gt_path):
        # try 表示尝试读取；若格式或维度不符合预期，会进入 except 而不让程序崩溃。
        try:
            # 读取 .mat 文件，得到一个“变量名 -> 数组”的字典。
            data = loadmat(gt_path)
            # 真值原网格的 X 坐标有 151 个点。
            xq = torch.linspace(-2, 2, 151)
            # 真值原网格的 Y 坐标有 101 个点。
            yq = torch.linspace(-2, 2, 101)
            # 真值原网格的 Z 坐标有 101 个点。
            zq = torch.linspace(-2, 2, 101)
            # 从 MATLAB 字典中取出名为 Vq 的三维真值数组。
            Vq = data['Vq']

            def fill_nan(values):
                """
                给局部滤波器使用：若窗口中心是 NaN，就用窗口内非 NaN 均值填补。

                values 是一个局部 6×6×6 窗口拉平后的一维数组。
                6×6×6=216，原代码检查索引 108 附近的中心元素。
                """
                # 如果局部窗口中心值是 NaN（缺失值）。
                if np.isnan(values[108]):
                    # 用该窗口中所有非 NaN 数值的平均值替代。
                    return np.nanmean(values)
                # 如果中心值有效，就原样返回中心值。
                return values[108]

            # 在 Vq 上滑动 6×6×6 窗口，调用 fill_nan 修补缺失值。
            Vq = scipy.ndimage.generic_filter(Vq, fill_nan, size=(6, 6, 6), mode='mirror')
            # 建立三维规则网格插值器，以便在预测网格坐标处查询真值。
            interpolator = RegularGridInterpolator(
                # 下面三条坐标轴的顺序必须与 Vq 的三个维度顺序一致。
                (zq.numpy(), yq.numpy(), xq.numpy()),
                # 只取 Vq 第三维前 151 个位置，与 xq 长度相配。
                Vq[:, :, :151],
                # 使用线性插值。
                method='linear',
                # 查询点超出原网格边界时不抛异常。
                bounds_error=False,
                # 越界点使用 Vq 的全局非 NaN 平均值。
                fill_value=np.nanmean(Vq)
            )
            # 按插值器要求的轴顺序组织每个查询点。
            query_points = np.stack((Y.cpu().numpy(), X.cpu().numpy(), Z.cpu().numpy()), axis=-1)
            # 在所有查询坐标上插值得到与预测网格对应的真值 n_gt。
            n_gt = interpolator(query_points)
            # 在终端说明真值读取和插值成功。
            print("[INFO] Ground Truth data loaded and interpolated successfully.")

            # 组合真值切片图的保存路径。
            gt_save_path = os.path.join(logfolder, 'validation_slices_GroundTruth.png')
            # 画出真值三维场的多方向切片。
            plot_multi_slices(n_gt, "Ground Truth", gt_save_path)
            # 在终端告诉用户真值图片保存在哪里。
            print(f"[INFO] Ground Truth slices saved to: {gt_save_path}")

        # 捕获上面真值读取、滤波或插值过程中的任何异常。
        except Exception as e:
            # 打印警告而不是终止整个程序；e 中包含具体错误原因。
            print(f"[Warning] Could not parse GT data: {e}")
    # 如果真值文件根本不存在，就只保留预测图。
    else:
        # 明确告诉用户缺少哪个文件。
        print(f"[Warning] GT file '{gt_path}' not found. Visualized predictions only.")

    # 在终端显示验证流程结束。
    print("[INFO] Validation finished.\n")


def main():
    """
    串起一次完整 TDBOST 实验的总控制函数。

    【建议第一次阅读只追这 8 步】
    配置 → 数据路径 → 输出目录 → 数据集 → 网格 → 模型 → Trainer → 训练/保存/评估。
    """
    # 如果 CUDA 可用就选择第 4 块 GPU（编号从 0 开始，所以写 cuda:3），否则使用 CPU。
    # 【注意】只有 1 块 GPU 的电脑通常应改成 cuda:0；这里保留师兄原始设置。
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # 在终端显示实际选择的计算设备。
    print(f"[INFO] Using device: {device}")

    # 建立模型超参数对象。
    modelcfg = ModelConfig()
    # 建立优化器和训练迭代超参数对象。
    optimcfg = OptimConfig()
    # 建立数据路径、相机和采样超参数对象。
    cfgdata = DataConfig()
    # 建立日志、随机种子等系统超参数对象。
    cfgsystem = SystemConfig()
    # 把 ModelConfig 转成 OmegaConf 配置；稍后可用 ** 展开为模型构造参数。
    config_dict = OmegaConf.structured(modelcfg)

    # 选择 spray（喷雾）数据集；这是一个普通字符串变量。
    dataset_target = "spray"
    # 根据数据集名称拼出目标数据文件夹，例如 ./data/bosdata_spray。
    target_data_dir = f"./data/bosdata_{dataset_target}"

    # 如果目标目录中没有训练元数据 JSON，说明数据还没有完成预处理。
    if not os.path.exists(os.path.join(target_data_dir, "transforms_train_s.json")):
        # 在终端说明接下来会自动预处理。
        print("[INFO] Dataset not found, triggering preprocessing...")
        # 调用预处理函数，把原始数据整理到 target_data_dir。
        preprocess_bost_data(
            # 告诉预处理函数当前是 spray 数据。
            dataset_type=dataset_target,
            # 原始训练数据所在目录。
            data_dir="./data/traindata",
            # 整理后的训练数据输出目录。
            output_dir=target_data_dir,
            # 使用 9 个相机角度或观测角度。
            angle_N=9,
            # 使用 20 个时间步。
            time_steps=20
        )

    # 把确定的数据目录写回数据配置，供 dataloader 使用。
    cfgdata.datadir = target_data_dir

    # 如果配置要求在实验名后加入时间戳。
    if cfgsystem.add_timestamp:
        # 把当前时间格式化，例如 -20260711-121530。
        time_str = datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")
        # 组合日志根目录、实验名和时间戳。
        logfolder = f"{cfgsystem.basedir}/{cfgsystem.expname}{time_str}"
    # 如果不加入时间戳，就使用固定实验目录。
    else:
        # 组合日志根目录与实验名。
        logfolder = f"{cfgsystem.basedir}/{cfgsystem.expname}"

    # 创建实验输出目录；exist_ok=True 表示已存在时不报错。
    os.makedirs(logfolder, exist_ok=True)
    # 创建专门存放可视化图片的子目录。
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    # 创建 TensorBoard 日志写入器，把事件文件存进 logs 子目录。
    summary_writer = SummaryWriter(os.path.join(logfolder, "logs"))

    # 在终端显示数据加载阶段开始。
    print("[INFO] Loading dataset...")
    # 如果采样类型不是 rays，训练数据会按图像/批次堆叠；当前默认 rays，所以为 False。
    is_stack_train = (cfgdata.datasampler_type != "rays")
    # 构造训练数据集对象。
    train_dataset = get_train_dataset(cfgdata, is_stack=is_stack_train, experiment=False)
    # 构造测试数据集对象；测试时通常保留堆叠结构，便于整幅图评估。
    test_dataset = get_test_dataset(cfgdata, is_stack=True, experiment=False)

    # 取出训练场景的三维轴对齐包围盒 aabb，并搬到计算设备。
    aabb = train_dataset.scene_bbox.to(device)
    # 根据初始体素总数和包围盒尺寸，算出初始三维网格分辨率。
    reso_cur = N_to_reso(modelcfg.N_voxel_init, aabb, modelcfg.nonsquare_voxel)

    # 如果要求“对齐式上采样”，就把每一维修正成奇数尺寸。
    if modelcfg.upsampling_type == "aligned":
        # 对每个分辨率先整除 2、再乘 2、再加 1，得到最近的奇数。
        reso_cur = [reso_cur[i] // 2 * 2 + 1 for i in range(len(reso_cur))]

    # 在终端显示模型初始化阶段开始。
    print("[INFO] Initializing MMmodel...")
    # 如果配置明确要求加载已有模型，并且给出了检查点路径。
    if modelcfg.existmodel and modelcfg.ckpt:
        # 从磁盘读取完整模型，并映射到当前设备。
        model = torch.load(modelcfg.ckpt, map_location=device)
    # 否则从头新建一个 MMmodel。
    else:
        # **config_dict 会把配置中的键值展开成一组命名参数传入模型。
        model = MMmodel(aabb, reso_cur, device, modelcfg.time_grid_init, cfgsystem.near_far, **config_dict)

    # 建立 Trainer；它保存模型、数据、配置、日志和设备，之后负责训练。
    trainer = Trainer(
        # 要被优化的折射率场模型。
        model,
        # 模型结构与正则化配置。
        modelcfg,
        # 学习率、批量大小和迭代次数配置。
        optimcfg,
        # 数据与相机配置。
        cfgdata,
        # 日志和系统配置。
        cfgsystem,
        # 当前网格分辨率。
        reso_cur,
        # 训练数据集。
        train_dataset,
        # 测试数据集。
        test_dataset,
        # TensorBoard 日志写入器。
        summary_writer,
        # 本次实验输出文件夹。
        logfolder,
        # CPU 或 GPU 设备。
        device,
    )

    # 在终端说明训练即将开始。
    print("[INFO] Starting training loop...")
    # 进入 Trainer 中的主训练循环；这是整个程序计算量最大的步骤。
    trainer.train()

    # 组合最终模型文件路径，扩展名 .th 是常见的 PyTorch 模型后缀。
    save_path = f"{logfolder}/{cfgsystem.expname}.th"
    # 把训练后的完整模型对象保存到磁盘。
    torch.save(model, save_path)
    # 在终端显示模型保存位置。
    print(f"[INFO] Training complete. Model saved to {save_path}")

    # 在 time_step=0.5 的时刻查询三维场，并保存预测/真值切片。
    evaluate_and_plot(model, device, logfolder, dataset_target, time_step=0.5)


# 只有直接执行本文件时条件才成立；若别的文件 import run，则不会自动训练。
if __name__ == "__main__":
    # 调用上面的总控制函数。
    main()
