# 【文件作用】集中保存 TDBOST 的全部可调参数。
# 零基础读者可把它理解成“实验控制面板”：这里只规定数值，不执行训练。

# dataclass 能自动生成保存配置所需的初始化函数。
from dataclasses import dataclass, field
# List 表示列表类型，Optional 表示该值也可以是 None。
from typing import List, Optional


# @dataclass 会把下面每一个字段自动变成 ModelConfig 对象中的属性。
@dataclass
class ModelConfig:
    """
    模型结构、张量网格、正则化和物理常数配置。

    创建 ModelConfig() 后，可以用 modelcfg.N_voxel_init 这样的写法取值。
    【建议】第一次只看 N_voxel_init、time_grid_init、density_n_comp、
    upsample_list 和 nSamples，其他参数先知道“可调”即可。
    """

    # 模型名字；主入口实际直接构造 MMmodel，这里主要起记录作用。
    model_name: str = "MMmodel"
    # 初始三维网格的目标体素总数：83×83×83。
    N_voxel_init: int = 83 * 83 * 83
    # 最终上采样后希望达到的体素总数：200×200×200。
    N_voxel_final: int = 200 * 200 * 200
    # 光线上相邻采样点间距相对于体素尺寸的比例。
    step_ratio: float = 1.2
    # 是否按照包围盒长宽高调整各轴分辨率，而不强迫三轴相同。
    nonsquare_voxel: bool = True
    # 初始时间网格包含 20 个时间节点。
    time_grid_init: int = 20
    # 最终时间网格仍包含 20 个时间节点。
    time_grid_final: int = 20
    # 坐标/特征使用的归一化类型名称。
    normalize_type: str = "normal"
    # 在第 3000、6000 次迭代时提高空间网格分辨率。
    upsample_list: List[int] = field(default_factory=lambda: [3000, 6000])
    # 在这些迭代次数上更新“空区域掩码”，减少无效空间计算。
    update_emptymask_list: List[int] = field(
        # default_factory 为每个配置对象新建一份列表，避免对象之间共享同一列表。
        default_factory=lambda: [4000, 8000, 16000, 20000, 30000, 40000]
    )

    # ---------- 张量平面初始化 ----------
    # 折射率/密度分支在三个平面方向各使用 30 个分量（可理解为张量秩）。
    density_n_comp: List[int] = field(default_factory=lambda: [30, 30, 30])
    # 外观分支在三个平面方向各使用 1 个分量。
    app_n_comp: List[int] = field(default_factory=lambda: [1, 1, 1])
    # 融合后的密度特征维数。
    density_dim: int = 20
    # 外观特征维数，3 常对应三个输出通道。
    app_dim: int = 3
    # 把密度特征变成输出时使用普通 MLP。
    DensityMode: str = "general_MLP"
    # 把外观特征变成输出时也使用普通 MLP。
    AppMode: str = "general_MLP"
    # 张量参数随机初始化的尺度。
    init_scale: float = 0.1
    # 张量参数初始化后的整体平移量。
    init_shift: float = 0.0

    # ---------- 特征融合方式 ----------
    # 第一阶段把部分特征逐元素相乘。
    fusion_one: str = "multiply"
    # 第二阶段把特征沿通道维拼接。
    fusion_two: str = "concat"

    # ---------- 密度/折射率特征 ----------
    # 使用 softplus 把原始网络值转换为平滑的非负值。
    fea2denseAct: str = "softplus"
    # softplus 前对数值施加的偏移。
    density_shift: float = 0.0
    # 把采样距离换算到密度/积分尺度的系数；当前为 0。
    distance_scale: float = 0.0

    # ---------- 密度回归 MLP ----------
    # 时间位置编码的频率级数；-1 表示按当前实现关闭。
    density_t_pe: int = -1
    # 空间坐标位置编码使用 8 级频率。
    density_pos_pe: int = 8
    # 观察方向位置编码；-1 表示关闭。
    density_view_pe: int = -1
    # 输入特征再做 2 级频率的位置编码。
    density_fea_pe: int = 2
    # 密度 MLP 隐藏层宽度为 200。
    density_featureC: int = 200
    # 密度 MLP 包含 3 层。
    density_n_layers: int = 3

    # ---------- 外观回归 MLP ----------
    # 外观分支的时间位置编码级数。
    app_t_pe: int = 0
    # 外观分支的空间位置编码级数。
    app_pos_pe: int = 0
    # 外观分支的视角位置编码级数。
    app_view_pe: int = 0
    # -1 表示不对外观输入特征做位置编码。
    app_fea_pe: int = -1
    # 外观 MLP 隐藏层宽度为 200。
    app_featureC: int = 200
    # 外观 MLP 包含 2 层。
    app_n_layers: int = 2

    # ---------- 空区域掩码 ----------
    # 低于此阈值的区域可被视为“空”。
    emptyMask_thes: float = 0.01
    # 光线行进中，权重低于此阈值的采样点可被忽略。
    rayMarch_weight_thres: float = 0.0001

    # ---------- 正则化和辅助损失 ----------
    # 是否在训练时随机改变背景颜色。
    random_background: bool = False
    # 是否启用深度损失。
    depth_loss: bool = False
    # 深度损失在总损失中的权重。
    depth_loss_weight: float = 1.0
    # 是否启用距离分布损失。
    dist_loss: bool = False
    # 距离分布损失在总损失中的权重。
    dist_loss_weight: float = 0.01

    # 时间方向与空间方向 TV 正则的相对比例。
    TV_t_s_ratio: float = 1.0
    # 密度张量的总变分（TV）正则权重，使场在邻近位置更平滑。
    TV_weight_density: float = 0.001
    # 外观张量的 TV 正则权重；0 表示关闭。
    TV_weight_app: float = 0.0
    # 密度张量的 L1 正则权重；0 表示关闭。
    L1_weight_density: float = 0.0
    # 外观张量的 L1 正则权重；0 表示关闭。
    L1_weight_app: float = 0.0

    # ---------- 光线采样 ----------
    # grid_sample 是否把网格角点精确对齐到 -1 和 1。
    align_corners: bool = True
    # 网格上采样方式；unaligned 表示不强制奇数网格对齐。
    upsampling_type: str = "unaligned"
    # 每条光线可使用的最大采样点数量上限；实际值通常会在模型中重新计算。
    nSamples: int = 1000000

    # ---------- 检查点与三维空间边界 ----------
    # 是否直接载入已经训练好的模型。
    existmodel: bool = False
    # 模型检查点路径；None 表示没有指定。
    ckpt: Optional[str] = None
    # 重构空间包围盒最小角点 [xmin, ymin, zmin]。
    scene_bbox_min: List[float] = field(default_factory=lambda: [-2.0, -2.0, -2.0])
    # 重构空间包围盒最大角点 [xmax, ymax, zmax]。
    scene_bbox_max: List[float] = field(default_factory=lambda: [2.0, 2.0, 2.0])
    # 时间归一化缩放系数。
    time_scale: float = 1.0
    # 参考密度或背景物性参数；具体单位要结合论文/数据说明确认。
    rho0: float = 23.6
    # rho 的允许边界范围。
    rho_bd: List[float] = field(default_factory=lambda: [5.0, 50.0])

    # ---------- BOS 物理常数 ----------
    # 探测器或物理长度的离散步长：2.2/320。
    physical_D_level: float = 2.2 / 320
    # Gladstone-Dale 相关物理系数：2.48/10000。
    physical_GD: float = 2.48 / 10000

    # ---------- 位移/畸变损失 ----------
    # 预测位移 dp 的数值缩放倍数，改善优化时的数值量级。
    dp_scale: float = 100.0
    # 位移损失加入总损失时的权重。
    dp_loss_weight: float = 0.001


@dataclass
class DataConfig:
    """
    数据目录、相机内参、采样方式和三阶段训练安排。

    【主线】BOSDataset 会读取这里的 datadir 和相机参数，
    再把每个像素变成“光线 + 观测位移 + 时间”训练样本。
    """

    # 默认数据目录；run.py 会根据 spray/fuel 选择结果覆盖它。
    datadir: str = "./data/bosdata"
    # 数据集注册名称；dataloader 根据 bos 选择 BOSDataset。
    dataset_name: str = "bos"
    # 图像下采样倍数；1.0 表示保持原分辨率。
    downsample: float = 1.0
    # 按独立光线随机采样训练样本。
    datasampler_type: str = "rays"
    # 是否根据相机光线进一步计算更紧的场景包围盒。
    cal_fine_bbox: bool = False
    # 可视化样本数；-1 常表示使用全部或由其他逻辑决定。
    N_vis: int = -1
    # 相机 X 方向焦距（像素单位）。
    fx: float = 1500.0
    # 相机 Y 方向焦距（像素单位）。
    fy: float = 1500.0
    # 相机主点的 X 像素坐标。
    cx: float = 99.0
    # 相机主点的 Y 像素坐标。
    cy: float = 100.0

    # 数据配置中的场景最小边界。
    scene_bbox_min: List[float] = field(default_factory=lambda: [-2.0, -2.0, -2.0])
    # 数据配置中的场景最大边界。
    scene_bbox_max: List[float] = field(default_factory=lambda: [2.0, 2.0, 2.0])
    # 随机生成的相机姿态数量。
    N_random_pose: int = 20000

    # 第一训练阶段迭代 30000 次。
    stage_1_iteration: int = 30000
    # 第二训练阶段迭代 25000 次。
    stage_2_iteration: int = 25000
    # 第三训练阶段迭代 10000 次。
    stage_3_iteration: int = 10000
    # 关键帧数量。
    key_f_num: int = 10
    # 第一阶段使用的 gamma 系数。
    stage_1_gamma: float = 0.001
    # 第二阶段使用的 gamma 系数。
    stage_2_gamma: float = 0.02
    # 第三阶段使用的 alpha 系数。
    stage_3_alpha: float = 0.1
    # 数据时间坐标缩放系数。
    time_scale: float = 1.0


@dataclass
class OptimConfig:
    """
    优化器配置：决定参数每次更新多大、一次使用多少样本、训练多久。
    """

    # 密度/折射率张量网格参数的学习率。
    lr_density_grid: float = 0.02
    # 外观张量网格参数的学习率。
    lr_app_grid: float = 0.02
    # 密度 MLP 参数的学习率。
    lr_density_nn: float = 0.001
    # 外观 MLP 参数的学习率。
    lr_app_nn: float = 0.001

    # Adam 优化器的一阶动量衰减系数。
    beta1: float = 0.9
    # Adam 优化器的二阶动量衰减系数。
    beta2: float = 0.99
    # 学习率采用指数衰减。
    lr_decay_type: str = "exp"
    # 衰减结束时，学习率变成初始学习率的 0.1 倍。
    lr_decay_target_ratio: float = 0.1
    # 用 10000 步完成规定的学习率衰减。
    lr_decay_step: int = 10000
    # 网格上采样后是否重置学习率。
    lr_upsample_reset: bool = True

    # 每次训练随机取 2048 条光线/样本。
    batch_size: int = 2048
    # Trainer 主训练循环总共迭代 20000 次。
    n_iters: int = 20000


@dataclass
class SystemConfig:
    """
    运行环境和输出目录配置。
    """

    # 随机种子；相同环境下有助于复现实验采样。
    seed: int = 20240131
    # 所有实验输出的根目录。
    basedir: str = "./log"
    # 系统层面的检查点路径；当前为 None。
    ckpt: Optional[str] = None
    # 进度条每 10 步刷新一次。
    progress_refresh_rate: int = 10
    # 每 19999 步执行一次可视化。
    vis_every: int = 19999
    # 是否在实验文件夹名后加入当前时间。
    add_timestamp: bool = True
    # 实验名称，也用于最终模型文件名。
    expname: str = "BOSdata_TDBOST"
    # 沿相机光线进行采样的近、远距离边界。
    near_far: List[float] = field(default_factory=lambda: [27.0, 33.0])
