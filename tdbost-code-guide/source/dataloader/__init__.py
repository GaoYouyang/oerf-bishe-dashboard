"""数据集入口：根据配置创建训练集或测试集。"""

# 从同一 dataloader 包中导入真正负责读文件、造光线的 BOSDataset 类。
from .BOS_dataset import BOSDataset

def get_train_dataset(cfgdata, is_stack=False,experiment=True):
    """【主线】根据 ``cfgdata`` 创建并返回 BOS 训练数据集。

    ``cfgdata`` 是带有 datadir、downsample、相机内参等属性的配置对象。
    默认 ``is_stack=False`` 会把所有图像展平成大量单光线，适合随机抽取训练。
    """
    # 当配置明确指定 bos 数据集时，进入当前项目唯一实现的分支。
    if cfgdata.dataset_name == "bos":
        # 实例化 BOSDataset，并把配置中的路径、尺度、包围盒和相机参数原样传入。
        train_dataset = BOSDataset(
            # 数据根目录。
            cfgdata.datadir,
            # 固定选择训练划分，对应 transforms_train*.json。
            "train",
            # 图像下采样倍数。
            cfgdata.downsample,
            # False 通常将帧维与像素维合并成光线表。
            is_stack=is_stack,
            # 决定读真实实验相机数据，还是仿真数据。
            experiment=experiment,
            # 是否由光线 near/far 重新估计更紧的三维包围盒。
            cal_fine_bbox=cfgdata.cal_fine_bbox,
            # 可视化/加载帧数配置。
            N_vis=cfgdata.N_vis,
            # 时间坐标缩放系数。
            time_scale=cfgdata.time_scale,
            # 三维场 xyz 下界。
            scene_bbox_min=cfgdata.scene_bbox_min,
            # 三维场 xyz 上界。
            scene_bbox_max=cfgdata.scene_bbox_max,
            # 几何正则化可用的随机相机位姿数。
            N_random_pose=cfgdata.N_random_pose,
            # x 方向焦距（像素单位）。
            fx=cfgdata.fx,
            # y 方向焦距。
            fy=cfgdata.fy,
            # 原代码用于推导仿真图宽的中心 x 参数。
            cx=cfgdata.cx,
            # 原代码用于推导仿真图高的中心 y 参数。
            cy=cfgdata.cy,
        )
    else:
        # 项目没有实现其他 dataset_name，因此明确报错。
        raise NotImplementedError("No such dataset")
    # 将已完成加载/预处理的 Dataset 对象交给训练代码。
    return train_dataset


def get_test_dataset(cfgdata, is_stack=True,experiment=True):
    """【主线】根据 ``cfgdata`` 创建并返回 BOS 测试数据集。

    与 ``get_train_dataset`` 的参数来源相同，但 split 为 ``"test"``，且默认
    ``is_stack=True``，会保留每帧图像边界，便于重排成完整位移图做评估。
    """
    # 当数据集名称是 bos 时创建测试集。
    if cfgdata.dataset_name == "bos":
        # 实例化 BOSDataset，所有参数均从同一配置对象读取。
        test_dataset = BOSDataset(
            # 数据根目录。
            cfgdata.datadir,
            # 固定选择测试划分。
            "test",
            # 图像下采样倍数。
            cfgdata.downsample,
            # 默认按帧堆叠测试数据。
            is_stack=is_stack,
            # 选择实验/仿真读取路径。
            experiment=experiment,
            # 是否计算更紧包围盒。
            cal_fine_bbox=cfgdata.cal_fine_bbox,
            # 可视化/帧抽样数。
            N_vis=cfgdata.N_vis,
            # 时间缩放系数。
            time_scale=cfgdata.time_scale,
            # xyz 下界。
            scene_bbox_min=cfgdata.scene_bbox_min,
            # xyz 上界。
            scene_bbox_max=cfgdata.scene_bbox_max,
            # 随机相机数。
            N_random_pose=cfgdata.N_random_pose,
            # x 焦距。
            fx=cfgdata.fx,
            # y 焦距。
            fy=cfgdata.fy,
            # 仿真图宽参数。
            cx=cfgdata.cx,
            # 仿真图高参数。
            cy=cfgdata.cy,
        )
    else:
        # 对未实现的数据集名称报错。
        raise NotImplementedError("No such dataset")
    # 返回可由评估代码按帧索引的数据集。
    return test_dataset
