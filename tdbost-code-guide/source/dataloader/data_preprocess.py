"""把原始 BOS 背景点/偏折数组整理成 TDBOST 数据集。

【主线】原始 .npy 文件中，每一行可理解为一个背景点：
``[原始x, 原始y]`` 以及 ``[偏折/位移x, 偏折/位移y]``。本文件会把它们合并、
按帧保存，再生成记录文件路径、时间和相机位姿的 JSON 索引。
"""

# os 用于拼接路径、检查文件和创建输出目录。
import os
# json 用于写出 transforms_train_s.json 和 transforms_test_s.json。
import json
# PyTorch 负责张量拼接、时间构造和相机矩阵堆叠。
import torch
# NumPy 负责读写 .npy 文件、生成观测角度和高斯噪声。
import numpy as np
# random 用固定种子打乱帧列表，以划分互不重叠的训练/测试集。
import random  # 新增：用于彻底打乱列表
# Matplotlib 用于画出一幅偏折场样例，检查预处理结果。
import matplotlib.pyplot as plt
# Literal 仅用于类型提示，约束 dataset_type 应为 fuel 或 spray。
from typing import Literal

# 假设 pose_spherical 在你的 All_util 模块中，如果没有请自行导入
# 尝试导入「绕物体生成相机外参」的函数。
try:
    # pose_spherical(theta, phi, radius) 返回 ``[4,4]`` 相机到世界矩阵。
    from All_util.extinM import pose_spherical
# 如果该项目路径不可用，原代码只显示警告，并继续加载本模块。
except ImportError:
    # 【注意】警告后继续调用 preprocess_bost_data 仍会因 pose_spherical 未定义而失败。
    print("[Warning] Could not import pose_spherical from All_util.extinM. Please check the path.")

def preprocess_bost_data(
    dataset_type: Literal["fuel", "spray"],
    data_dir: str = "./data/traindata",
    output_dir: str = "./data/bosdata",
    angle_N: int = 9,
    time_steps: int = 20,
    noise_level: float = 0.0,
    Cx: int = 99,
    Cy: int = 100,
    test_split_ratio: float = 0.1
):
    """【主线】将多视角、多时刻 BOS .npy 数据转为训练/测试帧与 JSON 索引。

    参数说明：
    - ``dataset_type``：``"fuel"`` 或 ``"spray"``，会嵌入原始文件名；
    - ``angle_N``/``time_steps``：相机视角数和时间帧数；
    - ``noise_level``：人工高斯噪声强度相对于当帧最大偏折值的比例；
    - ``Cx``/``Cy``：原代码以 ``Cx*Cy*4`` 作为每帧点数；
    - ``test_split_ratio``：打乱后分给测试集的帧比例。

    函数通过写文件产生结果，没有显式返回值。每个输出 datatensor 形状预期为
    ``[Cx*Cy*4, 4]``，4 列依次是原始 xy 与偏折 xy。
    """
    # 打印本次处理的数据类型，upper() 只用于让日志更醒目。
    print(f"[INFO] Starting data preprocessing for dataset: {dataset_type.upper()}")
    
    # 创建输出目录
    # 所有逐帧 .npy 都写入 output_dir/train 子目录。
    train_dir = os.path.join(output_dir, "train")
    # 递归创建目录；目录已存在时不报错。
    os.makedirs(train_dir, exist_ok=True)
    
    # 预计算相机外参矩阵 (C2W)
    # 在 0°～180° 之间等间距生成 angle_N 个观测角，每个角转成 ``[4,4]`` C2W。
    C2W = torch.stack([pose_spherical(angle, 0.0, 30.0) for angle in np.linspace(0, 180, angle_N)[:]], 0)
    
    # 收集所有帧的原始背景点 xy，稍后一次性拼接。
    XYoriginal_t = []
    # 收集所有帧加噪后的偏折/deflection xy。
    XYfinal_t = []
    # 收集与每个点对齐的归一化时间。
    XYtime_t = []
    
    # 打印阶段日志，提示下面开始读取原始数据。
    print("[INFO] Reading original .npy files...")
    # 循环读取文件
    # 外层遍历相机视角，idxangle 从 0 到 angle_N-1。
    for idxangle in range(angle_N):
        # 内层遍历时间帧；原文件命名从 1ms 开始，所以起点为 1。
        for idx in range(1, time_steps + 1):
            # 为文件名保留一个与 idx 相同的中间变量。
            id_val = idx#idx#10
            
            # 构造未偏折背景点文件的完整路径。
            orig_path = os.path.join(data_dir, f"XYoriginal_{dataset_type}{id_val}ms{idxangle}.npy")
            # 构造对应偏折数据文件路径。
            deflect_path = os.path.join(data_dir, f"XYdeflection_{dataset_type}{id_val}ms{idxangle}.npy")
            
            # 两个配对文件任意一个缺失都不能构成完整样本。
            if not os.path.exists(orig_path) or not os.path.exists(deflect_path):
                # 立即报错并显示原始点路径，阻止生成不完整数据集。
                raise FileNotFoundError(f"Data file missing! Please check {orig_path}")

            # 读取原始背景点坐标，形状通常为 ``[N_points, 2]``。
            good_old = np.load(orig_path)
            # 读取对应偏折/deflection xy，形状也通常为 ``[N_points, 2]``。
            good_deta = np.load(deflect_path)
            
            # 添加高斯噪声
            # 取当帧偏折数组最大值，作为噪声尺度的参考。
            stdmax = good_deta.max()
            # 生成与 good_deta 完全同形状的零均值高斯噪声。
            noise = np.random.normal(0, stdmax * noise_level, good_deta.shape)
            
            # 把 NumPy 原始坐标转为 float32 PyTorch 张量。
            xyoriginal = torch.tensor(good_old, dtype=torch.float32)
            # 先向偏折数据加噪，再转为 float32 张量。
            xyfinal = torch.tensor(good_deta + noise, dtype=torch.float32)
            
            # 时间归一化
            # 对本帧每个偏折元素填入相同时间，从 0 递增到 (time_steps-1)/time_steps。
            xytime = (idx - 1) * torch.ones_like(xyfinal) / time_steps
            
            # 暂存本帧原始 xy。
            XYoriginal_t.append(xyoriginal)
            # 暂存本帧偏折 xy。
            XYfinal_t.append(xyfinal)
            # 暂存本帧时间张量。
            XYtime_t.append(xytime)
            
    # 按第 0 维把所有视角/时刻的原始点连成一张大表。
    xyoriginal = torch.cat(XYoriginal_t, 0)
    # 以相同顺序拼接所有偏折。
    xyfinal = torch.cat(XYfinal_t, 0)
    # 以相同顺序拼接时间，保证行与点一一对应。
    xytime = torch.cat(XYtime_t, 0)
    
    # 构建视角索引张量
    # 检查总点数能被视角数整除，否则无法为每个视角分配等长段。
    assert xytime[:, 0].numel() % angle_N == 0, "Total rays must be divisible by number of angles."
    # 计算一个视角在大表中占据多少行。
    segment_length = xytime[:, 0].numel() // angle_N
    # 依次生成 segment_length 个 0、segment_length 个 1 ……，与数据读取顺序对齐。
    angle_indices = torch.cat([torch.full((segment_length,), i) for i in range(angle_N)])
    
    # 拼接原始坐标与光线偏折量
    # 原代码把一帧点数定义为 4*Cx*Cy，等价于 ``(2Cx)*(2Cy)`` 网格。
    pixels_per_frame = Cx * Cy * 4
    # 用总行数整除每帧点数，得到可以切出的完整帧数。
    num_images = xytime.size(0) // pixels_per_frame
    
    # 明确取偏折的第 0/1 列并重新堆成 ``[N_total, 2]``。
    xydeta = torch.stack((xyfinal[:, 0], xyfinal[:, 1]), dim=-1)
    # 沿列方向拼成 ``[N_total, 4]``：[原始x, 原始y, 偏折x, 偏折y]。
    xy_original_deta = torch.cat((xyoriginal, xydeta), dim=-1)
    
    # 打印合并大张量形状，便于检查总点数和 4 个字段。
    print(f"[INFO] Total merged tensor shape: {xy_original_deta.shape}")
    # 提示下面开始逐帧保存并组装 JSON。
    print("[INFO] Saving split frames and generating transforms.json...")

    # 创建全部帧的临时 JSON 字典，后面再打乱并分成 train/test。
    transforms_dict = {
        "camera_angle_x": 0.1,  # FOV/Angle x (假设值)
        "frames": []
    }
    
    # 逐个完整帧切分大张量。
    for idx in range(num_images):
        # 计算本帧在大张量中的起始行（包含）。
        start_idx = idx * pixels_per_frame
        # 计算本帧结束行（不包含）。
        end_idx = (idx + 1) * pixels_per_frame
        
        # 切出形状约 ``[pixels_per_frame, 4]`` 的本帧数据。
        datatensor = xy_original_deta[start_idx:end_idx]
        # 本帧所有点时间相同，因此只读第一行第 0 列并转成 Python 数。
        time_val = xytime[start_idx, 0].item()
        # 查询本帧属于哪个相机视角，转为 Python 整数。
        angle_idx = int(angle_indices[start_idx])
        # 取出该视角的 ``[4,4]`` C2W 矩阵，转列表以便 JSON 序列化。
        transform_matrix = C2W[angle_idx].tolist()
        
        # 保存 NPY 数据
        # 构造相对 output_dir 的帧文件路径，也将被写进 JSON。
        file_rel_path = f"train/datatensor_{idx}.npy"
        # 将相对路径与输出根目录合成实际写入路径。
        file_abs_path = os.path.join(output_dir, file_rel_path)
        # 先把 PyTorch 张量转回 NumPy，再保存为 .npy。
        np.save(file_abs_path, datatensor.numpy())
        
        # 记录 JSON
        # 为本帧组装元数据：数据路径、视角索引、时间和 C2W。
        frame = {
            "file_path": f"./{file_rel_path}",
            "rotation": angle_idx,
            "time": time_val,
            "transform_matrix": transform_matrix
        }
        # 按生成顺序把本帧追加到总帧列表。
        transforms_dict["frames"].append(frame)

    # ==========================================
    # 核心修改：彻底打乱并严格隔离训练集与测试集
    # ==========================================
    # 复制帧列表，让下面的原地打乱不修改 transforms_dict 内部列表对象。
    all_frames = transforms_dict['frames'].copy()
    
    # 设置随机种子以保证实验可复现（可选）
    # 固定 Python random 的伪随机序列，使每次得到相同的 train/test 划分。
    random.seed(42) 
    # 物理打乱整个列表，保证帧的顺序是完全随机的
    # 原地重排 all_frames 元素顺序。
    random.shuffle(all_frames)
    
    # 记录打乱后的总帧数。
    total_frames = len(all_frames)
    # 按比例计算测试帧数，int 直接截去小数部分。
    num_test_frames = int(total_frames * test_split_ratio)
    
    # 列表切片：前 10% 做测试集，剩下的 90% 做训练集（互不重叠）
    # 测试集复用相同相机角字段，帧列表取打乱后的前 num_test_frames 项。
    test_transforms = {
        "camera_angle_x": transforms_dict["camera_angle_x"], 
        "frames": all_frames[:num_test_frames]
    }
    # 训练集从 num_test_frames 之后取到末尾，因此与测试切片不重叠。
    train_transforms = {
        "camera_angle_x": transforms_dict["camera_angle_x"], 
        "frames": all_frames[num_test_frames:]
    }

    # 保存 JSON 文件
    # 生成训练集 JSON 的完整路径。
    train_json_path = os.path.join(output_dir, 'transforms_train_s.json')
    # 生成测试集 JSON 的完整路径。
    test_json_path = os.path.join(output_dir, 'transforms_test_s.json')
    
    # 以文本写入模式打开训练 JSON。
    with open(train_json_path, 'w') as f:
        # indent=4 使 JSON 换行缩进，便于人工查看。
        json.dump(train_transforms, f, indent=4)
    # 以写入模式打开测试 JSON。
    with open(test_json_path, 'w') as f:
        # 将测试帧索引序列化到磁盘。
        json.dump(test_transforms, f, indent=4)
        
    # 打印数据类型与完成提示。
    print(f"[INFO] Dataset '{dataset_type}' generation complete!")
    # 打印训练帧数和 JSON 路径。
    print(f"       Saved {len(train_transforms['frames'])} train frames to {train_json_path}")
    # 打印测试帧数和 JSON 路径。
    print(f"       Saved {len(test_transforms['frames'])} test frames to {test_json_path}")

    # 绘制测试可视化
    # 立即读回测试划分的第一帧，生成检查图。
    verify_data(test_json_path, Cx, Cy)


def verify_data(json_path: str, Cx: int, Cy: int):
    """读取生成 JSON 的第一帧，将 y 方向偏折画成图以验证。

    ``json_path`` 指向 transforms JSON；``Cx``/``Cy`` 用于将一维点表重排为
    ``[2*Cx, 2*Cy]``。函数会写出 ``sample_deflection.png``，没有显式返回值。
    """
    # 以只读文本模式打开 transforms JSON。
    with open(json_path, 'r') as f:
        # 解析 JSON 为 Python 字典。
        meta = json.load(f)
    
    # 如果测试帧列表为空，就无样本可画，直接结束。
    if not meta['frames']:
        return
        
    # 获取 JSON 所在目录作为根目录
    # 帧内 file_path 是相对路径，因此先取 JSON 的父目录。
    base_dir = os.path.dirname(json_path)
    # 取出第一帧相对路径
    # lstrip('./') 去掉开头用于表示当前目录的点和斜杠字符。
    rel_path = meta['frames'][0]['file_path'].lstrip('./')
    # 把 JSON 目录与相对帧路径合成完整路径。
    point_path = os.path.join(base_dir, rel_path)
    
    # 读取 ``[N,4]`` 数组并只保留后两列偏折 xy，得到 ``[N,2]``。
    point = np.load(point_path)[:, 2:]
    # 打印偏折平均值，快速发现全零或数值爆炸等异常。
    print(f"[VERIFY] Loaded random validation frame 0, Deflection Mean: {point.mean():.4f}")
    
    # 创建 6×5 英寸的绘图画布。
    plt.figure(figsize=(6, 5))
    # 取偏折最后一列（y 分量），重排为 ``[2Cx,2Cy]`` 并用 viridis 色表显示。
    plt.imshow(point[..., -1].reshape(2 * Cx, 2 * Cy), cmap='viridis')
    # 在图旁加上颜色与 y 偏折数值的对应标尺。
    plt.colorbar(label='Deflection Y')
    # 设置图标题。
    plt.title("Sample Deflection Visualization")
    # 自动调整边距，避免标题或色标被裁切。
    plt.tight_layout()
    # 将验证图写到 JSON 同级目录。
    plt.savefig(os.path.join(base_dir, "sample_deflection.png"))
    # 关闭当前画布，释放内存。
    plt.close()
    # 打印图像的完整保存路径。
    print(f"[VERIFY] Saved sample visualization to {os.path.join(base_dir, 'sample_deflection.png')}")


# 只有当本文件被直接执行，而不是被 import 时，才运行下面的默认预处理。
if __name__ == "__main__":
    # 用仓库中给定的默认参数生成 fuel BOS 数据集。
    preprocess_bost_data(
        # 文件名中使用 fuel 类型。
        dataset_type="fuel",
        # 原始 .npy 输入目录。
        data_dir="./data/traindata",          
        # 预处理结果输出目录。
        output_dir="./data/bosdata_fuel",
        # 使用 9 个观测角。
        angle_N=9,
        # 每个视角使用 20 个时间步。
        time_steps=20,
        # 不额外添加人工噪声。
        noise_level=0.0,
        # 水平半尺寸参数。
        Cx=99,
        # 竖直半尺寸参数。
        Cy=100
    )
