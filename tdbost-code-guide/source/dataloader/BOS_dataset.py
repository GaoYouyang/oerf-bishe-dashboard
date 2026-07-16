"""TDBOST 的 BOS 数据集：读取背景点位移，构造相机光线并向训练器提供样本。

【主线】对每个背景点，数据集最终需要给出四类东西：
1. ``rays``：三维光线起点 xyz + 方向 xyz，通常每行 6 个数；
2. ``rgbs``：这个名称继承自 NeRF，但在本 BOS 项目中实际是 x/y 位移真值；
3. ``time``：该光线对应的流场时刻；
4. ``w2c``：世界坐标到相机坐标的旋转，用于解释偏折方向。

``read_meta`` 读仿真数据，``read_expdata`` 读真实实验相机数据。
"""

# json 用于读取每帧文件路径、时间和相机位姿。
import json
# os 用于拼接数据根目录与相对文件路径。
import os

# NumPy 用于读取 .npy/.dat，计算三角函数和处理相机矩阵。
import numpy as np
# PyTorch 用于张量运算、光线堆叠以及数据集输出。
import torch
# Pillow 图像类为原 NeRF 数据加载框架保留，当前 BOS .npy 主路径未使用。
from PIL import Image
# Dataset 是 PyTorch 数据集基类，使本类可被 DataLoader 按索引取样。
from torch.utils.data import Dataset
# torchvision.transforms 用于定义图像转张量操作，虽然 .npy 主路径直接用 torch.from_numpy。
from torchvision import transforms as T
# tqdm 在读取多帧数据时显示进度条。
from tqdm import tqdm

# 从同包光线工具中导入像素造方向、坐标变换和 PFM 读取函数。
from .ray_utils import get_ray_directions_blender, get_rays, read_pfm

# 定义一个 ``[4,4]`` 单位矩阵；原项目保留 Blender 到 OpenCV 转换接口，此处实际不改变坐标。
blender2opencv = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def trans_t(t):
    """创建沿 z 轴平移 ``t`` 的 ``[4,4]`` 齐次变换矩阵。"""
    # 前 3×3 保持单位旋转，只在第 3 行第 4 列写入 z 平移 t。
    return torch.Tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
    ).float()


def rot_phi(phi):
    """创建绕 x 轴旋转 ``phi`` 弧度的 ``[4,4]`` 齐次矩阵。"""
    # 使用正弦/余弦填充 yz 子矩阵，x 轴本身保持不变。
    return torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def rot_theta(th):
    """创建绕 y 轴旋转 ``th`` 弧度的 ``[4,4]`` 齐次矩阵。"""
    # 使用正弦/余弦填充 xz 子矩阵，y 轴本身保持不变。
    return torch.Tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def pose_spherical(theta, phi, radius):
    """根据球面角度（度）和半径生成 ``[4,4]`` 相机到世界矩阵。

    【物理直觉】先把相机推离原点 ``radius``，再绕 x/y 轴旋转，就得到绕流场观测的相机位姿。
    """
    # 先沿 z 轴平移到指定观测半径。
    c2w = trans_t(radius)
    # 将 phi 从度转为弧度，左乘 x 轴旋转。
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    # 将 theta 从度转为弧度，左乘 y 轴旋转。
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    # 用固定矩阵交换/翻转坐标轴，再乘上项目保留的 Blender→OpenCV 单位变换。
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )
        @ c2w
        @ blender2opencv
    )
    # 返回组合后的相机到世界矩阵。
    return c2w


class BOSDataset(Dataset):
    """【主线】将 BOS 背景点数据整理成 PyTorch 可按索引访问的光线数据集。

    仿真模式调用 ``read_meta``；实验模式调用 ``read_expdata``。
    ``is_stack=False`` 用于训练，所有帧展成光线表；``is_stack=True`` 通常用于测试，保留帧维。
    """

    def __init__(
        self,
        datadir,
        split="train",
        downsample=1.0,
        is_stack=False,
        experiment=False,
        cal_fine_bbox=False,
        N_vis=-1,
        time_scale=1.0,
        scene_bbox_min=[-1.0, -1.0, -1.0],
        scene_bbox_max=[1.0, 1.0, 1.0],
        N_random_pose=1000,
        fx=1500.0,
        fy=1500.0,
        cx=99,
        cy=100,
    ):
        """保存数据/相机配置，选择实验或仿真读取路径，并立即加载数据。"""
        # 保存数据根目录，之后所有相对路径都与它拼接。
        self.root_dir = datadir
        # 保存 train/test 划分名称，用于选择 transforms JSON。
        self.split = split
        # 保存下采样倍数。
        self.downsample = downsample
        ####需要修改
        # 打印实验模式布尔值，便于确认当前走哪条数据路径。
        print( experiment)
        # experiment=True 表示使用真实相机实验数据。
        if experiment:
            # 打印实验模式日志。
            print('experiment:' ,experiment)
            # 真实实验原始尺寸固定为 305×300，按 downsample 缩小并取整。
            self.img_wh = (int(305 / downsample), int(300 / downsample))
        else:
            # 【注意】原代码使用位反 ``~experiment`` 打印，对 Python bool 会得到整数 -1/-2。
            print('simulation:', ~experiment)
            # 仿真图像宽高分别以 2*cx、2*cy 计算，再除以下采样倍数。
            self.img_wh = (int(cx*2 / downsample), int(cy*2 / downsample))
            # 图像缩小时，x 焦距同比例缩小。
            self.fx = fx / downsample
            # y 焦距同比例缩小。
            self.fy = fy / downsample
        # 记录是否保留帧维度，这会决定后面用 stack 还是 cat。
        self.is_stack = is_stack
        # 记录评估要抽取的帧数/间隔配置。
        self.N_vis = N_vis  # evaluate images for every N_vis images

        # 保存时间坐标缩放系数。
        self.time_scale = time_scale
        # 包围盒额外扩张比例；默认 1 表示不扩展。
        self.world_bound_scale = 1

        

        # 创建图像转 PyTorch 张量的 transform 对象。
        self.define_transforms()  # transform to torch.Tensor

        # 把 xyz 下界/上界堆成 ``[2,3]`` 三维场包围盒。
        self.scene_bbox = torch.stack((torch.tensor(scene_bbox_min), torch.tensor(scene_bbox_max)), dim=0)
        # 保存 NumPy 形式的 4×4 单位坐标转换矩阵。
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        
        # 根据 experiment 选择真实实验加载器或仿真加载器。
        if experiment:
            # 读取 transforms_train/test.json 与真实多相机标定。
            self.read_expdata()
        else:
            # 读取 transforms_train/test_s.json 仿真数据。
            self.read_meta()  # Read meta data

        # Calculate a more fine bbox based on near and far values of each ray.
        # 如果配置要求紧包围盒，用已加载光线 near/far 重新估计。
        if cal_fine_bbox:
            # 计算所有光线近远端点的 xyz 总最小/最大。
            xyz_min, xyz_max = self.compute_bbox()
            # 用新下界/上界替换初始配置包围盒。
            self.scene_bbox = torch.stack((xyz_min, xyz_max), dim=0)

        #self.define_proj_mat()

        # BOS 数据不使用白色背景混合。
        self.white_bg = False
        # 光线默认不是 NDC 坐标。
        self.ndc_ray = False
        # 本数据集不提供直接深度真值。
        self.depth_data = False
        #self.read_data()
        # self.N_random_pose = N_random_pose
        # self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        # self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        # # Generate N_random_pose random poses, which we could render depths from these poses and apply depth smooth loss to the rendered depth.
        # # if split == "train":
        # #     self.init_random_pose()

    def init_random_pose(self):
        """生成随机相机位姿、时间与光线，用于 RegNeRF 风格的几何正则化。

        输出保存在对象属性：``random_rays`` 预期为
        ``[N_random_pose, W, H, 6]``，``random_times`` 为 ``[N_random_pose]``。
        【注意】当前 __init__ 中设定 self.N_random_pose 的原代码已被注释，本注释版不修改这一现状。
        """
        # Randomly sample N_random_pose radius, phi, theta and times.
        # 从均值 4、标准差 0.1 的正态分布采样每个相机半径。
        radius = np.random.randn(self.N_random_pose) * 0.1 + 4
        # 在 [-180°,180°) 均匀采样 phi。
        phi = np.random.rand(self.N_random_pose) * 360 - 180
        # 在 [-180°,180°) 均匀采样 theta。
        theta = np.random.rand(self.N_random_pose) * 360 - 180
        # 在 [-time_scale,time_scale) 均匀采样时间。
        random_times = self.time_scale * (torch.rand(self.N_random_pose) * 2.0 - 1.0)
        # 将随机时间保存为 Dataset 属性。
        self.random_times = random_times

        # Generate rays from random radius, phi, theta and times.
        # 创建列表收集每个随机相机的所有光线。
        self.random_rays = []
        # 逐个随机相机生成 C2W 和世界光线。
        for i in range(self.N_random_pose):
            # 用当前 theta/phi/radius 构造 ``[4,4]`` 相机到世界矩阵。
            random_poses = pose_spherical(theta[i], phi[i], radius[i])
            # 将已有像素方向变换到当前随机相机，得到 ``[N_pixels,3]`` 起点/方向。
            rays_o, rays_d = get_rays(self.directions, random_poses)
            # 沿列拼成 ``[N_pixels,6]`` 的 [origin xyz, direction xyz]，加入列表。
            self.random_rays += [torch.cat([rays_o, rays_d], 1)]
        # 取图像宽高参数，下方保持原代码 w,h 顺序。
        w, h = self.img_wh
        # 堆叠所有相机，再重排为 ``[-1,w,h,6]``，-1 由 PyTorch 自动推导。
        self.random_rays = torch.stack(self.random_rays, 0).reshape(
            -1, w,h, 6
        )

    def compute_bbox(self):
        """从所有光线的 near/far 端点估计三维场轴对齐包围盒。

        返回 ``xyz_min``/``xyz_max``，均为 ``[3]`` 张量。【物理直觉】
        如果只在所有相机视锥真正穿过的空间建网格，可减少空体素计算。
        """
        # 打印开始日志。
        print("compute_bbox_by_cam_frustrm: start")
        # 用正无穷初始化 xyz 下界，任何有限点都能更新它。
        xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
        # 将 xyz 上界初始化为负无穷。
        xyz_max = -xyz_min
        # 从 ``all_rays`` 前 3 列取所有光线起点。
        rays_o = self.all_rays[:, 0:3]
        # 从 3:6 列取所有光线方向。
        viewdirs = self.all_rays[:, 3:6]
        # 用 ``o+d*near`` 和 ``o+d*far`` 得到每条光线的两个视锥端点。
        pts_nf = torch.stack(
            [rays_o + viewdirs * self.near, rays_o + viewdirs * self.far]
        )
        # 对端点类型维和光线维一起取最小，更新 xyz 下界。
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1)))
        # 对所有端点取 xyz 最大值。
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1)))
        # 打印估计下界。
        print("compute_bbox_by_cam_frustrm: xyz_min", xyz_min)
        # 打印估计上界。
        print("compute_bbox_by_cam_frustrm: xyz_max", xyz_max)
        # 打印主要估计完成日志。
        print("compute_bbox_by_cam_frustrm: finish")
        # 根据 world_bound_scale 计算包围盒每侧要额外扩展的距离。
        xyz_shift = (xyz_max - xyz_min) * (self.world_bound_scale - 1) / 2
        # 下界向外减去扩展量。
        xyz_min -= xyz_shift
        # 上界向外加上扩展量。
        xyz_max += xyz_shift
        # 返回最终 xyz 下界和上界。
        return xyz_min, xyz_max

    def read_depth(self, filename):
        """读取一张 PFM 深度图，返回 float32 NumPy 数组 ``[H,W]``。"""
        # read_pfm 返回(数据,尺度)，此处只取数据并强制转 float32。
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        # 返回深度图。
        return depth
    
    def load_processed_data(self):
        """从 ``processed_data.pt`` 快速恢复预先合并好的光线、时间、位移和 W2C。"""
        # 用 torch.load 读取 Python 字典/张量包。
        data = torch.load(self.processed_data_path)
        # 恢复光线表，通常末维为 6。
        self.all_rays = data['rays']
        # 恢复每条光线的时间。
        self.all_times = data['times']
        # 恢复名为 rgbs、实为 BOS xy 位移的真值。
        self.all_rgbs = data['rgbs']
        # 恢复每条光线对应的世界到相机旋转。
        self.all_W2C = data['W2C']

    def read_data(self):
        """读取 20 个 DNS 时刻的网格、密度、速度、压力和组分原始 .dat 文件。

        结果展平后保存为 ``self.Xs`` 等一维张量。
        【暂时不用深究】该函数在当前 __init__ 中没有被调用，且路径为作者机器的绝对路径。
        """
        # 为坐标、时间和 7 种物理量分别创建逐时刻收集列表。
        Xs,Ys,Zs,Ts,RHOs,Us,Vs,Ws,Ps,YO2s,YCH4s=[],[],[],[],[],[],[],[],[],[],[]
        # 遍历 0～19 共 20 个 DNS 时刻。
        for index in range(20): 
            # 固定网格坐标文件目录。
            gridPath='/home/hyz/Project/Data/hyz/CH4O2_HIT_DNS/grid'
            # 固定物理场数据目录。
            dataPath='/home/hyz/Project/Data/hyz/CH4O2_HIT_DNS/data'
            # 以小端 float32 二进制格式读取所有 X 坐标。
            X = np.fromfile(f'{gridPath}/X_m.dat', dtype='<f4')
            # 读取 Y 坐标。
            Y = np.fromfile(f'{gridPath}/Y_m.dat', dtype='<f4')
            # 读取 Z 坐标。
            Z = np.fromfile(f'{gridPath}/Z_m.dat', dtype='<f4')
            # Assume X, Y, Z are 1D arrays defining the grid
            # 分别取 xyz 不重复坐标，恢复三个一维网格轴。
            x1d, y1d, z1d = np.unique(X), np.unique(Y), np.unique(Z)
            # 用 ij 索引生成完整三维网格 X/Y/Z，三者形状相同。
            X, Y, Z = np.meshgrid(x1d, y1d, z1d, indexing='ij')
            # 构造当前时刻密度 RHO 文件名。
            filename = f'{dataPath}/RHO_kgm-3_id0{index:02}.dat'  # example for one file
            # 读密度并恢复成与 xyz 网格相同的三维形状。
            RHO = np.fromfile(filename, dtype='<f4').reshape(X.shape)
            # 构造 x 速度 U 文件名。
            filename = f'{dataPath}/UX_ms-1_id0{index:02}.dat'  # example for one file
            # 读取并重排 U。
            U = np.fromfile(filename, dtype='<f4').reshape(X.shape)
            # 构造 y 速度 V 文件名。
            filename = f'{dataPath}/UY_ms-1_id0{index:02}.dat'  # example for one file
            # 读取并重排 V。
            V = np.fromfile(filename, dtype='<f4').reshape(X.shape)
            # 构造 z 速度 W 文件名。
            filename = f'{dataPath}/UZ_ms-1_id0{index:02}.dat'  # example for one file
            # 读取并重排 W。
            W = np.fromfile(filename, dtype='<f4').reshape(X.shape)
            # 构造压力 P 文件名。
            filename = f'{dataPath}/P_Pa_id0{index:02}.dat'  # example for one file
            # 读取并重排压力。
            P = np.fromfile(filename, dtype='<f4').reshape(X.shape)
            # 构造氧气质量分数 YO2 文件名。
            filename = f'{dataPath}/YO2_id0{index:02}.dat'  # example for one file
            # 读取并重排 YO2。
            YO2 = np.fromfile(filename, dtype='<f4').reshape(X.shape)
            # 构造甲烷质量分数 YCH4 文件名。
            filename = f'{dataPath}/YCH4_id0{index:02}.dat'  # example for one file
            # 读取并重排 YCH4。
            YCH4 = np.fromfile(filename, dtype='<f4').reshape(X.shape)
            # 把三个 NumPy 网格转为 PyTorch 张量。
            X, Y ,Z = torch.Tensor(X),torch.Tensor(Y),torch.Tensor(Z)
            # 生成与 X 同形状的时间场，当前值为 index/10000。
            T =torch.ones_like(X)*index/10000
            # 将 X 展平、乘原代码尺度 0.8*10000，加入列表。
            Xs.append(X.reshape(-1)*0.8*10000)
            # 展平并缩放 Y。
            Ys.append(Y.reshape(-1)*0.8*10000)
            # 展平并缩放 Z。
            Zs.append(Z.reshape(-1)*0.8*10000)
            # 展平时间 T。
            Ts.append(T.reshape(-1))
            # 把密度转张量并展平。
            RHOs.append(torch.Tensor(RHO).reshape(-1))
            # 展平 U。
            Us.append(torch.Tensor(U).reshape(-1))
            # 展平 V。
            Vs.append(torch.Tensor(V).reshape(-1))
            # 展平 W。
            Ws.append(torch.Tensor(W).reshape(-1))
            # 展平压力 P。
            Ps.append(torch.Tensor(P).reshape(-1))
            # 展平 YO2。
            YO2s.append(torch.Tensor(YO2).reshape(-1))
            # 展平 YCH4。
            YCH4s.append(torch.Tensor(YCH4).reshape(-1))
        
        # 按时刻顺序连接所有 X 网格点。
        self.Xs=torch.cat(Xs,0)
        # 连接所有 Y。
        self.Ys=torch.cat(Ys,0)
        # 连接所有 Z。
        self.Zs=torch.cat(Zs,0)
        # 连接所有 T。
        self.Ts=torch.cat(Ts,0)
        # 连接所有密度。
        self.RHOs=torch.cat(RHOs,0)
        # 连接所有 U。
        self.Us=torch.cat(Us,0)
        # 连接所有 V。
        self.Vs=torch.cat(Vs,0)
        # 连接所有 W。
        self.Ws=torch.cat(Ws,0)
        # 连接所有 P。
        self.Ps=torch.cat(Ps,0)
        # 连接所有 YO2。
        self.YO2s=torch.cat(YO2s,0)
        # 连接所有 YCH4。
        self.YCH4s=torch.cat(YCH4s,0)
        # 打印时间张量最大/最小值，检查时间尺度。
        print(self.Ts.max(),self.Ts.min())

    def read_meta(self):
        """【主线】读取仿真 BOS JSON/.npy，构造光线、xy 位移真值、时间和 W2C。

        每帧 .npy 预期为 ``[N_pixels,4]``：前 2 列是背景点 xy，后 2 列是 BOS 位移 xy。
        非堆叠模式最终生成 ``all_rays [N_frames*N_pixels,6]``；堆叠模式保留帧维。
        """
        # 打开当前 split 对应的 transforms_train_s.json 或 transforms_test_s.json。
        with open(os.path.join(self.root_dir, f"transforms_{self.split}_s.json")) as f:
            # 解析 JSON；``frames`` 列表记录每帧路径、时间和相机位姿。
            self.meta = json.load(f)

        # 取得图像宽高，保持原代码 w,h 顺序。
        w, h = self.img_wh
        ######修改焦距
        # x 焦距使用 __init__ 中按下采样调整后的 fx。
        self.focalx = self.fx#800 
        # y 焦距使用调整后的 fy。
        self.focaly = self.fy#800 # modify focal length to match size self.img_wh
       
     
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        # 构造 ``[3,3]`` 相机内参 K，主点位于 w/2、h/2。
        self.intrinsics = torch.tensor(
            [[self.focalx, 0, w / 2], [0, self.focaly, h / 2], [0, 0, 1]]
        ).float()
        #self.img_wh= h,w 
        # 以下列表逐帧收集数据，加载完后再 cat/stack。
        # 保存每帧 .npy 绝对路径。
        self.image_paths = []
        # 保存每帧 C2W 位姿。
        self.poses = []
        # 保存每帧 ``[N_pixels,6]`` 光线。
        self.all_rays = []
        # 保存每帧 ``[N_pixels,1]`` 时间。
        self.all_times = []
        # 保存每帧 ``[N_pixels,2]`` BOS xy 位移；``rgbs`` 是继承自 NeRF 的名称。
        self.all_rgbs = []
        # 为深度真值保留空列表，当前主路径不填充。
        self.all_depth = []
        # 保存每条光线对应的 ``[3,3]`` W2C 旋转。
        self.all_W2C = []#和时间一起存在
        # N_vis<0 时间隔为 1；否则用总帧数//N_vis 估算抽帧间隔。
        img_eval_interval = (
            1 if self.N_vis < 0 else len(self.meta["frames"]) // self.N_vis
        )
        # 生成实际要加载的帧索引。
        idxs = list(range(0, len(self.meta["frames"]), img_eval_interval))

        # 取第一帧，用它的背景点坐标构造所有帧共用的相机方向。
        frame = self.meta["frames"][0]
        # 拼出第一帧 .npy 绝对路径。
        image_path = os.path.join(self.root_dir,f"{frame['file_path']}")
        # 读取前两列像素 xy，转为 ``[N_pixels,2]`` float32 张量。
        imgcord=torch.from_numpy(np.load(image_path)[:,:2]).float()
        # 保存背景点坐标。
        self.point= imgcord
        # 用针孔相机公式生成 ``[(x-cx)/fx,-(y-cy)/fy,-1]`` 方向。
        self.directions=torch.stack(((imgcord[:,0]-w/2)/self.focalx,-(imgcord[:,1]-h/2)/self.focaly,-torch.ones(imgcord.shape[0])),-1)
        # 构造可选预处理缓存路径。
        self.processed_data_path = os.path.join(self.root_dir, "processed_data.pt")
        # 缓存存在且是训练集时走快速路径。【注意】``&`` 为原代码按位与。
        if (os.path.exists(self.processed_data_path))&(self.split=='train'):
            # 打印缓存加载日志。
            print("Loading processed data...")
            # 从 .pt 恢复已合并好的张量。
            self.load_processed_data()
        else:
            # 打印当前 transforms 文件名。
            print( f"transforms_{self.split}_s.json")
            # 带进度条地逐帧加载。
            for i in tqdm(
                idxs, desc=f"Loading data {self.split} ({len(idxs)})"
            ):  # img_list:#
                # 取第 i 帧 JSON 元数据。
                frame = self.meta["frames"][i]
                # 把 4×4 transform_matrix 列表转为 NumPy 矩阵。
                pose = np.array(frame["transform_matrix"])# @ self.blender2opencv
                # 把位姿转成 float32 PyTorch C2W 张量。
                c2w = torch.FloatTensor(pose)
                # 收集当前 C2W。
                self.poses += [c2w]
                
                # 拼出当前帧 .npy 绝对路径。
                image_path = os.path.join(self.root_dir,f"{frame['file_path']}")
                # 记录路径便于回溯。
                self.image_paths += [image_path]
                # (h*w, 2)偏移量
                # 读取后两列 BOS xy 位移，得 ``[N_pixels,2]`` float32 张量。
                img= torch.from_numpy(np.load(image_path)[:,2:]).float()
                
                # img = img[:, :3] * img[:, -1:] + (
                #     1 - img[:, -1:]
                # )  # blend A to RGB, white background
                # 原代码将人工噪声等级固定为 0。
                level = 0.0
                # 用当前位移绝对值最大值作为噪声尺度参考。
                stdmax = torch.max(abs(img)) 
                # 生成与 img 同形状高斯噪声；当前标准差为 0，所以噪声全为 0。
                noise = torch.normal(
                    mean=0.0,
                    std=stdmax * level,
                    size=img.shape,
                )
                # 把加噪位移收集到 all_rgbs；当前与 img 数值相同。
                self.all_rgbs += [img+noise]
                #print(c2w.shape)
                #ocxyz=torch.zeros(xp.shape[0],yp.shape[0],3).to(torch.float64)
                # 将共用相机方向经当前 C2W 转为世界光线，两者均为 ``[N_pixels,3]``。
                rays_o, rays_d = get_rays(self.directions, c2w)  # Get rays, both (h*w, 3).
                # 拼成 ``[N_pixels,6]`` [origin xyz,direction xyz] 并收集。
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
                # 优先取 JSON time；若缺失则按帧号归一化，然后扩展至每条光线。
                cur_time = torch.tensor(
                    frame["time"]
                    if "time" in frame
                    else float(i) / (len(self.meta["frames"]) - 1)
                ).expand(rays_o.shape[0], 1)
                # 收集 ``[N_pixels,1]`` 时间。
                self.all_times += [cur_time]
                # 对 C2W 旋转求逆，得 ``[3,3]`` W2C 旋转。
                w2c=torch.inverse(c2w[:3,:3])
                # 将同一 W2C 复制到本帧每个像素，得 ``[N_pixels,3,3]``。
                self.all_W2C += [w2c.unsqueeze(0).repeat(self.directions.shape[0],1,1)]
            # 将所有 C2W 堆成 ``[N_frames,4,4]``。
            self.poses = torch.stack(self.poses)
            #  self.is_stack stacks all images into a big chunk, with shape (N, H, W, 3).
            #  Otherwise, all images are kept as a set of rays with shape (N_s, 3), where N_s = H * W * N
            # 非堆叠模式将帧维与像素维合并，用于训练时按光线抽样。
            if not self.is_stack:
                # 连接成 ``[N_frames*N_pixels,6]`` 光线表。
                self.all_rays = torch.cat(
                    self.all_rays, 0
                )  # (len(self.meta['frames])*h*w, 3)
                # 连接成 ``[N_frames*N_pixels,2]`` 位移真值表。
                self.all_rgbs = torch.cat(
                    self.all_rgbs, 0
                )  # (len(self.meta['frames])*h*w, 3)
                # 连接时间，形状约 ``[N_frames*N_pixels,1]``。
                self.all_times = torch.cat(self.all_times, 0)
                # 连接 W2C，形状约 ``[N_frames*N_pixels,3,3]``。
                self.all_W2C = torch.cat(self.all_W2C, 0)
                # Save all tensors in a file for later use
                ###################################################################################################################################
                # if self.split=='train':
                #     torch.save({
                #         "rays": self.all_rays,
                #         "times": self.all_times,
                #         "rgbs": self.all_rgbs,
                #         "W2C": self.all_W2C
                #     }, os.path.join(self.root_dir, "processed_data.pt")
                #     )
            else:
                # 堆叠模式保留帧维，光线为 ``[N_frames,N_pixels,6]``。
                self.all_rays = torch.stack(
                    self.all_rays, 0
                )  # (len(self.meta['frames]),h*w, 3)h
                # 堆叠位移后重排为 ``[-1,w,h,2]``，-1 由 PyTorch 推导帧数。
                self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(
                    -1, w,h, 2
                )  # (len(self.meta['frames]),h,w,3)
                # 堆叠时间并保留帧维。
                self.all_times = torch.stack(self.all_times, 0)
                # 堆叠 W2C 并保留帧维。
                self.all_W2C = torch.stack(self.all_W2C, 0)
        #self.all_times = 1/self.time_scale * self.all_times * 2.0 - 1.0
        #print('all_rgbs,all_c2w:', self.all_rgbs.shape,self.all_W2C.shape)
    def undistort_points_lm(self,x_d, y_d, k1, k2, max_iter=20, tol=1e-8):
        """用带阻尼的迭代法从径向畸变坐标估计无畸变归一化坐标。

        ``x_d/y_d`` 是同形状畸变点，``k1/k2`` 是径向畸变系数。返回同形状 ``x_u/y_u``。
        【暂时不用深究】可把它理解为反复猜测「畸变前的点」，直到再畸变后足够接近观测点。
        """
        # 先用畸变 x 作为无畸变 x 的初始猜测；clone 避免原地改写输入。
        x_u = x_d.clone()
        # 同样初始化 y 猜测。
        y_u = y_d.clone()
        # 设置很小的 Levenberg–Marquardt 阻尼，减少雅可比近奇异时不稳定。
        damping = 1e-12  # L-M 参数
        
        # 最多迭代 max_iter 次；若误差提前小于 tol 会 break。
        for _ in range(max_iter):
            # 计算每个归一化点到光轴的平方半径 r²。
            r2 = x_u ** 2 + y_u ** 2
            # 两阶径向畸变因子：1+k1*r²+k2*r⁴。
            radial = 1 + k1 * r2 + k2 * r2**2

            # 当前猜测经畸变后的 x 减观测 x，得到 x 残差。
            fx = x_u * radial - x_d
            # 计算 y 残差。
            fy = y_u * radial - y_d

            # 对径向因子分别求 x 偏导。
            dradial_dx = 2 * k1 * x_u + 4 * k2 * x_u * r2
            # 对径向因子求 y 偏导。
            dradial_dy = 2 * k1 * y_u + 4 * k2 * y_u * r2

            # 构造残差对坐标的 2×2 雅可比矩阵四个元素。
            J11 = radial + x_u * dradial_dx
            # x 残差对 y 的偏导。
            J12 = x_u * dradial_dy
            # y 残差对 x 的偏导。
            J21 = y_u * dradial_dx
            # y 残差对 y 的偏导。
            J22 = radial + y_u * dradial_dy

            # 添加阻尼项，确保矩阵非奇异
            # 在雅可比主对角第一项加阻尼。
            J11 += damping
            # 在第二个主对角元素加阻尼。
            J22 += damping

            # 求解增量
            # 计算 2×2 雅可比行列式；1e-12 进一步避免除零。
            det = J11 * J22 - J12 * J21 + 1e-12
            # 按 2×2 矩阵求逆公式计算逆矩阵 (1,1)。
            inv_J11 =  J22 / det
            # 计算逆矩阵 (1,2)。
            inv_J12 = -J12 / det
            # 计算逆矩阵 (2,1)。
            inv_J21 = -J21 / det
            # 计算逆矩阵 (2,2)。
            inv_J22 =  J11 / det

            # 逆雅可比乘残差，得 x 更新增量。
            delta_x = inv_J11 * fx + inv_J12 * fy
            # 计算 y 更新增量。
            delta_y = inv_J21 * fx + inv_J22 * fy

            # 以 0.5 步长沿负增量更新 x，避免一步跳得过远。
            x_u = x_u - 0.5*delta_x
            # 同样更新 y。
            y_u = y_u - 0.5*delta_y

            # 计算所有点中最大的二维更新步长，作为收敛判断。
            err = torch.max(torch.sqrt(delta_x ** 2 + delta_y ** 2))
            #print(f"[{_ + 1}] max delta: {err.item():.3e}")
            # 最大更新已小于容差时，提前结束迭代。
            if err < tol:
                break

        # 返回最终无畸变归一化 x/y 坐标。
        return x_u, y_u

    def read_expdata(self):
        """【主线】读取真实多相机 BOS 数据，用标定内/外参为每帧构造光线。

        帧 .npy 仍预期为 ``[N_pixels,4]``；JSON 还需提供 ``rotation``，指定 0～8 号相机。
        标定文件每台相机提供 Intrinsics、Rotation、TranslationVector。
        【注意】标定 JSON 使用作者机器的绝对路径；本注释版仅解释，不改路径。
        """
        # 打开 transforms_train.json 或 transforms_test.json（实验版文件名没有 _s）。
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json")) as f:
            # 解析帧元数据。
            self.meta = json.load(f)

        # 取实验图像宽高。
        w, h = self.img_wh
        
        ######修改焦距
        
        # 拼出第一帧数据路径，用于读取背景点像素坐标。
        point_path = os.path.join(self.root_dir, self.meta['frames'][0]['file_path'])
     
        # 读取第一帧前两列 xy，保存为张量 ``[N_pixels,2]``。
        self.point=torch.from_numpy(np.load(point_path)[:,:2]) 
        # ray directions for all pixels, same for all images (same H, W, focal)
        # 以只读模式打开真实相机标定数据。
        with open('/home/hyz/Project/Data/hyz/BOStimedata/cameraData0.json', 'r') as file:
            # 解析 9 台相机的内参、旋转和平移。
            camera_data = json.load(file) 
        # 取 0 号相机作为初始占位；后面会按每帧 rotation 重新选择。
        selected_camera_data = camera_data[0]  # 例如，选择第一个相机
        # 创建列表收集 9 台相机到参考点 [25,15,0] 的距离。
        distances=[]
        # 【暂时不用深究】原代码立即再赋一次空列表，结果不变。
        distances=[]
        # 创建列表收集每台相机的 ``[3,3]`` 内参 K。
        Ks=[]
        # 绘制相机位置和指向
        # 遍历固定 0～8 号共 9 台相机。
        for i in [0,1,2,3,4,5,6,7,8]:
            # 取当前相机标定字典。
            data=camera_data[i]
            # 将 Rotation 列表转为 NumPy ``[3,3]`` 矩阵。
            rotation_matrix = np.array(data['Rotation'])
            # 读取 3 分量平移向量。
            translation_vector = data['TranslationVector']  # 平移向量
            # 旋转矩阵转置作为 C2W 旋转。
            C2W = rotation_matrix.T  # 旋转矩阵的转置
            # 由外参 [R|t] 计算世界坐标中的相机中心 ``-Rᵀ t``。
            position = -C2W@ translation_vector
            # 用于交换 x 和 y 轴的置换矩阵
            # 调整后的旋转矩阵
            # 计算相机中心与项目参考点 [25,15,0] 的欧氏距离。
            distance = np.linalg.norm(position-np.array([25,15,0]))
            # 打印每台相机的距离，便于检查标定。
            print(f"相机到原点的距离: {distance}")
            # 绘制相机位置
            # 将距离按相机编号存入列表，后面为光线设置 near/far 时使用。
            distances.append(distance)
        # 从所选数据中提取内参和外参
            # 读取当前相机内参 ``[3,3]`` K。
            K = np.array(data['Intrinsics'])  # 相机内参矩阵
            # 将 K 转为 float32 PyTorch 张量并收集。
            Ks.append(torch.from_numpy(K).float())
        # 保存 9 个内参张量的 Python 列表。
        self.intrinsics =Ks #torch.from_numpy(K).float()

        # 下面初始化逐帧/逐光线数据容器。
        # 收集每帧 .npy 路径。
        self.image_paths = []
        # 收集每帧相机旋转。
        self.poses = []
        # 收集光线。
        self.all_rays = []
        # 收集时间。
        self.all_times = []
        # 收集 BOS xy 位移。
        self.all_rgbs = []
        # 为深度数据保留空容器。
        self.all_depth = []
        # 收集每条光线的 W2C 旋转。
        self.all_W2C = []#和时间一起存在
        # 收集每帧的相机编号/view number。
        self.all_viewN=[]
        # 计算抽帧间隔；N_vis<0 时加载每一帧。
        img_eval_interval = (
            1 if self.N_vis < 0 else len(self.meta["frames"]) // self.N_vis
        )
        # 生成要加载的帧索引。
        idxs = list(range(0, len(self.meta["frames"]), img_eval_interval))

        # 再取第一帧用于构造初始相机方向。
        frame = self.meta["frames"][0]
        # 拼出第一帧 .npy 路径。
        image_path = os.path.join(self.root_dir,f"{frame['file_path']}")
        # 读取前两列像素 xy，转为 float32。
        imgcord=torch.from_numpy(np.load(image_path)[:,:2]).float()
        # 用当前变量 K 将像素转为 ``[(x-cx)/fx,(y-cy)/fy,1]`` 方向。
        # 【注意】循环后 K 保留为最后一台相机内参，原代码如此。
        self.directions=torch.stack(((imgcord[:,0]-K[0,2])/K[0,0],(imgcord[:,1]-K[1,2])/K[1,1],torch.ones(imgcord.shape[0])),-1)
        # 构造可选预处理缓存路径。
        self.processed_data_path = os.path.join(self.root_dir, "processed_data.pt")
        # 训练集且缓存存在时，从 .pt 直接加载。
        if (os.path.exists(self.processed_data_path))&(self.split=='train'):
            # 打印缓存加载日志。
            print("Loading processed data...")
            # 恢复缓存张量。
            self.load_processed_data()
        else:
            # 打印当前实验 transforms 文件名。
            print( f"transforms_{self.split}.json")
            # 带进度条地逐帧加载真实相机数据。
            for i in tqdm(
                idxs, desc=f"Loading data {self.split} ({len(idxs)})"
            ):  # img_list:#
                # 取第 i 帧 JSON 记录。
                frame = self.meta["frames"][i]
                # 读取帧内位姿列表为 NumPy；【暂时不用深究】当前后续未使用 pose。
                pose = np.array(frame["transform_matrix"])# @ self.blender2opencv
                # 读取该帧对应的 0～8 号相机索引。
                angle= frame['rotation']
                
                
                # 拼出当前帧 .npy 绝对路径。
                image_path = os.path.join(self.root_dir,f"{frame['file_path']}")
                # 记录该帧路径。
                self.image_paths += [image_path]
                # (h*w, 2)偏移量
                # 读取后两列，得到 ``[N_pixels,2]`` BOS xy 位移。
                img= torch.from_numpy(np.load(image_path)[:,2:]).float()
                
                # img = img[:, :3] * img[:, -1:] + (
                #     1 - img[:, -1:]
                # )  # blend A to RGB, white background
                
                # 收集位移真值；实验路径在此处不额外加噪。
                self.all_rgbs += [img]
                
                #ocxyz=torch.zeros(xp.shape[0],yp.shape[0],3).to(torch.float64)
                # 按 rotation 索引选择该帧真正对应的相机标定。
                selected_camera_data = camera_data[angle]  # 例如，选择第一个相机
                # 把相机编号转为标量张量并收集。
                self.all_viewN+=[torch.tensor(angle)]
                # 从所选数据中提取内参和外参
                # 读取当前相机 ``[3,3]`` 内参 K。
                K = np.array(selected_camera_data['Intrinsics'])  # 相机内参矩阵
                # 读取旋转并转为 PyTorch ``[3,3]``。
                rotation_matrix = torch.Tensor(selected_camera_data['Rotation'])  # 旋转矩阵
                # 读取平移并转为 ``[3]`` 张量。
                translation_vector =torch.Tensor(selected_camera_data['TranslationVector'])  # 平移向量
                #dist_coeffs = torch.tensor(selected_camera_data['RadialDistortion'])  # k1, k2
                # 转置标定旋转，用于将相机方向转到世界坐标。
                rotation_matrix=rotation_matrix.T
                # 为转置后旋转创建 C2W 别名。
                C2W = rotation_matrix  # 旋转矩阵的转置
                # 由标定平移计算世界坐标中的相机光心 ``-C2W*t``。
                TC2W = -C2W@ translation_vector
                # 使用 torch.load 加载文件
                # 创建 4×4 单位矩阵，用于组装标定的齐次变换。
                transform_matrix = np.eye(4)  # 创建一个4x4的单位矩阵
                # 将 C2W.T（即标定原旋转）写入左上 3×3。
                transform_matrix[:3, :3] = C2W.T  # 将旋转矩阵放入左上角
                # 将标定平移写入最后一列前 3 项。
                transform_matrix[:3, 3] = translation_vector 
                #print(transform_matrix)
                # 【暂时不用深究】创建零起点占位 ``[N_pixels,3]``，后面未使用。
                ocxyz=torch.zeros(imgcord.shape[0],3)
                # 用当前 K 将像素转为归一化相机方向 ``[(x-cx)/fx,(y-cy)/fy,1]``。
                dcxyz=torch.stack(((imgcord[:,0]-K[0,2])/K[0,0],(imgcord[:,1]-K[1,2])/K[1,1],torch.ones(imgcord.shape[0])),-1)
                # xu, yu = self.undistort_points_lm(dcxyz[:,0].reshape(-1), dcxyz[:,1].reshape(-1), dist_coeffs[0], dist_coeffs[1], max_iter=100)
                # dcxyz = torch.stack((xu, yu, torch.ones_like(xu)), -1)  # 归一化相机坐标
                # 以当前相机距离为中心，前后各放宽 50 得 near/far。
                # 【注意】这两个局部变量在本函数后续没有拼入 all_rays。
                near,far=distances[angle]-50,distances[angle]+50
                # 将每个相机方向乘 C2W，得世界坐标光线方向 ``[N_pixels,3]``。
                rays_d = torch.sum(dcxyz[..., np.newaxis, :] * C2W , -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
                # 所有光线从同一相机光心 TC2W 出发，扩展到与 rays_d 同形状。
                rays_o = TC2W.expand(rays_d.shape)
                # 收集本帧 ``[3,3]`` C2W 旋转。
                self.poses += [C2W]







                
                # 拼成 ``[N_pixels,6]`` [光心 xyz,方向 xyz] 并收集。
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
                # 优先用 JSON time，否则按帧号归一化，再扩展至每个像素。
                cur_time = torch.tensor(
                    frame["time"]
                    if "time" in frame
                    else float(i) / (len(self.meta["frames"]) - 1)
                ).expand(rays_o.shape[0], 1)
                # 收集本帧 ``[N_pixels,1]`` 时间。
                self.all_times += [cur_time]
                # 从齐次标定矩阵取左上 3×3 W2C 旋转并转为 PyTorch。
                w2c=torch.from_numpy(transform_matrix[:3,:3])
                # 本帧每个像素复制同一 W2C，得 ``[N_pixels,3,3]``。
                self.all_W2C += [w2c.unsqueeze(0).repeat(self.directions.shape[0],1,1)]
            # 将所有帧 C2W 旋转堆成 ``[N_frames,3,3]``。
            self.poses = torch.stack(self.poses)
            #  self.is_stack stacks all images into a big chunk, with shape (N, H, W, 3).
            #  Otherwise, all images are kept as a set of rays with shape (N_s, 3), where N_s = H * W * N
            # 非堆叠模式将所有帧展成一张光线表，便于训练抽样。
            if not self.is_stack:
                # 连接为 ``[N_frames*N_pixels,6]`` 光线。
                self.all_rays = torch.cat(
                    self.all_rays, 0
                )  # (len(self.meta['frames])*h*w, 3)
                # 连接为 ``[N_frames*N_pixels,2]`` 位移真值。
                self.all_rgbs = torch.cat(
                    self.all_rgbs, 0
                )  # (len(self.meta['frames])*h*w, 3)
                # 连接每条光线的时间。
                self.all_times = torch.cat(self.all_times, 0)
                # 连接每条光线的 W2C。
                self.all_W2C = torch.cat(self.all_W2C, 0)
                #############################################################################################################
                # Save all tensors in a file for later use
                # if self.split=='train':
                #     torch.save({
                #         "rays": self.all_rays,
                #         "times": self.all_times,
                #         "rgbs": self.all_rgbs,
                #         "W2C": self.all_W2C
                #     }, os.path.join(self.root_dir, "processed_data.pt")
                #     )
            else:
                # 堆叠模式保留帧维，光线为 ``[N_frames,N_pixels,6]``。
                self.all_rays = torch.stack(
                    self.all_rays, 0
                )  # (len(self.meta['frames]),h*w, 3)h
                self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(
                    -1, w,h, 2
                )  # (len(self.meta['frames]),h,w,3)
                # 堆叠时间并保留帧维。
                self.all_times = torch.stack(self.all_times, 0)
                # 堆叠 W2C 并保留帧维。
                self.all_W2C = torch.stack(self.all_W2C, 0)
                # 把每帧标量相机编号堆成 ``[N_frames]``。
                self.all_viewN = torch.stack(self.all_viewN, 0)
    def define_transforms(self):
        """创建 Pillow/NumPy 图像转 PyTorch CHW 张量的 torchvision 操作。"""
        # 保存 ToTensor 可调用对象；当前 .npy 主路径直接 from_numpy，但保留原框架接口。
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        """由相机内参和位姿计算每帧 ``[3,4]`` 投影矩阵并保存到 ``self.proj_mat``。"""
        # 将 K 增加 batch 维，乘 C2W 齐次矩阵的逆之前 3 行，得 K[R|t]。
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def world2ndc(self, points, lindisp=None):
        """将世界坐标点按场景中心/半径归一化到 NDC 尺度。

        ``points`` 末维为 xyz，返回同形状张量。``lindisp`` 为原接口保留参数，本函数未使用。
        【注意】``center/radius`` 的初始化代码在 __init__ 中已被注释。
        """
        # 读取输入点所在 CPU/GPU，便于搬运 center/radius。
        device = points.device
        # 先减场景中心，再逐 xyz 除以半径，完成归一化。
        return (points - self.center.to(device)) / self.radius.to(device)

    def __len__(self):
        """返回 Dataset 可索引样本数，定义为 ``all_rgbs`` 第 0 维长度。

        非堆叠训练模式下通常是总光线数；堆叠测试模式下通常是帧数。
        """
        # Python len 对张量返回第 0 维长度。
        return len(self.all_rgbs)

    def get_val_pose(self):
        """生成 40 个绕场景旋转的验证 C2W 位姿和对应时间。

        返回 ``render_poses [40,4,4]`` 与 ``render_times [40]``。
        【物理直觉】这是合成一台绕重建体旋转的虚拟相机，可用于新视角渲染。
        """
        # 在 [-180°,180°) 取 40 个角度，phi=-30°、半径=15，生成并堆叠 C2W。
        render_poses = torch.stack(
            [
                pose_spherical(angle, -30.0, 15.0)
                for angle in np.linspace(-180, 180, 40 + 1)[:-1]
            ],
            0,
        )
        # 生成 0～1 时间，再线性映射到 [-1,1]。
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        # 返回位姿，同时用 time_scale 缩放时间。
        return render_poses, self.time_scale * render_times

    def get_val_rays(self):
        """为 ``get_val_pose`` 产生的每个虚拟相机生成完整像素光线。

        返回 ``rays_all`` Python 列表，每项 ``[N_pixels,6]``；以及 ``[40]`` 时间张量。
        """
        # 先生成 40 个验证位姿和时间。
        val_poses, val_times = self.get_val_pose()  # get valitdation poses and times
        # 创建列表收集每个虚拟相机的光线表。
        rays_all = []  # initialize list to store [rays_o, rays_d]

        # 遍历所有验证 C2W。
        for i in range(val_poses.shape[0]):
            # 将第 i 个位姿转为 float32 张量。
            c2w = torch.FloatTensor(val_poses[i])
            # 用共用像素方向与 C2W 生成 ``[N_pixels,3]`` 起点/方向。
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            # 拼成 ``[N_pixels,6]`` 光线表。
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            # 收集当前视角光线。
            rays_all.append(rays)
        # 返回光线列表和 float32 验证时间。
        return rays_all, torch.FloatTensor(val_times)

    def __getitem__(self, idx):
        """【主线】按索引取出一个训练光线样本或一幅完整测试帧。

        返回字典固定包含：``rays``、``rgbs``（BOS xy 位移）、``time``、``w2c``。
        训练非堆叠模式下，单项常见形状分别是 ``[6]``、``[2]``、``[1]``、``[3,3]``；
        测试堆叠模式下则对应整帧 ``[N_pixels,...]``/位移图。
        """
        # 训练划分直接从已展平缓冲区按 idx 取同一光线的各字段。
        if self.split == "train":  # use data in the buffers
            # 组装一个四键 Python 字典。
            sample = {
                # 三维光线 [origin xyz,direction xyz]。
                "rays": self.all_rays[idx],
                # BOS x/y 位移真值。
                "rgbs": self.all_rgbs[idx],
                # 光线所属时刻。
                "time": self.all_times[idx],
                # 世界到相机旋转矩阵。
                "w2c": self.all_W2C[idx],
            }
        else:  # create data for each image separately
            # 测试模式先取第 idx 帧位移真值。
            img = self.all_rgbs[idx]
            # 取该帧所有光线。
            rays = self.all_rays[idx]
            # 取该帧所有光线的时间。
            time = self.all_times[idx]
            # 取该帧所有光线的 W2C。
            w2c=self.all_W2C[idx]
            # 用与训练分支相同的四个键组装整帧样本。
            sample = {"rays": rays, "rgbs": img, "time": time,"w2c":w2c}
        # 返回上层训练/评估代码期望的样本字典。
        return sample

    def get_random_pose(self, batch_size, patch_size, batching="all_images"):
        """从随机相机光线图中抽取若干个连续方形补丁，用于 RegNeRF 几何正则化。

        ``batch_size`` 是期望的总光线数，``patch_size`` 是方形边长，
        因此补丁数为 ``batch_size//patch_size²``。返回抽取光线 ``out`` 与相应随机时间。
        【注意】该路径依赖先成功调用 ``init_random_pose``。
        """
        # 每个 patch 有 patch_size² 条光线，整除得到可抽取的完整补丁数。
        n_patches = batch_size // (patch_size**2)

        # 读取随机相机光线图数量。
        N_random = self.random_rays.shape[0]
        # Sample images
        # all_images 模式下，每个补丁独立选一幅随机光线图。
        if batching == "all_images":
            # 在 [0,N_random) 中生成 ``[n_patches,1]`` 图像索引。
            idx_img = np.random.randint(0, N_random, size=(n_patches, 1))
        # single_image 模式下，所有补丁来自同一幅随机图。
        elif batching == "single_image":
            # 先随机选一个图像索引标量。
            idx_img = np.random.randint(0, N_random)
            # 将该索引复制成 ``[n_patches,1]``。【注意】np.int 是原代码的旧 NumPy 别名。
            idx_img = np.full((n_patches, 1), idx_img, dtype=np.int)
        else:
            # 不支持其他抽样方式，明确报错。
            raise ValueError("Not supported batching type!")
        # 将 NumPy 图像索引转为 PyTorch int64，用于高级索引。
        idx_img = torch.Tensor(idx_img).long()
        # 从第一幅随机光线图读取 H、W。
        H, W = self.random_rays[0].shape[0], self.random_rays[0].shape[1]
        # Sample start locations
        # 在图像水平中间 50% 区域选择每个补丁的左上 x，并确保 patch 不越界。
        x0 = np.random.randint(
            int(W // 4), int(W // 4 * 3) - patch_size + 1, size=(n_patches, 1, 1)
        )
        # 在竖直中间 50% 区域选择左上 y。
        y0 = np.random.randint(
            int(H // 4), int(H // 4 * 3) - patch_size + 1, size=(n_patches, 1, 1)
        )
        # 沿末维拼成每个补丁左上角 ``[x0,y0]``。
        xy0 = np.concatenate([x0, y0], axis=-1)
        # 生成一个 patch_size×patch_size 的局部 xy 坐标网格，加到每个左上角。
        patch_idx = xy0 + np.stack(
            np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing="xy"),
            axis=-1,
        ).reshape(1, -1, 2)

        # 把补丁像素索引转为 PyTorch int64。
        patch_idx = torch.Tensor(patch_idx).long()
        # Subsample images
        # 用图像索引、y 索引和 x 索引一次取出所有 patch 光线。
        out = self.random_rays[idx_img, patch_idx[..., 1], patch_idx[..., 0]]

        # 返回补丁光线与对应随机相机时间。
        return out, self.random_times[idx_img]
