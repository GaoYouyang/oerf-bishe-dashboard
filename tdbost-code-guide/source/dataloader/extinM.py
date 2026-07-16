"""用球坐标角度构造相机外参（camera-to-world）矩阵。

【主线】相机外参回答「相机在三维世界的什么位置、朝哪个方向」。
本文件先定义平移和两个旋转，再在 ``pose_spherical`` 中按固定顺序组合。
"""

# 【暂时不用深究】原仓库导入的绘图库，本文件激活路径中未使用。
import matplotlib.pyplot as plt
# NumPy 提供π、正弦和余弦，用于构造旋转矩阵。
import numpy as np
# PyTorch 用于创建 4×4 齐次变换张量和进行矩阵乘法。
import torch
# 沿 z 轴把相机平移 t；返回 ``[4,4]`` 浮点变换矩阵。
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

# 绕 x 轴旋转 phi（弧度）；返回 ``[4,4]`` 齐次旋转矩阵。
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

# 绕 y 轴旋转 th（弧度）；返回 ``[4,4]`` 齐次旋转矩阵。
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    """根据球面角度和半径生成 ``[4,4]`` 相机到世界（C2W）矩阵。

    ``theta``/``phi`` 以度为单位，``radius`` 是相机与旋转中心的距离。
    【物理直觉】先把相机推到半径位置，再绕两个轴旋转，就像相机绕被测流场走一圈。
    """
    # 第一步：沿 z 轴平移 radius，得到初始相机位置。
    c2w = trans_t(radius)
    # 将 phi 从角度转弧度，左乘 x 轴旋转矩阵。【注意】矩阵乘法顺序会影响结果。
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    # 将 theta 从角度转弧度，再左乘 y 轴旋转。
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    # 通过固定坐标轴置换/翻转矩阵，对齐项目使用的相机坐标约定。
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    # 返回组合完成的相机到世界齐次变换矩阵。
    return c2w
