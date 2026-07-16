"""
生成球面相机位姿（camera-to-world 矩阵）的小工具。

【主线】先把相机沿 z 轴移到指定半径，再绕 x/y 轴旋转，最后换成项目使用的坐标轴约定。
"""

# Matplotlib 的绘图接口；【注意】本文件当前没有实际调用它，但保留原 import 以确保代码行为不变。
import matplotlib.pyplot as plt
# NumPy：提供圆周率 pi 以及 sin/cos 三角函数。
import numpy as np
# PyTorch：用张量保存 4×4 齐次坐标变换矩阵。
import torch

# 【暂时不用深究】齐次坐标用四维向量 (x,y,z,1) 同时表示旋转和平移。
# 输入 t 是沿 z 轴平移的距离；输出是 [4,4] 浮点张量。
trans_t = lambda t : torch.Tensor([
    # x' = x，x 方向不平移。
    [1,0,0,0],
    # y' = y，y 方向不平移。
    [0,1,0,0],
    # z' = z+t：右上角的 t 就是 z 方向的平移量。
    [0,0,1,t],
    # 齐次坐标的最后一行固定为 [0,0,0,1]。
    [0,0,0,1]]).float()

# 输入 phi 为弧度；输出绕 x 轴旋转的 [4,4] 矩阵。
rot_phi = lambda phi : torch.Tensor([
    # 绕 x 轴旋转时，x 坐标保持不变。
    [1,0,0,0],
    # y-z 平面中的标准旋转矩阵第一行。
    [0,np.cos(phi),-np.sin(phi),0],
    # y-z 平面中的标准旋转矩阵第二行。
    [0,np.sin(phi), np.cos(phi),0],
    # 保留齐次坐标分量。
    [0,0,0,1]]).float()

# 输入 th 为弧度；输出绕 y 轴旋转的 [4,4] 矩阵。
rot_theta = lambda th : torch.Tensor([
    # x-z 平面中的旋转矩阵第一行。
    [np.cos(th),0,-np.sin(th),0],
    # 绕 y 轴旋转时，y 坐标保持不变。
    [0,1,0,0],
    # x-z 平面中的旋转矩阵第二行。
    [np.sin(th),0, np.cos(th),0],
    # 保留齐次坐标分量。
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    """
    【主线】根据球坐标角度和半径构造相机到世界的位姿矩阵。

    输入：
        theta: 方位角，单位是度，函数内会转为弧度并绕 y 轴旋转。
        phi: 俯仰角，单位是度，函数内会转为弧度并绕 x 轴旋转。
        radius: 相机与原点的距离。

    输出：
        c2w: [4,4] 的 camera-to-world 齐次变换矩阵。

    三维重建意义：每个观测视角都需要一个这样的矩阵，把相机坐标系中的光线转到统一的世界坐标系。
    """
    # 第 1 步：从单位矩阵出发，把相机沿 z 轴移动 radius。
    c2w = trans_t(radius)
    # 第 2 步：将 phi 从角度转成弧度，左乘绕 x 轴的旋转矩阵。
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    # 第 3 步：将 theta 从角度转成弧度，左乘绕 y 轴的旋转矩阵。
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    # 第 4 步：轴置换/反向，把常见 NeRF 相机坐标约定转成本项目使用的世界坐标约定。
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    # 返回最终的相机位姿矩阵。
    return c2w
