"""用于约束重建结果平滑性的总变分（Total Variation, TV）损失。"""

# PyTorch 提供张量运算与自动求导。
import torch
# torch.nn 提供神经网络模块基类 nn.Module。
import torch.nn as nn


class TVLoss(nn.Module):
    """【主线】惩罚相邻网格值突然跳变的总变分损失。

    【物理直觉】密度、折射率等真实物理场通常在空间中连续；TV 损失
    会鼓励相邻体素更接近，从而减少噪点。输入 ``x`` 通常是
    ``[batch, channel, height, width]`` 的 4 维张量，也保留了对非 4 维输入的原代码分支。
    """

    def __init__(self, TVLoss_weight_dim1=1.0, TVLoss_weight_dim2=1.0):
        """保存两个空间维度上的损失权重。"""
        # 先初始化 PyTorch 的 nn.Module 基类，才能正常参与训练。
        super(TVLoss, self).__init__()
        # 保存第一个空间维度的惩罚强度。
        self.TVLoss_weight_dim1 = TVLoss_weight_dim1
        # 保存第二个空间维度的惩罚强度。
        self.TVLoss_weight_dim2 = TVLoss_weight_dim2

    def forward(self, x):
        """计算输入张量 ``x`` 的 TV 损失，返回一个标量张量。"""
        # 读取第 0 维的 batch 数，最后用它对损失取平均。
        batch_size = x.size()[0]
        # 【注意】原代码为非 4 维张量保留了一个单方向差分分支。
        if x.dim() != 4:
            # 取第 2 维长度，后面用于对齐相邻切片。
            h_x = x.size()[2]
            # 计算这个分支中纵向差分的元素数。
            count_h = x.size()[1] *( x.size()[2]-1)
            # 【暂时不用深究】原代码同时保留 count_w，虽然此分支中 w_tv 为 0。
            count_w = x.size()[1] *( x.size()[2]-1)
            # 计算相邻位置之差的平方和，再乘第一维权重。
            h_tv = (
                torch.pow((x[:, :, 1:,] - x[:, :, : h_x - 1]), 2).sum()
                * self.TVLoss_weight_dim1
            )
            # 非 4 维分支不计算第二方向的差分。
            w_tv =0
        else:
            # 4 维情况下，第 2 维被视为高度 H。
            h_x = x.size()[2]
            # 第 3 维被视为宽度 W。
            w_x = x.size()[3]
            # 统计参与纵向差分的元素数，供归一化使用。
            count_h = self._tensor_size(x[:, :, 1:, :])
            # 统计参与横向差分的元素数。
            count_w = self._tensor_size(x[:, :, :, 1:])
            # 【物理直觉】比较上下相邻网格：变化越剧烈，h_tv 越大。
            h_tv = (
                torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
                * self.TVLoss_weight_dim1
            )
            # 比较左右相邻网格：变化越剧烈，w_tv 越大。
            w_tv = (
                torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
                * self.TVLoss_weight_dim2
            )
        # 将两个方向分别按元素数和 batch 数归一化，返回一个损失标量。
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        """计算 4 维张量除 batch 维之外的元素数，返回 Python 整数。"""
        # 相乘 channel、height、width，不把 batch 数计入单个样本的归一化分母。
        return t.size()[1] * t.size()[2] * t.size()[3]

