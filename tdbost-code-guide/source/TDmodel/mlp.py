"""
【主线】这个文件定义了“坐标编码 + 多层感知机（MLP）”。

零基础读法：可以先把 MLP 想成一台可学习的函数机。它接收某个三维位置、
时刻、分解特征和观看方向，最后输出该处的密度、颜色或其他物理量。
在三维重建中，上游的张量分解先为每个空间点提取一组特征，
本文件的网络再把这些特征“翻译”成真正要预测的物理值。
"""

# 【暂时不用深究】从 typing 导入的是类型标注工具；本文件保留原作者的导入，其中多个名称当前未使用。
from typing import Callable, Collection, Dict, Iterable, List, Optional, Sequence, Union

# 【主线】导入 PyTorch：本项目用它表示张量、搭建神经网络并自动求导。
import torch


def positional_encoding(positions, freqs):
    """
    【主线】对连续坐标或特征做正弦/余弦位置编码。

    参数：
        positions：待编码张量，形状通常是 ``(..., D)``；``D`` 是原始维数。
        freqs：频率档数 ``F``。

    返回：
        形状是 ``(..., 2 * F * D)`` 的张量。前半是 sin，后半是 cos。

    【数学直觉】直接给网络一个坐标，它很难表示细密的快速变化；
    把坐标同时映射到多个频率的波形上，网络更容易拟合三维场中的细节。
    """
    # 【数学直觉】构造 [1, 2, 4, ..., 2^(F-1)] 的倍频数组，并放到与输入相同的计算设备。
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)
    # 【主线】给原张量最后增加“频率”轴，与每个倍频相乘，再将频率和原特征轴合并。
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )
    # 【主线】分别取 sin 和 cos，然后沿最后一维拼在一起。
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    # 【主线】把编码后的高维特征交给调用者。
    return pts

# ...existing code...
class General_MLP(torch.nn.Module):
    """
    【主线】把“时间 + 张量分解特征 + 三维坐标 + 视线方向”映射到目标物理量。

    MLP 是 Multi-Layer Perceptron（多层感知机）。在这里，每个样本可以理解为
    射线上的一个三维采样点。输入的前导形状都应相同，只有最后一维不同：

    - ``pts``：``(..., 3)``，空间坐标 ``(x, y, z)``。
    - ``features``：``(..., inChanel)``，从 CP/MM 张量分解中查到的潜在特征。
    - ``frame_time``：``(..., 1)``，时刻。
    - ``views``：通常是 ``(..., 3)``，射线/相机观看方向。
    - 输出：``(..., outChanel)``，例如 1 维密度或 3 维位移/颜色。

    编码参数的规则：``pe > 0`` 表示使用原值与 PE；``pe == 0`` 只用原值；
    ``pe < 0`` 表示完全不把该类输入交给网络。

    【三维重建意义】张量分解用少量平面/线保存了整个四维时空场；
    这个 MLP 是解码器，它决定某个坐标和时刻上应该重建出什么值。
    """
    def __init__(
        self,
        inChanel: int,
        outChanel: int,
        t_pe: int = 6,
        fea_pe: int = 6,
        pos_pe: int = 6,
        view_pe: int = 6,
        featureC: int = 128,
        n_layers: int = 3,
        use_sigmoid: bool = False,
        zero_init: bool = True,
        batchnorm: bool = False,
    ):
        """
        【主线】根据输入类型、位置编码阶数和层数，搭建一个 MLP。

        ``featureC`` 是隐藏层宽度，``n_layers`` 是线性层总数；``zero_init`` 只将
        最后一层的偏置设为 0，``batchnorm`` 在原代码中实际对应 LayerNorm。
        """
        # 【暂时不用深究】初始化 PyTorch 的 nn.Module 基类，使参数能被训练器发现。
        super().__init__()

        # 【主线】从 0 开始累计拼接后会送入 MLP 的总通道数。
        self.in_mlpC = 0
        # 【注意】编码阶数大于等于 0 才启用对应输入；负数是“禁用”开关。
        self.use_t = t_pe >= 0
        # 是否使用张量分解查出的 features。
        self.use_fea = fea_pe >= 0
        # 是否使用三维空间坐标。
        self.use_pos = pos_pe >= 0
        # 是否使用观看/光线方向。
        self.use_view = view_pe >= 0
        # 保存各类位置编码的频率档数，供 forward 判断。
        self.t_pe = t_pe
        self.fea_pe = fea_pe
        self.pos_pe = pos_pe
        self.view_pe = view_pe
        # 【注意】保留这个开关，但本版 forward 中对 sigmoid 的调用被注释，所以当前不生效。
        self.use_sigmoid = use_sigmoid

        # 【数学直觉】原始 1 维时间 + sin/cos 各 F 维，因此共 1 + 2F 维。
        if self.use_t:
            self.in_mlpC += 1 + 2 * t_pe * 1
        # 特征原有 inChanel 维，每个频率又生成 sin/cos 两份。
        if self.use_fea:
            self.in_mlpC += inChanel+2 * fea_pe * inChanel
        # 空间坐标原有 x/y/z 三维，再加 2 * pos_pe * 3 维编码。
        if self.use_pos:
            self.in_mlpC += 3 + 2 * pos_pe * 3
        # 视线方向同样是三维向量。
        if self.use_view:
            self.in_mlpC += 3 + 2 * view_pe * 3

        # 【注意】本结构至少需要“输入层 + 输出层”两个线性层。
        assert n_layers >= 2  # Ensure there are at least two layers
        # 记录是否在最后对输出做 LayerNorm。
        self.batchnorm = batchnorm
        # 保存线性层数，方便 forward 了解网络配置。
        self.n_layers = n_layers
        # 【数学直觉】网络较深时启用跳连接，避免原始坐标信息经过多层后丢失。
        self.use_skip = n_layers > 5
        # 第 4 个线性层接收“当前隐藏特征 + 原始 MLP 输入”。
        self.skip_at = 4  # 1-based index of linear layer receiving the skip

        # 【暂时不用深究】ModuleList 会正确登记其中每层的可训练参数。
        layers = torch.nn.ModuleList()
        # 记录当前正在创建第几个线性层，第一层从 1 开始。
        linear_idx = 1
        # 【主线】第一个线性层：把拼接后的输入映射到 featureC 维隐藏空间。
        layers.append(torch.nn.Linear(self.in_mlpC, featureC))
        # SiLU 是非线性激活函数；没有它，多层线性变换仍等价于一层。
        layers.append(torch.nn.SiLU())

        # 【主线】创建 n_layers - 2 个中间线性层，每层后都接 SiLU。
        for _ in range(n_layers - 2):
            # 进入下一个线性层的编号。
            linear_idx += 1
            # 如果当前是预定的跳连接层，输入维数要加上原始输入的维数。
            if self.use_skip and linear_idx == self.skip_at:
                # 【数学直觉】这层将看到 featureC + in_mlpC 个数。
                layers.append(torch.nn.Linear(featureC + self.in_mlpC, featureC))
            else:
                # 普通中间层的输入和输出宽度都是 featureC。
                layers.append(torch.nn.Linear(featureC, featureC))
            # 为这个中间线性层配一个 SiLU 激活函数。
            layers.append(torch.nn.SiLU())

        # 【主线】最后一层把 featureC 维隐藏特征转成 outChanel 维物理输出。
        layers.append(torch.nn.Linear(featureC, outChanel))

        # 把创建好的层挂到 self 上，之后调用 model(...) 时会按顺序执行。
        self.layers = layers
        # 如果开启归一化，创建对 outChanel 维输出的 LayerNorm。
        if self.batchnorm:
            self.norm = torch.nn.LayerNorm(outChanel)
        # 【暂时不用深究】可选地把最后线性层的偏置初始化为 0。
        if zero_init:
            torch.nn.init.constant_(self.layers[-1].bias, 0)

    def forward(
        self,
        pts: torch.Tensor,
        features: torch.Tensor,
        frame_time: torch.Tensor,
        views=torch.zeros(1,1)
    ) -> torch.Tensor:
        """
        【主线】执行一次前向计算：编码并拼接输入，再逐层生成预测。

        输入和输出形状见类的说明。“前向”只表示从输入算到输出；
        训练时的梯度反向传播会由 PyTorch 自动完成。
        """
        # 【主线】先用列表收集所有已启用的输入块。
        indata = []
        # 如果启用时间，先放入原始时间值。
        if self.use_t:
            indata += [frame_time]
            # 频率数大于 0 时，再放入时间的 sin/cos 编码。
            if self.t_pe > 0:
                indata += [positional_encoding(frame_time, self.t_pe)]
        # 如果启用分解特征，放入原特征。
        if self.use_fea:
            indata += [features]
            # 按配置追加分解特征的多频编码。
            if self.fea_pe > 0:
                indata += [positional_encoding(features, self.fea_pe)]
        # 如果启用坐标，放入每个采样点的 x/y/z。
        if self.use_pos:
            indata += [pts]
            # 按配置追加坐标的多频编码。
            if self.pos_pe > 0:
                indata += [positional_encoding(pts, self.pos_pe)]
        # 如果启用视线，放入观看/射线方向。
        if self.use_view:
            indata += [views]
            # 按配置追加视线方向的多频编码。
            if self.view_pe > 0:
                indata += [positional_encoding(views, self.view_pe)]
        # 【注意】所有输入前导形状必须一致，这里仅沿最后的“通道”维拼接。
        mlp_in = torch.cat(indata, dim=-1)

        # 【主线】x 是正在层与层之间传递的中间张量，起点就是拼接好的输入。
        x = mlp_in
        # 单独统计已经遇到多少个 Linear，因为 ModuleList 里还混有 SiLU。
        linear_count = 0
        # 按创建顺序遍历每一层。
        for module in self.layers:
            # 如果当前层是线性层，先处理可能的跳连接。
            if isinstance(module, torch.nn.Linear):
                # 线性层计数加 1。
                linear_count += 1
                # 在指定层之前，将原始 mlp_in 再次拼回来。
                if self.use_skip and linear_count == self.skip_at:
                    # 【数学直觉】跳连接给深层网络一条保留原始坐标信息的“快车道”。
                    x = torch.cat([x, mlp_in], dim=-1)
                # 用当前线性层变换 x。
                x = module(x)
            else:
                # 非 Linear 层（本结构中是 SiLU）直接作用于 x。
                x = module(x)

        # 若配置了归一化，对最终通道做 LayerNorm。
        if self.batchnorm:
            x = self.norm(x)
        # 【注意】变量名 rgb 是历史命名；它也可以代表密度、位移或其他物理量。
        rgb = x
        # if self.use_sigmoid:
        #     rgb = torch.sigmoid(rgb)

        # 【主线】返回形状 ``(..., outChanel)`` 的网络预测。
        return rgb
# ...existing code...
# class General_MLP(torch.nn.Module):
#     """
#     A general MLP module with potential input including time position encoding(PE): t_pe, feature PE: fea_pe, 3D position PE: pos_pe,
#     view direction PE: view_pe.

#     pe > 0: use PE with frequency = pe.
#     pe < 0: not use this feautre.
#     pe = 0: only use original value.
#     """
#     def __init__(
#         self,
#         inChanel: int,
#         outChanel: int,
#         t_pe: int = 6,
#         fea_pe: int = 6,
#         pos_pe: int = 6,
#         view_pe: int = 6,
#         featureC: int = 128,
#         n_layers: int = 3,
#         use_sigmoid: bool = False,
#         zero_init: bool = True,
#     ):
#         super().__init__()

#         self.in_mlpC = 0
#         self.use_t = t_pe >= 0
#         self.use_fea = fea_pe >= 0
#         self.use_pos = pos_pe >= 0
#         self.use_view = view_pe >= 0
#         self.t_pe = t_pe
#         self.fea_pe = fea_pe
#         self.pos_pe = pos_pe
#         self.view_pe = view_pe
#         self.use_sigmoid = use_sigmoid

#         # Adjust input channel size based on positional encodings
#         if self.use_t:
#             self.in_mlpC += 1 + 2 * t_pe * 1
#         if self.use_fea:
#             self.in_mlpC += inChanel+2 * fea_pe * inChanel
#         if self.use_pos:
#             self.in_mlpC += 3 + 2 * pos_pe * 3
#         if self.use_view:
#             self.in_mlpC += 3 + 2 * view_pe * 3

#         assert n_layers >= 2  # Ensure there are at least two layers
#         layers = [torch.nn.Linear(self.in_mlpC, featureC), torch.nn.Tanh()]#Tanh()

#         for _ in range(n_layers - 2):
#             layers += [torch.nn.Linear(featureC, featureC), torch.nn.Tanh()]
#         layers += [torch.nn.Linear(featureC, outChanel)]

#         self.mlp = torch.nn.Sequential(*layers)

#         # Optionally initialize the last layer to zero
#         if zero_init:
#             torch.nn.init.constant_(self.mlp[-1].bias, 0)

#     def forward(
#         self,
#         pts: torch.Tensor,
#         features: torch.Tensor,
#         frame_time: torch.Tensor,
#         views=torch.zeros(1,1)
#     ) -> torch.Tensor:
#         """
#         MLP forward.
#         """
#         # Collect input data
#         indata = []
#         if self.use_t:
#             indata += [frame_time]
#             if self.t_pe > 0:
#                 indata += [positional_encoding(frame_time, self.t_pe)]
#         if self.use_fea:
#             indata += [features]
#             if self.fea_pe > 0:
#                 indata += [positional_encoding(features, self.fea_pe)]
#         if self.use_pos:
#             indata += [pts]
#             if self.pos_pe > 0:
#                 indata += [positional_encoding(pts, self.pos_pe)]
#         if self.use_view:
#             indata += [views]
#             if self.view_pe > 0:
#                 indata += [positional_encoding(views, self.view_pe)]
#         mlp_in = torch.cat(indata, dim=-1)

#         rgb = self.mlp(mlp_in)
#         # if self.use_sigmoid:
#         #     rgb = torch.sigmoid(rgb)

#         return rgb
