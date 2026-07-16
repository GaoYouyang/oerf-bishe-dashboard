# 【文件作用】这是 proj 二进制扩展的“类型说明书”，不是实际算法源码。
# 它由 Nuitka 自动生成，帮助编辑器知道 proj 模块中有哪些名字。
# 真正执行的机器码位于 proj.cpython-311-x86_64-linux-gnu.so。

# 让类型注解使用较新的延迟解析规则。
from __future__ import annotations
# 二进制模块内部会调用球面角度到相机位姿的转换函数。
from All_util.extinM import pose_spherical
# 三维绘图坐标轴类型。
from mpl_toolkits.mplot3d import Axes3D
# 规则三维网格插值器。
from scipy.interpolate import RegularGridInterpolator
# MATLAB 文件读取和保存函数。
from scipy.io import loadmat, savemat
# SymPy 中的 false 符号；这是生成器记录的内部依赖。
from sympy import false
# PyTorch 的批量数据加载器和张量数据集。
from torch.utils.data import DataLoader, TensorDataset
# Any 表示类型检查器暂时不限定具体类型。
from typing import Any
# Matplotlib 绘图模块。
import matplotlib.pyplot
# NumPy 数值数组模块。
import numpy
# SciPy 的多维图像/数组处理模块。
import scipy.ndimage
# PyTorch 张量计算模块。
import torch
# PyTorch 常用函数接口。
import torch.nn.functional


def proj(isspray: Any) -> Any:
    """
    调用已编译的数据投影程序。

    参数：
        isspray: 是否使用喷雾案例。projget.py 传入 True。
    返回：
        自动生成的存根只标成 Any，无法从公开 Python 源码确定精确类型。

    省略号表示“这里只声明接口，具体实现由 .so 二进制提供”。
    """
    # 类型存根中的 ... 不是漏写代码，而是有意表示实现不可见。
    ...


# 告诉类型检查器该二进制模块也具有标准的 __name__ 属性。
__name__ = ...


# 下面都是二进制模块内部曾使用的依赖。
# Nuitka 把它们列出，以便打包/类型工具看见隐式依赖。
# PyTorch 主模块。
import torch
# Matplotlib 主模块。
import matplotlib
# Matplotlib 绘图子模块。
import matplotlib.pyplot
# 三维绘图工具包。
import mpl_toolkits
# 三维坐标轴子模块。
import mpl_toolkits.mplot3d
# 三维坐标轴类型模块。
import mpl_toolkits.mplot3d.Axes3D
# 本项目公共工具包。
import All_util
# 本项目相机位姿工具。
import All_util.extinM
# NumPy 主模块。
import numpy
# SymPy 符号计算主模块。
import sympy
# SymPy 的 false 符号模块。
import sympy.false
# SciPy 主模块。
import scipy
# SciPy 的 MATLAB 文件输入输出模块。
import scipy.io
# PyTorch 神经网络模块。
import torch.nn
# PyTorch 常用函数模块。
import torch.nn.functional
# SciPy 插值模块。
import scipy.interpolate
# PyTorch 数据工具模块。
import torch.utils
# PyTorch 数据集/加载器模块。
import torch.utils.data
# 数据加载器类型。
import torch.utils.data.DataLoader
# 张量数据集类型。
import torch.utils.data.TensorDataset
# SciPy 多维数组处理模块。
import scipy.ndimage
