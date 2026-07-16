# 【文件作用】调用已经编译好的 proj 扩展，生成或处理 BOS 投影数据。
# 【注意】真正的 proj 实现不在可读的 Python 源码里，而在同目录的 Linux .so 二进制文件中。

# 导入 PyTorch。原代码保留了此导入，虽然本文件没有直接调用 torch。
import torch

# 从编译扩展 proj 中导入同名函数 proj。
from proj import proj

# 只有直接运行 python projget.py 时，下面的代码才会执行。
if __name__ == '__main__':
    # 以喷雾数据模式运行投影/数据生成程序。
    proj(isspray=True)
    # 师兄原注释的意思：
    # 若测试 spray 案例，应把 isspray 设为 True，并确保 BOS_dataset.py 指向喷雾数据。
    # 完整喷雾原始数据因为体积较大，没有全部放进当前代码仓库。
