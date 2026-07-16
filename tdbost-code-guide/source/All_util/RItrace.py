"""折射率场中的光线追迹工具。

【主线】这个文件做两件事：
1. 用“中心差分”近似计算折射率的空间梯度；
2. 根据折射率及其梯度，逐步更新光线的位置和方向。

这里只增加中文导读注释，保留原代码的计算公式、变量名和执行顺序。
"""

# PyTorch：用张量表示一批光线，并可在 GPU 上并行计算。
import torch

# 【注意】如果有 CUDA GPU 就选第 0 块 GPU，否则使用 CPU。
# 这个全局 device 主要被 compute_gradientMLP 用来放置差分步长张量。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 【暂时不用深究】禁止 PyTorch 为这个函数建立反向传播计算图，可减少内存占用。
@torch.no_grad()
def compute_gradient(func, x, y,z,h = 0.001, *args):
    """
    【主线】用中心差分计算一个标量场在 x、y、z 三个方向的一阶偏导数。

    输入：
        func: 可调用的标量场，调用方式为 func(x, y, z, *args)；本项目中通常是折射率场 n(x,y,z)。
        x, y, z: 坐标张量，通常都是 [N]，N 是同时追迹的光线数。
        h: 差分步长，默认 0.001；步长越小不代表一定越准。
        *args: 原样转交给 func 的其他位置参数。

    输出：
        grad_x, grad_y, grad_z: 三个方向的梯度，形状通常为 [N]。
        N: func 在原坐标处的场值，形状通常为 [N]。

    物理意义：折射率梯度指向折射率增长最快的方向，它决定光线在非均匀介质中如何弯曲。
    """
    # 【主线】中心差分公式：f'(x) ≈ [f(x+h)-f(x-h)]/(2h)。
    # 只改变 x，y、z 保持不变，得到 ∂func/∂x。
    grad_x = (func(x+h, y,z, *args) - func(x - h, y,z, *args)) / (2 * h)
    # 只改变 y，x、z 保持不变，得到 ∂func/∂y。
    grad_y = (func(x, y+h,z, *args) - func(x, y - h,z, *args)) / (2 * h)
    # 只改变 z，x、y 保持不变，得到 ∂func/∂z。
    grad_z = (func(x, y ,z+ h, *args) - func(x, y,z - h, *args)) / (2 * h)
    # 再计算未偏移的原坐标处的场值 n；原代码将它命名为 N。
    N=func(x, y ,z, *args)
    # 按 x、y、z 梯度和场值的顺序返回，供光线更新公式使用。
    return grad_x, grad_y,grad_z,N


def compute_gradientMLP(func, points, h=0.001, *args):
    """
    【主线】对以 MLP（多层感知机）表示的场函数做中心差分。

    输入：
        func: 接收 [N, 3] 坐标张量的神经网络/场函数。
        points: [N, 3]，每一行是一个 (x,y,z) 点。
        h: 中心差分步长。
        *args: 原样传给 func 的附加参数。

    输出：
        grad_x, grad_y, grad_z: 通常为 [N, 1] 或 [N]，具体取决于 func 的输出形状。
        N: func(points) 的场值。

    与 compute_gradient 的区别：这里的 x/y/z 合在一个 [N,3] 张量中，更符合 MLP 的输入格式。
    """
    # 【暂时不用深究】view(1,3) 把向量变成一行，PyTorch 会把它广播到所有 N 个点。
    # x 方向偏移量 (h,0,0)，并放到前面选定的 CPU/GPU 设备上。
    h_x = torch.tensor([h, 0.0, 0.0]).view(1, 3).to(device)
    # y 方向偏移量 (0,h,0)。
    h_y = torch.tensor([0.0, h, 0.0]).view(1, 3).to(device)
    # z 方向偏移量 (0,0,h)。
    h_z = torch.tensor([0.0, 0.0, h]).view(1, 3).to(device)

    # 对全部 N 个点同时用中心差分计算 x 偏导。
    grad_x = (func(points + h_x, *args) - func(points - h_x, *args)) / (2 * h)
    # 对全部点计算 y 偏导。
    grad_y = (func(points + h_y, *args) - func(points - h_y, *args)) / (2 * h)
    # 对全部点计算 z 偏导。
    grad_z = (func(points + h_z, *args) - func(points - h_z, *args)) / (2 * h)

    # 计算原始坐标处的场值。
    N = func(points, *args)

    # 把三个梯度分量和场值一起交给光线追迹主函数。
    return grad_x, grad_y, grad_z, N


# 【主线】下面是本文件的核心：将一批光线沿折射率场向前积分。
def TracingRI(start_points,directions,refractive_index_field,steps=1000,dd=0.001,ddt=0.1,mlp=False):
    """
    【主线】在三维折射率场中追迹一批光线。

    输入：
        start_points: [N, 3]，N 条光线的起点，每行是 (x,y,z)。
        directions: [N, 3]，每条光线的初始方向向量。
        refractive_index_field: 折射率函数。mlp=False 时分别接收 x,y,z；mlp=True 时接收 [N,3]。
        steps: 积分步数，默认 1000。
        dd: 计算折射率梯度时的基础差分尺度，实际传入上述函数的是 dd/0.01。
        ddt: 光线积分的时间/弧长缩放因子。
        mlp: 是否按 MLP 的 [N,3] 输入方式调用折射率场。

    输出：
        points: [steps*N, 3]，把每一步的 N 个光线点沿第 0 维串接。
        current_points: [N, 3]，最后一步的光线位置。
        current_directions: [N, 3]，最后一步使用的位移/方向量。

    物理意义：折射率不均匀时，光线不再是直线；本函数用数值积分近似它的弯曲路径。
    """
    # clone() 复制起点，避免在函数内就地改掉调用者传入的 start_points。
    current_points = start_points.clone()
    # 同理，复制初始方向。
    current_directions = directions.clone()
    # points 将保存每一步的位置，用于返回完整光路。
    points=[]

    def D(R):
        """
        【主线】计算积分方程中的驱动项 D(R)=n(R)·∇n(R)。

        输入 R: [N,3] 的空间坐标。
        输出 D: [N,3]，折射率与其梯度的乘积。
        这是 TracingRI 内部函数，会自动使用外层的 mlp、dd 和 refractive_index_field。
        """
        # MLP 场函数直接接收 [N,3] 点集。
        if mlp:
            # 【注意】原代码将 dd/0.01 作为中心差分步长。
            dnx1,dny1,dnz1,n1=compute_gradientMLP(refractive_index_field, R,dd/0.01)
        # 非 MLP 场函数需要把三列坐标拆成 x、y、z 三个张量。
        else:
            # unbind(1) 沿第 1 维拆分：[N,3] -> 三个 [N]。
            ox1, oy1 ,oz1= R.unbind(1)
            # 对拆分后的坐标计算折射率梯度和折射率值。
            dnx1,dny1,dnz1,n1=compute_gradient(refractive_index_field, ox1, oy1,oz1,dd/0.01)
        # 将三个偏导数按最后一维合并：通常 [N]x3 -> [N,3]。
        dn1=torch.stack((dnx1,dny1,dnz1),1)
        # 去掉长度为 1 的维度，兼容场函数返回 [N,1] 的情况。
        dn1=dn1.squeeze()
        # 同样去掉折射率张量中的单例维度。
        n1=n1.squeeze()
        #print(dn1.shape,n1.shape)
        # unsqueeze(1) 把 [N] 变成 [N,1]，expand(-1,3) 复用为 [N,3]，以便与梯度逐元素相乘。
        D=n1.unsqueeze(1).expand(-1,3)*dn1
        #print(dn1)
        # 返回 n∇n，它控制光线方向的变化。
        return D

    # 【暂时不用深究】下面的三引号内容是旧调试代码，Python 会把它当成未使用的字符串，不执行。
    '''dnx,dny,dnz,_=compute_gradient(refractive_index_field, ox, oy,oz)
    dn=torch.stack((dnx,dny,dnz),1)'''
    # 先计算起点处的折射率 n。
    if mlp:
        # MLP 方式：直接传入 [N,3]。
        n = refractive_index_field(current_points)
    else:
        # 普通方式：拆成三个 [N] 坐标分量。
        ox, oy ,oz= current_points.unbind(1)
        # 用拆分后的 x、y、z 调用折射率函数。
        n = refractive_index_field(ox, oy,oz)
    # 计算每条初始方向向量的模（长度），结果形状为 [N]。
    kk=torch.norm(current_directions, dim=-1)
    #print(current_directions)
    # 【主线】构造光线动量样的量 T=n·direction/|direction|，形状 [N,3]。
    T=n.squeeze().unsqueeze(1)*current_directions/kk.squeeze().unsqueeze(1)#.expand(-1,3)
    #print(T)
    #ABSDB=0
    # 按指定步数逐步积分光线路径。i 是当前步号，从 0 到 steps-1。
    for i in range(steps):
        # 重新计算当前位置的折射率，因为光线已在上一步移动。
        #print(T)
        #D=n.unsqueeze(1).expand(-1,3)*dn
        if mlp:
            # MLP 场函数的调用方式。
            n = refractive_index_field(current_points)
        else:
            # 将当前位置拆成三个坐标分量。
            ox, oy ,oz= current_points.unbind(1)
            # 普通场函数的调用方式。
            n = refractive_index_field(ox, oy,oz)
        # 【注意】步长 t 与当前折射率成反比；dd/n 先决定每条光线的基础步长。
        t=ddt*torch.ones_like(dd/n)#.to(device)#步长不能乱设
        # 将步长整理为 [N,1]，便于与 [N,3] 的向量广播相乘。
        t=t.squeeze().unsqueeze(1)#.expand(-1,3)
        # R 是本步积分开始时的光线位置，形状 [N,3]。
        R=current_points
        #print(torch.mean(T))
        # 【暂时不用深究】A、B、C 是多阶显式积分的中间斜率，用多次 D(·) 评估提高路径近似精度。
        A=t*D(R)
        # 在由 T 和 A 预测的中间位置评估驱动项。
        B=t*D(R+T*t/2+A*t/8)
        # 在由 T 和 B 预测的步末位置评估驱动项。
        C=t*D(R+T*t+t*B/2)
        # 组合 T、A、B，得到本步的位移/方向更新量。
        current_directions=t*(T+1/6*(A+2*B))
        # 用 A、B、C 的加权平均更新 T。
        T=T+1/6*(A+4*B+C)
        #ABSDB=ABSDB+torch.norm(1/6*(A+4*B+C))

        # 【暂时不用深究】三引号中是一个替代积分公式，原代码不执行它。
        '''
        current_directions=1/6*h*(T+2*(T+h/2*D(R))+2*(T+h/2*D(R+h/2*T))+T+h*D(R+h/2*T+h**2/4*D(R)))
        T=T+1/6*h*(D(R)+2*D(R+h/2*T)+2*D(R+h/2*T+h**2/4*D(R))+D(R+h*T+h**2/2*D(R)))
        '''
        #current_directions=current_directions/kk.unsqueeze(1).expand(-1,3)
        # 【主线】用本步计算得到的位移量更新光线坐标。
        current_points=current_points+current_directions#/torch.norm(current_directions, p=2, dim=1).unsqueeze(1).expand(-1,3)
        #print((T+(A+2*B)/6))
        # 【暂时不用深究】下面数行是保留的旧实验写法，以 # 开头，因此不会执行。
        #print(grad_inputs)
        #td=torch.tensor([0.01])
        #current_directions=current_directions/torch.norm(current_directions, p=2, dim=1).unsqueeze(1)
        #current_points = current_points + current_directions#*td
        # 将当前步的 [N,3] 位置保存到列表，以便最后组成完整光路。
        points.append(current_points)
        #print(current_points)
    # 将 steps 个 [N,3] 张量沿第 0 维串接为 [steps*N,3]。
    points=torch.cat(points,0)
    #print(ABSDB)
    # 同时返回全部中间点、最终位置和最终方向/位移量。
    return points,current_points,current_directions
