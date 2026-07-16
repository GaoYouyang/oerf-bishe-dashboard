"""将相机光线分块送入模型，组装 BOS 位移图，并评估/保存结果。

【主线】可以把本文件理解成「测试时的总调度员」：
1. ``OctreeRender_trilinear_fast`` 把大批光线切成小块，防止 GPU 显存不足；
2. ``evaluation`` 把模型输出重排成图像，与真实 BOS 位移比较，并写入磁盘。
"""

# os 用于创建评估结果目录和打开指标文本文件。
import os
# 【暂时不用深究】原代码保留的正则模块符号，本文件的激活路径未使用。
from re import U

# imageio 用于把预测位移图和真值图保存成 PNG。
import imageio
# NumPy 用于统计平均指标、数组拼接和图像数值转换。
import numpy as np
# PyTorch 负责张量运算，并提供关闭梯度的评估装饰器。
import torch
# MS-SSIM 从多个尺度比较两幅图的结构相似性。
from pytorch_msssim import ms_ssim as MS_SSIM
# tqdm 在循环测试帧时显示进度条。
from tqdm.auto import tqdm
# SciPy I/O 用于将位移数组保存为 MATLAB 可读的 .mat 文件。
import scipy.io as sio
# 项目自定义的 LPIPS 和 SSIM 评估函数。
from render.util.metric import rgb_lpips, rgb_ssim
# 【暂时不用深究】深度伪彩色函数为下方注释掉的路径保留。
from render.util.util import visualize_depth_numpy


def OctreeRender_trilinear_fast(
    rays,
    time,
    w2c,
    model,
    chunk=4096,
    N_samples=-1,
    ndc_ray=False,
    white_bg=True,
    is_train=False,
    device="cuda",
):
    """【主线】分块渲染所有光线，再把每块结果按原顺序拼回来。

    输入要点：
    - ``rays`` 通常是 ``[N_rays, 6]``，前 3 个数是光线起点，后 3 个数是方向；
    - ``time`` 通常是 ``[N_rays, 1]``，每条光线对应一个时刻；
    - ``w2c`` 通常是 ``[N_rays, 3, 3]``，表示世界坐标到相机坐标的旋转；
    - ``model`` 是已训练的 TDBOST 模型，``chunk`` 是每次送入 GPU 的光线数。

    返回 8 个沿第 0 维拼接的张量：无量纲 x/y 位移、x/y 位移、
    三个方向的中间导数/变化量，以及 ``rhobc`` 物理场中间量。
    每个返回张量的第 0 维都对应 ``N_rays``。
    """
    # 为 8 类输出分别创建空列表，先收集各个 chunk，最后再拼接。
    detaxs, detays, detaxNDs,detayNDs,dxds,dyds,dzds ,rhobcs= [], [], [], [],[], [], [],[]
    # 读取总光线数 N_rays，也就是 rays 第 0 维长度。
    N_rays_all = rays.shape[0]
    # 循环次数等于整块数，如有余数则多处理最后一个不满的 chunk。
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        # 按当前索引截取一块光线，并搬到 CPU/GPU ``device``。
        rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        # 用完全相同的索引截取时间，保证每条光线与时刻一一对应。
        time_chunk = time[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        # 同样截取这批光线对应的世界到相机旋转矩阵。
        w2c_chunk=w2c[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        # 【物理直觉】模型沿光线在重建场中采样/积分，预测背景点在 x、y 方向的偏移等量。
        detaxND,detayND,detax, detay,dxd,dyd,dzd,rhobc= model(
            # 当前光线块，形状约 ``[chunk_i, 6]``。
            rays_chunk,
            # 当前时间块，形状约 ``[chunk_i, 1]``。
            time_chunk,
            # 当前坐标旋转块，形状约 ``[chunk_i, 3, 3]``。
            w2c_chunk,
            # 把上层设定原样传给模型，决定是否启用训练时行为。
            is_train=is_train,
            # 告诉模型背景是否按白色处理。
            white_bg=white_bg,
            # 告诉模型光线是否已经处于 NDC 坐标。
            ndc_ray=ndc_ray,
            # 指定每条光线的采样点数；-1 通常表示由模型自行决定。
            N_samples=N_samples,
        )
        #print(dxyzd.shape)
        # 保存当前 chunk 的无量纲 x 位移，暂不在循环里拼接可减少频繁内存复制。
        detaxNDs.append(detaxND)
        # 保存当前 chunk 的无量纲 y 位移。
        detayNDs.append(detayND)
        # 保存当前 chunk 的 x 位移。
        detaxs.append(detax)
        # 保存当前 chunk 的 y 位移。
        detays.append(detay)
        # 保存 x 方向中间导数/变化量。
        dxds.append(dxd)
        # 保存 y 方向中间导数/变化量。
        dyds.append(dyd)
        # 保存 z 方向中间导数/变化量。
        dzds.append(dzd)
        # 保存当前 chunk 的 rhobc 中间物理量。
        rhobcs.append(rhobc)
        #rhobd.append(alpha_map)
        #z_vals.append(z_val_map)
        #TVs.append(densityTV.reshape(-1))
    # 按光线原先顺序沿第 0 维拼回 8 组完整结果。
    return (
        # 完整无量纲 x 位移，第 0 维长度为 N_rays_all。
        torch.cat(detaxNDs),
        # 完整无量纲 y 位移。
        torch.cat(detayNDs),
        # 完整 x 位移。
        torch.cat(detaxs),
        # 完整 y 位移。
        torch.cat(detays),
        # 完整 x 方向中间量。
        torch.cat(dxds),
        # 完整 y 方向中间量。
        torch.cat(dyds),
        # 完整 z 方向中间量。
        torch.cat(dzds),
        # 完整 rhobc 中间物理量。
        torch.cat(rhobcs),
        # torch.cat(rhobd),
        # #torch.cat(z_vals),
        # torch.cat(TVs),
        #None,
    )


@torch.no_grad()
def evaluation(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=5,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
):
    """【主线】在测试集上渲染 BOS x/y 位移，与真值比较并保存结果。

    ``test_dataset[idx]`` 应提供 ``rays``、``rgbs``（此项目中实为 2 通道位移真值）、
    ``time`` 和 ``w2c``。函数对每个测试帧计算 x/y 方向 PSNR，可选计算
    SSIM、MS-SSIM 和 LPIPS，并返回所有 PSNR 的 Python 列表。
    【注意】这个函数包含若干原仓库形状/指标假设；本注释版仅解释，不改变其行为。
    """
    # 分别创建指标和预测/真值位移图的收集列表。
    PSNRs, u_maps, v_maps, gt_u_maps, gt_v_maps = [], [], [], [], []
    # 额外指标也按测试帧逐个收集。
    msssims, ssims, l_alex, l_vgg = [], [], [], []
    # 创建总输出目录；exist_ok=True 表示已存在时不报错。
    os.makedirs(savePath, exist_ok=True)
    # 创建专门存放合并位移图的 deta 子目录。
    os.makedirs(savePath + "/deta", exist_ok=True)

    # 尝试清理 tqdm 残留的进度条实例，避免多次评估时终端显示混乱。
    try:
        # 直接清空 tqdm 内部记录的进度条集合。
        tqdm._instances.clear()
    # 【暂时不用深究】若 tqdm 版本不支持该内部属性，则忽略清理失败。
    except Exception:
        pass

    # 原代码将间隔固定为 1，因此每一个测试样本都会评估。
    img_eval_interval = 1 #if N_vis < 0 else max(len(test_dataset) // N_vis, 1)
    # 生成要评估的数据集索引列表：[0, 1, 2, ...]。
    idxs = list(range(0, len(test_dataset), img_eval_interval))
    # 打印实际要测试的索引，便于检查范围。
    print(idxs)
    # 带进度条地逐帧评估。
    for idx in tqdm(idxs):
        # 从 Dataset 中取出第 idx 个样本字典。
        data = test_dataset[idx]
        # 拆出光线、2 通道位移真值、时间和世界到相机旋转。
        samples, gt_d, sample_times,w2c = data["rays"], data["rgbs"], data["time"],data["w2c"]
        # 【暂时不用深究】为原评估框架保留的深度占位符，当前激活路径未使用。
        depth = None
        # 打印真值张量形状，零基础读者可用它检查「像素数 × 2位移分量」。
        print('gt_d.shape',gt_d.shape)
        # 从数据集取得两个图像尺寸；原代码在下方作为 H、W 使用。
        H, W = test_dataset.img_wh
        # 将可能带有图像维度的光线展平为 ``[N_rays, ray_features]``。
        rays = samples.view(-1, samples.shape[-1])
        # 同样将时间展平为 ``[N_rays, time_features]``。
        times = sample_times.view(-1, sample_times.shape[-1])
        # 打印关键形状和采样数，便于定位维度不匹配。
        print(rays.shape,times.shape,w2c.shape,N_samples)
        # 分块调用模型，返回所有光线的位移与中间物理量。
        detaxND, detayND,detax, detay,dxd1,dyd1,dxyzd1,_= OctreeRender_trilinear_fast(
            # 展平后的全部光线。
            rays,
            # 每条光线对应的时间。
            times,
            # 每条光线对应的坐标旋转。
            w2c,
            # 要评估的已训练模型。
            model,
            # 每次只处理 1000 条光线，以控制显存占用。
            chunk=1000,
            # 传递每条光线的采样点数。
            N_samples=N_samples,
            # 传递是否使用 NDC 光线。
            ndc_ray=ndc_ray,
            # 传递背景颜色设定。
            white_bg=white_bg,
            # 【注意】原代码在 evaluation 中仍向模型传入 is_train=True，注释版保持不变。
            is_train=True,
            # 指定张量运行设备。
            device=device
        )

        # 把展平的 x/y 位移预测重排为二维图，并搬回 CPU 便于评估/保存。
        u_map,v_map = (
            detaxND.reshape(H, W).cpu(),
            detayND.reshape(H, W).cpu(),
        )
        # 在函数内导入 skimage 的 PSNR，用于比较位移预测和真值。
        from skimage.metrics import peak_signal_noise_ratio as psnr
        # 只要测试数据集非空，就计算本帧指标。
        if len(test_dataset):
            # 取真值第 0 通道作为 x/u 方向位移，重排为 ``[H, W]``。
            gt_u = gt_d[...,0].view(H, W)#*2.1/3000/2.56*1e4
            # 取真值第 1 通道作为 y/v 方向位移。
            gt_v = gt_d[...,1].view(H, W)#*2.1/3000/2.56*1e4
            # 计算 u 预测与真值的均方误差；原代码保留此中间量但后面用 skimage 直接算 PSNR。
            loss = torch.mean((u_map - gt_u) ** 2)
            # 以当前 u 真值的极差作 data_range，计算 PSNR 并添加到列表。
            PSNRs.append(psnr(gt_u.numpy(), u_map.numpy(), data_range=gt_u.max() - gt_u.min()))#((20*np.log(gt_u.max())-10.0 * np.log(loss.item())) / np.log(10.0))
            # 重用变量 loss，改为计算 v 方向均方误差。
            loss = torch.mean((v_map - gt_v) ** 2)
            # 计算 v 方向 PSNR，因此 PSNRs 每个测试帧会追加两个数。
            PSNRs.append(psnr(gt_v.numpy(), v_map.numpy(), data_range=gt_v.max() - gt_v.min()))

            # 只有用户要求额外指标时，才计算较耗时的 SSIM/MS-SSIM/LPIPS。
            if compute_extra_metrics:
                # 计算 u 图 SSIM。【注意】rgb_ssim 要求 HWC 三通道，而此处上游形状为 [H,W]。
                ssim = rgb_ssim(u_map, gt_u, 1)
                # 将 u 图换为 MS-SSIM 期望的 NCHW 形状后计算多尺度结构相似性。
                ms_ssim = MS_SSIM(
                    u_map.permute(2, 0, 1).unsqueeze(0),
                    gt_u.permute(2, 0, 1).unsqueeze(0),
                    data_range=1,
                    size_average=True,
                )
                # 用 AlexNet 特征计算 u 图的 LPIPS 感知距离。
                l_a = rgb_lpips(gt_u.numpy(), u_map.numpy(), "alex", device)
                # 用 VGG 特征计算 v 图的 LPIPS 感知距离。
                l_v = rgb_lpips(gt_v.numpy(), v_map.numpy(), "vgg", device)
                # 保存本帧 SSIM。
                ssims.append(ssim)
                # 保存本帧 MS-SSIM。
                msssims.append(ms_ssim)
                # 保存本帧 AlexNet LPIPS。
                l_alex.append(l_a)
                # 保存本帧 VGG LPIPS。
                l_vgg.append(l_v)

        # 为下方保存步骤创建 u 真值别名，数据本身不变。
        gt_u_map =gt_u# 
        # 为 v 真值创建别名。
        gt_v_map =gt_v# )

        # 只在提供了输出路径时写入 PNG 和 MAT 文件。
        if savePath is not None:
            # 将 u 预测取绝对值、缩放至 8 位图像范围，保存为 PNG。
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}u.png", (abs(u_map.numpy())/2* 255).astype("uint8"))
            # 用同样的可视化尺度保存 u 真值。
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_ugt.png", (abs(gt_u_map.numpy())/2* 255).astype("uint8"))
            # 保存 v 预测图。
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}v.png", (abs(v_map.numpy())/2* 255).astype("uint8"))
            # 保存 v 真值图。
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_vgt.png", (abs(gt_v_map.numpy())/2* 255).astype("uint8"))
            # 保存为 .mat 文件
            # 生成带三位补零帧号的 MATLAB 文件名。
            mat_filename = f"{savePath}/{prefix}{idx:03d}.mat"
            # 将四个二维数组用明确键名写入 .mat，便于 MATLAB 后处理。
            sio.savemat(mat_filename, {
                'u_map': u_map,
                'gt_u_map': gt_u_map,
                'v_map': v_map,
                'gt_v_map': gt_v_map
            })
            # 水平拼接 u、v 预测图，得到一张宽度加倍的对照图。
            d_map = np.concatenate((u_map, v_map), axis=1)
            # 把拼接后的位移图缩放到 0～255，写入 deta 子目录。
            imageio.imwrite(f"{savePath}/deta/{prefix}{idx:03d}.png", (d_map/2* 255).astype("uint8"))

        # 在内存列表中保存 uint8 形式的 u 预测图。
        u_maps.append((abs(u_map.numpy())/2* 255).astype("uint8"))
        # 在内存列表中保存 uint8 形式的 v 预测图。
        v_maps.append((abs(v_map.numpy())/2* 255).astype("uint8"))
        # 【注意】原代码又向 v_maps 追加了原始张量，注释版保留这一行为不改行为。
        v_maps.append(v_map)
  

    # 只有实际收集到 PSNR 时才汇总并写指标文件。
    if PSNRs:
        # 把所有 u/v 的 PSNR 一起求算术平均。
        psnr = np.mean(np.asarray(PSNRs))
        # 如果已启用额外指标，则同时汇总 SSIM、MS-SSIM 和两种 LPIPS。
        if compute_extra_metrics:
            # 求所有帧 SSIM 的平均值。
            ssim = np.mean(np.asarray(ssims))
            # 求所有帧 MS-SSIM 的平均值。
            msssim = np.mean(np.asarray(msssims))
            # 求 AlexNet LPIPS 的平均值。
            l_a = np.mean(np.asarray(l_alex))
            # 求 VGG LPIPS 的平均值。
            l_v = np.mean(np.asarray(l_vgg))
            # 以写入模式创建/覆盖平均指标文件。
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                # 把所有平均指标写入第一行。
                f.write(
                    f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}\n"
                )
                # 在终端同步打印平均指标。
                print(
                    f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}\n"
                )
                # 遍历 PSNRs 列表，写入逐项结果。
                for i in range(len(PSNRs)):
                    # 【注意】PSNRs 每帧有 u/v 两项，而其他列表是每帧一项；保留原索引行为。
                    f.write(
                        f"Index {i}, PSNR: {PSNRs[i]}, SSIM: {ssims[i]}, MS-SSIM: {msssim}, LPIPS_a: {l_alex[i]}, LPIPS_v: {l_vgg[i]}\n"
                    )
        else:
            # 不计算额外指标时，只创建 PSNR 报告。
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                # 写入平均 PSNR。
                f.write(f"PSNR: {psnr} \n")
                # 在终端打印平均 PSNR。
                print(f"PSNR: {psnr} \n")
                # 遍历所有 u/v PSNR 项。
                for i in range(len(PSNRs)):
                    # 将当前索引和 PSNR 写入一行。
                    f.write(f"Index {i}, PSNR: {PSNRs[i]}\n")

    # 返回未求平均的所有 PSNR，供上层继续分析。
    return PSNRs


# @torch.no_grad()
# def evaluation_path(
#     test_dataset,
#     model,
#     cfg,
#     savePath=None,
#     N_vis=5,
#     prefix="",
#     N_samples=-1,
#     white_bg=False,
#     ndc_ray=False,
#     compute_extra_metrics=True,
#     device="cuda",
# ):
#     """
#     Evaluate the model on the valiation rays.
#     """
#     u_maps, v_maps = [], []
#     os.makedirs(savePath, exist_ok=True)
#     os.makedirs(savePath + "/deta", exist_ok=True)

#     try:
#         tqdm._instances.clear()
#     except Exception:
#         pass

#     near_far = test_dataset.near_far
#     val_rays, val_times = test_dataset.get_val_rays()

#     for idx in tqdm(range(val_times.shape[0])):
#         W, H = test_dataset.img_wh
#         rays = val_rays[idx]
#         time = val_times[idx]
#         time = time.expand(rays.shape[0], 1)
#         rgbd_map, _, depth_map, _, _ = OctreeRender_trilinear_fast(
#             rays,
#             time,
#             model,
#             chunk=8192,
#             N_samples=N_samples,
#             ndc_ray=ndc_ray,
#             white_bg=white_bg,
#             device=device,
#         )
#         rgb_map = rgb_map.clamp(0.0, 1.0)

#         rgb_map, depth_map = (
#             rgb_map.reshape(H, W, 3).cpu(),
#             depth_map.reshape(H, W).cpu(),
#         )

#         depth_map=depth_map.numpy()#, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

#         rgb_map = (rgb_map.numpy() * 255).astype("uint8")

#         rgb_maps.append(rgb_map)
#         depth_maps.append(depth_map)
#         if savePath is not None:
#             imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)
#             rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
#             imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)

#     imageio.mimwrite(
#         f"{savePath}/{prefix}video.mp4", np.stack(rgb_maps), fps=30, quality=8
#     )
#     imageio.mimwrite(
#         f"{savePath}/{prefix}depthvideo.mp4", np.stack(depth_maps), fps=30, quality=8
#     )

#     return 0
