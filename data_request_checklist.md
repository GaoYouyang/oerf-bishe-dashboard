# 向何远哲请求数据与代码的清单

目标：把“师兄能不能给点数据”变成可操作请求。每一项都尽量问到文件、单位、格式和可公开边界。

## 最低摩擦请求

如果师兄时间很紧，先只要这一包：

- 一组九视角 BOST 样例：flow-off 背景图、flow-on 扰动图、mask、相机/内窥视角参数。
- 对应的位移场或位移估计结果，哪怕是中间产物。
- 一份参考重构结果：体素场或 NeRIF 输出切片。
- 一份说明：坐标轴、单位、视角编号、能否写入本科论文。

拿到这四类，就足够做“真实数据接口 + 重投影验证 + 参数扫描”的第一版。

## BOST 原始图像

- 背景参考图：每个视角的 flow-off / reference image。
- 扰动图：flow-on image，最好包含多帧。
- 视角编号：九视角如何对应真实空间角度或内窥通道。
- 图像格式：png/tif/raw，bit depth，是否已去畸变。
- 图像尺寸：像素宽高，裁剪区域，mask。
- 采集参数：帧率、曝光、背景图案尺度、物距、视场大小。
- 坐标单位：像素到 mm 的换算关系。

## 标定与几何

- 相机内参：焦距、主点、畸变参数。
- 外参或等效视角参数：每个视角的 ray origin / direction / projection matrix。
- 视角方向：是随机分布、近似均匀、共面/受限，还是由内窥束几何固定。
- 内窥系统说明：一相机九视角时，子图裁剪和视角映射方式。
- 标定板图像或标定结果文件。
- 重建区域：物理尺寸、网格范围、坐标轴方向。
- 是否使用直线光线近似，还是做了 ray tracing / bending correction。
- 是否存在透明窗口、端口、光纤束、遮挡或有效视场裁剪导致的投影偏差。

## 位移估计中间产物

- BOS 位移场：u、v 分量，单位是 pixel 还是 mm。
- 位移网格：vector spacing、窗口大小、overlap。
- 互相关或 optical flow 方法参数。
- 异常向量剔除和插值规则。
- 位移噪声估计或 flow-off 重复测量误差。
- 何种文件格式：mat、npz、h5、csv、tif stack。

## BOST / NeRIF 重构配置

- 传统 baseline：体素网格大小、正则化方式、迭代次数。
- 早期 CTC/BOST 指标：是否有 spatial resolution phantom、edge/blobs、MTF 或其他分辨率评价。
- NeRIF 网络：输入坐标、输出量、坐标编码、层数、宽度、激活函数。
- Loss：重投影 loss、梯度 loss、正则项、权重。
- 采样策略：ray sampling、batch size、训练步数、学习率。
- 归一化：折射率、坐标、位移、密度/温度转换常数。
- 指标：L2、MRE、correlation、SSIM、reprojection error、MTF。
- mask / ROI：哪些区域参与 loss，哪些区域只可视化；bad view 如何剔除或降权。
- 可公开的图：切片、体渲染、曲线、原始图是否能放论文。

## PIV-BOST 数据

- PIV 图像对：时间间隔、粒子图、laser sheet 位置。
- PIV 结果：原始速度场、未补偿速度场、补偿后速度场。
- 如果有：粒子位置偏差估计、image deformation / ray tracing 验证、position-error 到 velocity-error 的换算方式。
- 同步方式：PIV 帧和 BOST 帧如何对齐。
- PIV 标定：像素到物理长度、速度单位、平面位置。
- 折射补偿公式或已有脚本。
- 2D PIV 还是 stereo-PIV；本科阶段是否只做 2D。

## 4D BOST 数据

- 时序帧数、时间步长、帧率。
- 每帧视角数，是否所有视角严格同步。
- 真实参考结果或只做重投影验证。
- 张量 rank、压缩格式、内存瓶颈。
- 师兄最希望我帮忙扫哪些参数：rank、分辨率、噪声、帧数、采样策略。

## 代码与复现边界

- 是否能读已有代码？能否只看不复制？
- 是否有可跑的 demo 或配置文件？
- Python/MATLAB/C++ 哪个环境？
- 依赖库和数据路径如何组织？
- 哪些内容本科论文可写，哪些只能内部汇报？
- 论文图能否用真实数据，还是必须用合成/开源数据？

## 会面后必须确认

- 第一周我交付什么：数据读取、图像可视化、重投影、baseline、参数表中的哪一个？
- 成果评价由谁看：何远哲、蔡老师、郑老师或其他师兄师姐？
- 下次见面时间。
- 如果真实数据暂时不给，是否认可 synthetic + open-source BOS dataset 的保底路线？

## 拿到数据后的整理方式

把数据放进一个单独目录，并复制 manifest 模板：

```bash
cp data_templates/bost_sample_manifest.json data_real/oerf_bost_sample_YYYYMMDD/manifest.json
```

先检查字段：

```bash
python3 data_templates/validate_manifest.py data_real/oerf_bost_sample_YYYYMMDD/manifest.json --allow-missing
```

第一次整理真实数据时，至少要填完：

- `dataset_id`
- `views`
- `raw.flow_off`
- `raw.flow_on`
- `geometry.volume_bounds`
- `displacement.fields`
- `permissions`

这样你和师兄的沟通会从“我还缺数据”变成“我还缺 view geometry、位移单位、可公开边界这三项”。
