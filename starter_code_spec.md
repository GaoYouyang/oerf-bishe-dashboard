# BOST / NeRIF Starter Code 规格

目标：第一周内跑通一个最小闭环，不追求论文级精度。

## 目录结构

```text
bost-nerif-bishe/
  data/
    synthetic/
    open_bos/
    oerf_sample/
  configs/
    demo_2d.json
    demo_3d.json
  src/
    phantom.py
    forward_bos.py
    measure_displacement.py
    voxel_baseline.py
    nerif_model.py
    train.py
    metrics.py
    visualize.py
  notebooks/
    01_phantom_and_forward.ipynb
    02_baseline_reconstruction.ipynb
    03_neural_field_demo.ipynb
  results/
  reports/
  README.md
```

## 最小模块

### `phantom.py`

功能：

- 生成 2D/3D Gaussian blobs。
- 生成火焰薄层或热羽流形状。
- 输出 `n_field.npy` 和 `grad_n.npy`。

最小函数：

```python
def make_gaussian_phantom(shape, centers, widths, amplitudes): ...
def gradient(field, spacing=(1, 1, 1)): ...
```

### `forward_bos.py`

功能：

- 定义多视角几何。
- 沿视线积分折射率梯度。
- 输出合成背景位移。

最小函数：

```python
def make_views(num_views, mode="circle"): ...
def project_gradient_to_displacement(grad_n, views, noise_std=0.0): ...
```

### `voxel_baseline.py`

功能：

- 提供传统重构 baseline。
- 初期可以用粗网格 Tikhonov 或 Landweber。

最小函数：

```python
def reconstruct_tikhonov(displacements, views, lambda_reg): ...
def reconstruct_landweber(displacements, views, steps, lr): ...
```

### `nerif_model.py`

功能：

- 坐标 MLP。
- Fourier encoding。
- 输出 `n` 或 `n + grad_n`。

最小函数：

```python
class CoordinateMLP(torch.nn.Module): ...
def fourier_encode(x, num_frequencies): ...
```

### `train.py`

功能：

- 训练 neural field。
- 记录 loss 和 checkpoint。

最小损失：

- displacement loss
- optional gradient consistency loss

### `metrics.py`

功能：

- RMSE
- correlation coefficient
- SSIM / PSNR
- re-projection error

### `visualize.py`

功能：

- 保存切片图。
- 保存误差热图。
- 保存重投影对比图。
- 保存参数扫描曲线。

## 第一周验收命令

```bash
python -m src.phantom --config configs/demo_3d.json
python -m src.forward_bos --config configs/demo_3d.json
python -m src.voxel_baseline --config configs/demo_3d.json
python -m src.train --config configs/demo_3d.json
python -m src.visualize --result results/demo_3d
```

## 第一周验收图

- `phantom_slice.png`
- `displacement_views.png`
- `baseline_reconstruction.png`
- `nerif_reconstruction.png`
- `error_map.png`
- `reprojection_error.png`
- `metrics.csv`

## 先不要做的事

- 不要一开始写完整真实相机模型。
- 不要一开始碰 stereo-PIV。
- 不要一开始承诺完整 4D BOST。
- 不要只追求 neural network loss，而忘记 re-projection 和物理量解释。

