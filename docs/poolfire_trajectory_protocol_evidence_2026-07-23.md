# PoolFire 多轨迹协议与首条新增数据接入

> 状态：`PASS_FROZEN_POOLFIRE_TRAJECTORY_PROTOCOL`
> 证据等级：`PUBLIC_DATASET_IDENTITY_AND_SPLIT_CONTRACT_ONLY`
> 算法突破：`false`
> 测试真值已打开：`false`

## 先说人话

此前的 warm-start 信号只来自 `p=14kw_size=03` 一条 PoolFire 轨迹中的不同
时间帧。它能证明代码闭环和一个值得继续检查的现象，却不能证明模型遇到另一团火时
仍然有效。

这次增量不训练新网络，而是先把之后所有模型都必须遵守的数据边界固定下来：

1. 官方 15 条轨迹按完整 trajectory 分成 11 train、2 validation、2 test；
2. 同一轨迹的相邻帧不能跨角色，不能把随机切帧写成泛化；
3. 两条 test 在模型、停止规则、指标和图表模板冻结前不得解压；
4. warm start 在部署时只能看 observation、冻结 geometry 和训练集统计量；
5. 最终比较必须让所有起点接同一个 CGLS/PCGLS，并在相同终点精度下比较成本。

## 唯一研究主线

```text
公开 PoolFire CFD rho
        ↓
独立高分辨率密度梯度代理正演
        ↓
多视角 observation b
        ↓
Zero / BP / ridge / 最小神经算子输出 x0
        ↓
完全相同的 CGLS/PCGLS 物理细化
        ↓
同终点精度下比较 A、A^T、端到端时间、内存和坏尾部
```

当前仍是 CFD morphology proxy，不是 calibrated BOST，也没有真实实验泛化结论。

## 官方轨迹角色

机器可读清单位于
`learning_labs/protocols/poolfire_trajectory_protocol_v1.json`。每个对象均固定官方
相对路径、字节数与 LFS SHA-256。

| 角色 | 完整轨迹 |
|---|---|
| train / initializer fit 候选 | `p=14kw_size=03`、`p=14kw_size=05`、`p=22kw_size=03`、`p=33kw_size=01`、`p=33kw_size=03`、`p=33kw_size=05`、`p=45kw_size=01`、`p=45kw_size=03`、`p=45kw_size=05`、`p=58kw_size=03`、`p=58kw_size=05` |
| validation / 模型与正则选择 | `p=14kw_size=01` |
| validation / correction budget 与停止规则 | `p=22kw_size=01` |
| untouched confirmatory test | `p=22kw_size=05`、`p=58kw_size=01` |

元数据对象固定为：

- 路径：`data/data.npz`
- 大小：`62,539 bytes`
- SHA-256：`8ecb161c1de3b038a8e46d35320a85f28054e4b3d7f785afe09fd000a3060738`
- 网格：`80 x 80 x 200`
- 时间：101 帧，标签从 30 到 32
- 原始变量标签：
  `CH4.npy, CO2.npy, H2O.npy, O2.npy, T.npy, rho.npy, Ux.npy, Uy.npy, Uz.npy`

真实本机元数据已通过组成员、变量顺序、网格和时间轴的一致性验证。

## 为什么不能叫 unseen-power OOD

两条 test 是训练集中已经出现过的功率值与火源尺寸值的新组合。例如，`22 kW` 和
`size=05` 都不是训练范围外的新参数。因此合规说法是：

> held-out power-size combinations within observed parameter values

不能写：

- unseen-power OOD；
- unseen-size OOD；
- geometry OOD；
- real-BOST generalization。

若以后要声称参数 OOD，需要另建超出训练功率或尺寸取值范围的数据；若要声称几何
泛化，需要改变并封存相机/射线几何。

## 已打开的旧轨迹怎样处理

`p=14kw_size=03` 已经用于第一轮开发：

- 0–48 的偶数帧：ridge fit；
- 55、60：ridge selection；
- 67、74：refinement depth validation；
- 80、85、90、95、100：已打开的 development evaluation。

这些角色永久保留为 development。尤其最后五帧不能在后续论文中重新命名为
fresh test。

## 测试集锁门

当前 acquisition 工具对 test 有三层约束：

1. test 下载必须显式使用 `--seal-test-only`；
2. test 原始文件名写成 `*.sealed.npz`，不会与 metadata trajectory stem 匹配；
3. 即使显式请求 `--extract`，工具也会在下载前拒绝。

底层 extractor 原来的 `--allow-test` 布尔开关也已删除。正式开 test 必须同时
提供 frozen trajectory protocol 与单轨迹 `TEST_RELEASE.json`；release 绑定模型、
预处理、观测生成器、几何、solver、停止规则、matched-accuracy tolerance、指标、
候选集、种子、排除规则和报告模板的 SHA。extractor 在读取 test stream **之前**
用排他创建写入 `OPENED` receipt，同一 release 不能再次消费。

这仍不是操作系统级不可读保险箱。它证明当前工具链 fail closed，不能声称任何人都
无法手工打开 test。

## 可续传接入工具

```bash
python site_tools/acquire_poolfire_trajectory.py \
  learning_labs/protocols/poolfire_trajectory_protocol_v1.json \
  'p=33kw_size=01' \
  /path/to/private/PoolFire \
  --metadata /path/to/data.npz \
  --extract
```

工具执行：

1. 先验证 15 轨迹协议；
2. 用 Python 外层重试逐次启动 `curl --http1.1 --continue-at -`，每次都按当前
   `.part` 长度续传；
3. 检查精确字节数与 SHA-256；
4. 仅对 train/validation 调用 bounded-memory `rho` extractor；
5. `rho.npy`、coords、times、manifest、checksums 和 READY 全部提交后才算完成；
6. 默认删除 6.4–6.7 GB 原始缓存，只保留约 493 MB 的 `rho` bundle 与 receipt。

最初实现使用了单个 `curl --retry` 进程。真实网络发生 HTTP/2
`CANCEL/SSL_ERROR_SYSCALL` 后，curl 的进程内重试把已增长的 `.part` 从头改写，
因此该实现已撤回。当前版本关闭 curl 内部 retry，强制 HTTP/1.1，并在每次外层重试
前后检查文件长度单调不减；连续五次无进度或文件缩小会 fail closed。

首条新增 train trajectory `p=33kw_size=01` 已完成。独立复核同时满足：

- receipt 中的协议 SHA、官方精确字节数和 source SHA 与冻结协议一致；
- manifest source SHA 与 receipt 一致；
- `rho.npy`、coords、times、manifest 四个 checksum 全部通过；
- READY 重新绑定 manifest 与 checksums；
- full-resolution `rho` 为 `(101,80,80,200)` float32，全部有限且严格为正；
- 原始 6.52 GB archive 和 partial 已删除，只保留派生 bundle 与 receipt；
- `test_truth_opened=false`。

因此当前新增正式状态为
`PASS_FIRST_ADDITIONAL_TRAIN_TRAJECTORY_READY`。第二条 train
`p=45kw_size=05` 按相同流程串行获取；它完成前仍只写 acquisition in progress。

## 后续判决顺序

1. 已完成首条新增 train trajectory 的 SHA、ZIP/NPY、full-resolution `rho` 和 READY；
2. 继续接入至少两条新增 train 与两条 validation trajectory；
3. 固定每条轨迹的抽帧规则、reference、normalization 和 proxy observation generator；
4. 先跑 Zero、normalized BP、CGLS/PCGLS、dual ridge；
5. 若 ridge 在未参与拟合的完整 trajectory 上仍有稳定 headroom，再训练最小
   FNO/UNO/3D U-Net；
6. 若经典 warm start 已解释全部收益，停止扩大网络；
7. test 只在模型、预算、阈值和报告模板全部冻结后打开一次。

## 当前允许和禁止的结论

允许：

- 官方 11/2/2 trajectory identity 与用途已冻结；
- 本机官方 metadata 与协议一致；
- 下载支持续传、字节数与 SHA 校验；
- 首条新增 train trajectory 已通过 receipt、bundle checksums 与 READY 复核；
- 当前工具拒绝 test extraction；
- 后续同精度成本比较有固定输入和账本边界。

禁止：

- 至少三条新增 train 与两条 validation 已全部接入；
- 跨 trajectory classical control 已完成；
- 神经算子已训练或优于 ridge/FNO/DeepONet；
- cross-trajectory 泛化已证明；
- unseen-power、unseen-size 或 geometry OOD 已证明；
- calibrated/real BOST 加速已证明；
- 算法突破或论文成功。

## 复现验证

```bash
python site_tools/validate_poolfire_trajectory_protocol.py \
  learning_labs/protocols/poolfire_trajectory_protocol_v1.json \
  --metadata /path/to/data.npz

python -m pytest -q \
  site_tools/test_validate_poolfire_trajectory_protocol.py \
  site_tools/test_acquire_poolfire_trajectory.py \
  site_tools/test_extract_poolfire_rho.py
```

当前定向结果为 `29 passed`。其中还单独覆盖“curl 零退出但文件短于冻结字节数”
不能提前成功。协议验证返回：

```text
PASS_FROZEN_POOLFIRE_TRAJECTORY_PROTOCOL
train=11, val=2, test=2
test_truth_opened=false
```

原始 CFD、私有本机路径、VPN 内容、账号和受限论文均不进入公开仓库。
