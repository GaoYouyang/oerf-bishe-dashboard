# OERF N2：真实物理失配、数据合同与独立留出协议

日期：2026-07-18
当前机器状态：`N2_WAITING_FOR_LAB_INPUT`；资料齐备度 `0/7`
证据等级：`CONTRACT_READINESS_ONLY_NO_RECONSTRUCTION_NO_MODEL_RESULT`

## 0. 先说结论

N1.9 已经关闭在同一 opened synthetic development 上继续拼 rank-6 basis 的路线。N2 不先写
DeepONet、FNO 或新网络，而先确定：

> 真实 OERF BOST 的主要误差到底来自有限孔径、曲线光线、标定漂移、位移提取，还是离散化？

这五种误差共享 raw images、geometry、mask、forward 和 split 等基础设施，但所需的额外对照数据、
forward fidelity 与主终点不同。若不先分清，网络可能只是在拟合某个处理链的偏差，held-out
reprojection 也可能因为同一个错误 forward 而“自洽”。因此本轮新增：

1. 可移植 JSON schema：`data_templates/oerf_n2_real_bost_contract.schema.json`；
2. 给师兄填写的空合同：`data_templates/oerf_n2_lab_intake.placeholder.json`；
3. fail-closed 验证器：`site_tools/validate_oerf_n2_real_bost_contract.py`；
4. 不含路径和原始 ID 的公开 readiness 摘要：
   `docs/oerf_n2_contract_readiness_public_summary.json`。

当前占位合同的资料齐备度是 `0/7`，**不授权预注册、不授权训练、不授权打开 audit，也不授权任何算法
主张**。这不是算法成绩或工作失败，而是把“缺什么”变成了机器可检查的清单。

## 1. 四类成像/观测失配的原始研究与权威综述依据，外加离散化控制

### 1.1 位移提取不是无害预处理

[Raffel 的 BOS 开放综述（权威综述）](https://doi.org/10.1007/s00348-015-1927-5)说明 BOS 的直接观测是由
背景相关得到的位移，并把它与沿光路积分的密度梯度联系起来；背景尺寸、照明、窗口和景深共同限制
可恢复尺度。因此“位移场”本身已经含有图像形成和估计算法误差，不是无噪声投影。

[Cakir 等的开放研究（同行评议原始研究）](https://doi.org/10.1007/s00348-022-03553-z)系统比较 optical flow 与
block matching，并改变背景纹理、算法参数和可压缩流结构来测量精度与可分辨范围。它支持 N2 把
`displacement_method`、背景 pattern、raw reference/flow-on 和 confidence 一起保留，而不是只收一张
处理后的 `.npy`。

[动态背景研究（同行评议原始研究）](https://doi.org/10.1007/s00348-021-03285-6)进一步表明，单一随机背景会产生与
pattern 相关的系统误差；多个独立背景得到的位移用 median 聚合，尤其有利于保留位移阶跃边缘。
这与“同一背景反复拍 flow-off”不是一回事：同背景 repeats 主要估计时间噪声、环境波动和慢漂移；
多个独立背景或受控背景位移才用于鉴别并削弱 pattern-dependent systematic error。动态背景论文不
支持“固定背景重复本身就能暴露 pattern bias”。

**对项目的直接约束：** 若主失配选 `displacement_extraction`，每个 audit view 都要保留 raw pair，
至少冻结两种位移方法；同一固定条件保留 flow-off repeats，若要研究背景依赖偏差则另保留多个独立
背景。两种证据在合同中分账，并在不看三维结果时先比较位移一致性。

### 1.2 有限孔径会把 sharp front 变成 forward-model mismatch

[Molnar 等的 cone-ray 论文开放预印本（arXiv 原始研究稿）](https://arxiv.org/html/2402.15954)明确指出，真实相机的有限孔径
接收一束而非一条光线；不同子光线穿过不同折射率梯度，flow-on 图像由此出现景深模糊。论文在
`f/22` 到 `f/4` 的模拟和实验条件下比较 thin-ray 与 cone-ray，并报告 pinhole 重建随孔径增大而平滑
shock，而 cone-ray 重建更一致。作者也强调孔径越大，逆问题越模糊、噪声放大越严重，cone-ray 并不
自动消除病态性。

**论文直接支持：** thin-ray 与有限孔径成像会产生可识别的 forward mismatch，且孔径越大越不能忽略。

**本项目为因果识别另设的工程门：** 只有拿到同一光学通道、同一几何下至少两个真实 f-number、
焦面/孔径信息与 cone/PSF 高保真 forward，才可把 `finite_aperture` 设为 primary mismatch。只有一个
固定 f-number 时，可以做 reconstruction，不能声称跨孔径泛化；不同相机镜头也不能直接写成受控
aperture sweep。

### 1.3 标定状态必须可追踪；漂移鉴别是本项目门

[PSU 开放 BOST 数据论文（arXiv 原始研究稿）](https://arxiv.org/html/2508.17120)公开了一个很清楚的标定范式：七相机、
十个旋转工况形成 70 views，标定板从多个位置采集；先独立求每台相机参数，再在共同全局坐标中联合
优化外参。论文还把未参与重建的 view 用作 validation reprojection，并保留 mask。

它支持“camera angle 不是充分几何”，并示范完整标定、global frame、mask 与留出视角的记录方式；
**它没有研究 calibration drift，也没有比较两个标定版本或两个独立 session。**N2 因而保留
calibration ID/version、内外参或 ray bundle、标定重投影误差、global frame、sensor-to-view mapping
与 session，先保证标定状态可追踪。

**本项目为区分真实漂移与人为扰动另设的鉴别门：** 若主失配选 `calibration_drift`，至少比较两个
时间分离的标定版本或 session；在旧 calibration 上训练、对同帧轻微扰动只算 synthetic sensitivity，
不算真实 drift 证据。

### 1.4 曲线 ray 与 NeRIF 的 straight-ray inversion 要区分

[NeRIF 开放预印本（arXiv 原始研究稿）](https://arxiv.org/html/2409.14722v2)把 neural field 输出的折射率与梯度沿相机 ray
积分，实验中由九个光纤输入端形成九路投影，并使用相机标定和 DeepFlow 位移；其数值数据生成附录
使用四阶 Runge-Kutta 追迹弯曲光线，而反演正文以反向采样的标称标定路径计算位移。由算法描述以及
作者把 nonlinear ray tracing 列为可扩展项，可推断当前 inversion 没有完成同等级曲线 ray 建模；
这是方法描述支持的审慎推断，**不是作者已经完成的 ray-bending 消融结论**。这个区别给 N2 一个问题：

> 用高保真曲线 ray 生成/观测的数据，若由较低保真 straight-ray inversion 重建，误差能否被一个
> 只读 geometry、residual 和 calibration 的算子校正，同时保持 held-out measurement consistency？

**对项目的直接约束：** 若主失配选 `ray_bending`，必须有 curved-ray forward 或可审计 JVP/VJP，
并在同一个 linearization context 下做 finite-difference test。只有“NeRIF 用了 ray”这句话不够。

### 1.5 离散化是控制分支，不与前四类文献证据等量齐观

NeRIF 对 voxel/grid 表示带来的离散化限制提供了动机，但当前来源不足以证明某个 multi-resolution
correction 已具有创新性或应成为首选算法。N2 把 `discretization` 保留为 inverse-crime 与网格收敛
控制分支；若收益随网格加密消失，应解释为离散误差，不包装成新的物理模型。

## 2. N2 为什么固定为七个门

| gate | 必须证明什么 | 当前占位状态 | 不通过时禁止什么 |
|---|---|---:|---|
| identity + units | case/run/session/geometry、来源 manifest、场变量、单位、轴顺序、网格、米制边界、support | 待实验室提供 | 禁止拼接或可视化后猜轴 |
| observation + geometry | raw/位移、mask、confidence、标定版本、ray 或内外参 | 待实验室提供 | 禁止训练固定 sensor tensor |
| operator + adjoint | 线性 `A/A^T` 或非线性 `F` 的 `JVP/VJP`、点积、有限差分、单位与 support | 待实验室提供 | 禁止同预算 solver 比较 |
| physical mismatch | 只冻结一个主失配，并有对应成对物理证据 | 待实验室提供 | 禁止给普通网络包装物理故事 |
| independent split | 按 view/sensor/run/session/geometry 分割，audit SHA-256 锁定 | 待实验室提供 | 禁止“fresh/OOD”措辞 |
| endpoint legality | 有真值才报 field/H1；无真值以 audit reprojection/PIV/时间稳定为主 | 待实验室提供 | 禁止用重投影冒充唯一 3D 真值 |
| permissions + claims | 本机保存/训练、组会/论文/公开边界逐项确认 | 待实验室提供 | 禁止把组内文件放 Git |

冻结数值门为：

- adjoint dot-product relative error `<= 1e-6`；
- nonlinear JVP finite-difference relative error `<= 1e-4`；
- 每个固定条件 flow-off repeats `>= 50`，且有逐文件 manifest；
- audit split 必须 `locked=true`、`opened=false`，且 split SHA-256 已写入合同。

前两个是工程一致性门，不代表 forward 物理正确。`50` 是本项目根据前序 covariance acquisition
screen 固定的最低采集门，**不是从文献推导的普适样本量**；真实相机的慢漂移更强时应增加，而不能
事后降门。

## 3. 推荐的实验单位与拆分

### 3.1 不能把 pixel 或同一序列帧当独立样本

优先独立单位顺序：`session > run > geometry/condition > sensor/view > frame > pixel`。论文统计至少按
run/session 聚合；像素只用于计算一个 unit 内的损失，不能把百万 pixel 写成百万实验样本。

### 3.2 最小四分法

1. `training_unit_ids`：模型参数拟合；
2. `tuning_unit_ids`：超参数和早停；
3. `validation_unit_ids`：冻结候选后的机制检查；
4. `audit_unit_ids`：方法、阈值、停止规则和图表模板全部冻结后才打开一次。

四个列表必须互斥。验证器会直接拒绝重叠；audit 打开后也不会自动给“成功授权”。若数据太少，宁可
只做 loader/adjoint/negative benchmark，也不把同一 run 随机拆成看似漂亮的 train/test。

## 4. 待导师确认后评估的候选问题，不构成创新性或可行性结论

| 师兄确认的主要困境 | 第一算法问题 | 必须比较的强基线 | 真实主终点 | 立即停止条件 |
|---|---|---|---|---|
| finite aperture | geometry/f-number-conditioned cone correction，保持 exact adjoint/fallback | thin-ray、cone-ray、UBOS/NIRT、NeRIF式 field | audit-view reprojection + front；有 CFD 才另报 field | 无成对 f-number 或无 cone/PSF forward |
| calibration drift | calibration-uncertainty-conditioned unrolling 或 robust data term | frozen calibration、joint refinement、Tikhonov/CGLS/NeRIF | 未见 session/view reprojection、front stability | 只有人为小扰动，没有独立 session |
| displacement extraction | raw-image/位移双域 uncertainty weighting，不直接生成场 | Horn-Schunck/DeepFlow/CC/WOF + 同一 inverse | held-out raw-image 或 displacement consistency | 没有 raw pair/confidence 或仅一种算法 |
| ray bending | straight-to-curved discrepancy operator，使用 JVP/VJP 和确定性回退 | straight ray、curved ray oracle、ReSeSOp/operator correction | held-out view + forward calls + front | 无 curved forward 或 FD/JVP 不过门 |
| discretization | multi-resolution residual correction，仅作控制线 | grid refinement、RBF/voxel、matched-call CGLS | synthetic truth +真实 reprojection分开 | 收益随网格加密消失，说明只是 inverse crime |

### 当前最容易形成可证伪实验的条件分支

- **若组内已有多 f-number/focus 数据：** 优先 finite-aperture；它有明确物理机制、可视化冲击强，且
  能自然连接 shock/flame front、NeRIF 与 operator correction。
- **若只有固定九视角但有多次标定/session：** 优先 calibration drift；工作量更适合本科，且真实
  holdout 更容易成立。
- **若只有 raw images + 一次标定：** 先做 displacement extraction；不要急着把误差全部归到 3D 网络。
- **若只有处理后位移 + 单一场：** 只做 loader、adjoint、强 baseline 与 held-out view benchmark；
  论文主张保持关闭。

这四条是条件决策，不是已选算法。师兄确认主失配后，再给模型命名和预注册。

## 5. 拿到数据后的 48 小时，不训练网络

### 0--6 小时：只读清单

1. 复制占位合同到 Git 忽略的私有数据根；
2. 填 case/run/session/geometry、权限和文件相对路径；
3. 计算文件 manifest、来源证明与可复算的 split SHA-256；
4. 运行 validator，公开页面只接脱敏 summary。

### 6--18 小时：几何和单位

1. 每个 view 显示 reference、flow-on、u/v displacement、mask、confidence；
2. 检查 pixel/metric/radian、世界轴、图像轴与 array axis；
3. 随机 ray 可视化 camera-to-background 路径；
4. 对 support 内外的 forward 输出做零值/有限性检查。

### 18--30 小时：operator

1. `dot(Fx, r)` 对 `dot(x, F^T r)`；
2. nonlinear 时做 `Jv` finite difference；
3. 检查 mask 后 row layout 和 u/v 顺序；
4. 记录调用数、ray samples、wall time 与 peak memory。

### 30--48 小时：只跑经典基线

1. matched-call CGLS/Landweber/Tikhonov；
2. fixed regularization 与 L-curve/validation 选择分账；
3. reconstruction views 优化，validation views 仅检查；
4. audit 仍封存；不训练 neural operator。

只有七门全过并提交预注册，才开始新算法。

## 6. 初学者学习路线：先能解释门，再能改模型

### 第一层：一天内能讲清

- BOS 观测的是背景位移，不是速度，也不是三维密度；
- 三维重建是从多视角路径积分反推场的病态逆问题；
- forward mismatch 是“求解器使用的物理模型”和真实成像链不同；
- held-out reprojection 是独立观测一致性，不是唯一三维真值。

### 第二层：一周内会手算和写测试

- 链式法则、Jacobian、JVP/VJP；
- adjoint 的内积定义与单位；
- CGLS 的 semi-convergence；
- pinhole ray、cone ray、curved ray 的区别；
- 为什么按 frame/pixel 随机拆分会泄漏 session 结构。

### 第三层：两周内会跑最小复现

- 从三台相机的 toy calibration 生成 ray；
- 对一个 3D Gaussian/界面 phantom 做 forward 与 adjoint；
- 人为加入一种且仅一种 mismatch；
- 画 field truth、重建、差值、held-out reprojection 和逐 case 尾部；
- 先证明经典基线公平，再接小网络。

## 7. 验证器怎样使用

```bash
cd /path/to/oerf-bishe-dashboard
.venv/bin/python -m pip install -r site_tools/requirements-n2.txt
.venv/bin/python site_tools/validate_oerf_n2_real_bost_contract.py \
  data_templates/oerf_n2_lab_intake.placeholder.json \
  --output docs/oerf_n2_contract_readiness_public_summary.json
```

当前预期输出：

```text
status: N2_WAITING_FOR_LAB_INPUT
passed gates: 0 / 7
may preregister: false
may train non-audit: false
may open audit: false
may claim success: false
```

占位合同会在写出报告后以退出码 `2` 结束；完整 fixture 默认退出 `3`，只有显式加
`--allow-fixture` 才返回 `0`；只有七门全过的真实 `DATASET_RECORD` 返回 `0`。因此 CI 不会把“报告成功
生成”误读成“合同已授权”。

验证器不检查私有文件是否真的存在；那一步必须在数据根内部由 loader 做 checksum/shape/dtype 审计。
它也不会输出 source path、dataset/case ID 或具体 permission values。即使未来七门全过，它只授权提交
预注册和使用 non-audit units，不会授权打开 audit 或宣称成功。

## 8. 可以直接发给何远哲师兄的消息

> 师兄，我把下一步收敛成了一个 N2 数据合同，不想先凭空写网络。能否请您先帮我确认一个最小 case：
> 每路 view 的 camera/ray、reference/flow-on 或 displacement、mask/confidence、标定版本、run/session，
> 以及现有 forward 是否能提供线性 A/A^T，或非线性 F 的 JVP/VJP？另外请只选一个组内最主要的真实误差：有限孔径、
> ray bending、标定漂移、位移提取或离散化。若允许，想永久封存一台 view 或一个 session，等方法和停止
> 规则冻结后再打开。第一周我只交 loader、单位/伴随测试、经典基线和 held-out 协议，不先训练大模型。

建议师兄只需回复：

1. 主失配选哪一个；
2. 可提供哪些最小字段；
3. 哪个单位可以永久留出；
4. 真实主终点认哪一个；
5. 本机保存、组会、毕业论文与公开派生图分别允许到哪一步。

## 9. 当前允许写进论文的内容

可以写：N1.9 负结果推动研究从 opened synthetic basis search 转向 real-data mismatch contract；公开文献
表明有限孔径、位移提取和相机标定都可能改变 BOST forward/observation；因此预先冻结数据、operator、
split、endpoint 与权限门。

不能写：已找到 OERF 的主要误差、已实现真实算法、已优于 NeRIF/TDBOST/FNO/DeepONet、50 repeats 是
普适统计结论、held-out reprojection 等于三维真值，或占位合同已经代表实验室数据。
