# LGWO-A24-L1 fit stack 独立审计

**审计身份：** 独立科学软件审计员<br>
**审计日期：** 2026-07-20<br>
**审计范围：**

- `demo_t16_operator/lgwo_a24_l1_arms.py`
- `demo_t16_operator/lgwo_a24_l1_losses.py`
- `demo_t16_operator/lgwo_a24_l1_fit_manifest.py`
- `demo_t16_operator/lgwo_a24_l1_checkpoint.py`
- 四个对应测试文件
- `docs/lgwo_a24_l1_fit_runner_contract_2026-07-20.md`

**执行边界：** 只读测试和静态阅读；没有生成或读取任何 `fit/early/route/fresh` 科学数据，也没有修改实现文件。

## 结论

当前四个模块的常规测试通过，但不能以当前状态授权第一次科学 fit。发现 7 项训练前必须关闭的 P1 风险，以及 3 项需要在论文比较前处理的 P2 风险。主要风险集中在：cache descriptor 不能证明其指向的文件真的无泄漏、manifest 可以被自洽篡改或覆盖、checkpoint 没有绑定真实模型状态和成本账本、以及固定方向消融的坐标含义可能随 support 改变。

**明确结论：无算法突破。** 本审计只评价可复现性、泄漏边界、预算记录和故障安全性；没有运行科学 fit，没有性能结果，没有真实 BOST 结论，也不能支持优于 DeepONet、FNO、iFNO 或其他基线的声明。

## 主审修复回执

本节记录审计之后的修复，不改写原始发现。协议已经在 scientific partition 物化数仍为 0 时修订为 `1.2`；当前 canonical config SHA256 为 `74ade98f298d6f03b54a76fae9de7423d48d6a3172c1e90bd71e5d00894cdeae`。

| 原发现 | 当前处置 | 训练门状态 |
| --- | --- | --- |
| P1-01 cache 只验 descriptor | 新增三根隔离 entity cache、typed inference-only scientific proposal builder、NPY/JSON dtype/shape/hash/symlink/path/unknown-file/overwrite 独立重读；每个 triplet 外绑 source/config/fit-manifest/entry hashes | 已实现，待提交后外部证据封存 |
| P1-02 manifest 覆盖与根路径绕过 | 根及子路径 sealed-token 检查、真实 commit/config hash、临时文件+fsync+原子 rename、目标存在 hard fail | 已关闭 |
| P1-03 顶层自洽篡改 | validator 重建并比较完整 expected manifest，不再只比较 entries | 已关闭 |
| P1-04 空 residual/projection | 所有 A/B residual、A delta/Ag 与 active support 强制 nonempty 且 ray shape 一致 | 已关闭 |
| P1-05 checkpoint 缺成本链 | epoch 0..30 typed ledger、每 epoch `192F/192AT + 8B-F + 8 optimizer steps`、metric-history 链和 route/early seal | 已关闭 |
| P1-06 checkpoint 未绑定真实 arm | exact arm registry、参数量、parameter/buffer state schema digest、全 float64 与 tensor hashes 双向绑定 | 已关闭 |
| P1-07 缺外部锁定根 | 31 节点独立 chain validator 已要求外部 expected contract；真实 Git commit、fit manifest 与独立 checksum evidence 必须在首次 fit 前最后封存 | 仍是最终提交门，未授权 fit |
| P2-01 optimizer 漂移 | AdamW 全量 group 字段及 gradient clip 固定并要求 exact equality | 已关闭 |
| P2-02 fixed-direction 坐标漂移 | binary support、恰好 1000 active voxels、flattened z-y-x 递增映射、batch index-map 一致性 | 已关闭 |
| P2-03 arm 容量不等 | 保留为输入/能力消融，参数量必须并列报告；不能解释成容量匹配的算法优越性 | 解释风险保留 |

截至修复回执，协议、arm、loss、manifest、entity cache、checkpoint、集成与 chain validator 的聚焦回归为 `123 passed`。这仍只是 pre-fit 基础设施证据，不是性能进展。

## 验证记录

执行命令：

```text
PYTHONPATH=. .venv/bin/pytest -q \
  demo_t16_operator/test_lgwo_a24_l1_arms.py \
  demo_t16_operator/test_lgwo_a24_l1_losses.py \
  demo_t16_operator/test_lgwo_a24_l1_fit_manifest.py \
  demo_t16_operator/test_lgwo_a24_l1_checkpoint.py
```

结果：`53 passed`。测试覆盖了正常构造、seed replay、support 边界、损失手算、cache 字段白名单、checkpoint round-trip、篡改和 pickle 拒绝；没有覆盖下面列出的关键负向路径。`.venv/bin/ruff` 在本机不存在，因此本次没有把 Ruff 结果记为通过。

## P1：训练前必须修复

### P1-01 Cache 白名单只检查 descriptor，不检查真实内容

**证据：** `lgwo_a24_l1_fit_manifest.py:54-70` 声明了 `_FORBIDDEN_PERSISTED_FRAGMENTS`，但该集合没有被任何校验函数使用；`250-268` 的 `_validate_cache_record` 只检查字段集合、schema 和“非空字符串引用”。例如，proposal descriptor 可以合法地把 `observation_uv_ref` 指向名为 `truth.npy` 的文件，也可以指向包含 truth、B 数据或 case-specific 元数据的文件。

**科学后果：** 目前只能证明“proposal descriptor 没有显式 truth 字段”，不能证明 proposal cache 的实际内容 truth-free。训练标签与 proposal 输入一旦通过文件引用间接重叠，fit 结果会发生信息泄漏，后续 no-leak 或泛化结论不可用。

**修复门槛：** proposal cache 必须由独立 validator 打开并逐数组检查：canonical path、禁止 symlink/path traversal、NPZ/NPY schema、dtype/shape、内容 hash 和字段级 allow-list；proposal 文件中不得出现 truth、clean、B、family、split、case 或 seed。validator 要从 manifest 外部读取期望 hash，并增加一个故意把 truth 文件塞进 proposal reference 的负向测试。training-label cache 可以含 truth，但必须与 proposal cache 使用不同 artifact root 和不同独立 hash。

### P1-02 fit manifest materializer 可覆盖旧结果，且 closed-path 检查可被根目录绕过

**证据：** `_reject_dedicated_closed_paths` 在 `301-312` 只扫描 `root.rglob("*")` 返回的子项，不检查 `root` 自身；当 `output_dir` 本身命名为 `route`、`early` 或 `fresh` 时，函数仍可能继续。`materialize_fit_manifest` 在 `360-365` 使用 `mkdir(..., exist_ok=True)` 和 `write_text`，既不拒绝已有 `fit_manifest.json`，也不是排他创建或原子写入。

**科学后果：** 一个错误的路径或中断后的半文件可以被后续调用覆盖；同一路径可能先后承载不同 source/config placeholder，从而破坏“第一次物化后不可变”的 provenance。即使 entries 相同，覆盖行为也会让目录中的时间、文件内容和外部缓存绑定失去确定性。

**修复门槛：** 对 `Path(output_dir).resolve()` 的每个路径组件（包括 root）执行 closed-token 检查；只允许显式的空 `fit` artifact root。要求 source commit 和 config SHA 都是外部锁定的真实值，禁止 placeholder。使用临时文件加 `fsync`、排他 rename，并在目标已存在、残留临时文件、或目录非空时 hard fail；写完后立即用独立 validator 重读并核对完整 hash。

### P1-03 `validate_fit_manifest` 没有比较完整的顶层合同

**证据：** `315-338` 只核对 self-claimed `manifest_sha256`、schema、partition、24 entries，并把 manifest 自己的 `source_commit` 和 `config_sha256` 重新传给 expected builder。它没有逐字段比较 `status`、`family_order`、`cluster_count`、`cases_per_cluster`、cache schema、closed-materialization 标志等顶层字段，也没有接收外部期望的 source/config hash。

**科学后果：** 调用者可以修改这些顶层值后重新计算 manifest 自身 hash，而校验仍可能通过。这样可以把“plan only”“cache separation”或“early/route/fresh closed”状态改写成看似合法的自洽文件，审计记录不再具有独立证据价值。

**修复门槛：** validator 必须接收外部锁定的 protocol hash、source commit 和 config hash；将完整的 expected manifest（除最终 hash 字段外）做 canonical equality comparison，并显式检查所有顶层字段。manifest hash 必须写入独立 commit/checksum evidence，不能只相信文件内部的 self-claim。

### P1-04 loss 允许空 residual 或空 projection 通过

**证据：** `lgwo_a24_l1_losses.py:35-44` 的 tensor 校验没有拒绝 zero-sized tensor；`225-242` 只检查 residual 的维度、shape 和 batch，未检查每个 case 的 measurement 元素数。于是 `clean_b_residual`、`a_delta_forward` 或 `a_g_forward` 可以是形如 `[B, 0]` 的空张量；`_per_case_l2` 会返回零，loss 仍然有限，最后的 `all finite` 检查也会通过。

**科学后果：** B evaluator 实际没有产出，或者 A delta/Ag 没有被计算时，loss 可能把缺失项当作零惩罚。这会直接造成 B-side 监督缺失和投影预算的虚假降低，属于错误科学结论和预算失真风险。

**修复门槛：** 对所有 residual、`a_delta_forward`、`a_g_forward` 强制 `numel > 0`，并校验 expected ray/voxel shape、case 数和 finite/nonempty support。更重要的是，runner 必须从调用账本产生这些 tensor，loss API 不能成为绕过一次 B forward 或一次 A projection 的入口；添加空 B、空 A delta、错误 ray count 的负向测试。

### P1-05 checkpoint 没有实现合同要求的 metric history、cost ledger 和 route sealed 状态

**证据：** 合同 `docs/lgwo_a24_l1_fit_runner_contract_2026-07-20.md:150-164` 要求 metric-history hash、cost ledger、route-still-sealed 标志以及每 epoch 的 A/B calls、fallback、nonfinite、clipping 和资源记录。当前 `save_checkpoint` 在 `demo_t16_operator/lgwo_a24_l1_checkpoint.py:353-361` 只保存单 epoch 的任意 `metrics` 映射、`metrics_sha256` 和 tensor hashes；`311-316` 仅检查 JSON 可序列化，没有要求字段存在、有限或与 optimizer/A/B 调用相符。

**科学后果：** checkpoint 可以合法存在，却无法证明该 epoch 支付了规定的 `24/24` A/AT 和 `1/0` B 调用、没有 fallback/nonfinite、没有跳步，也无法证明 route 当时仍封闭。后续结果表可能只保留模型权重而丢失预算证据。

**修复门槛：** 定义不可扩展的 typed metrics schema；拒绝 NaN/Inf；保存累计 metric-history hash、精确 cost ledger、route-sealed 布尔值和每 epoch 的 call/fallback/nonfinite/clipping/RSS/wall 字段。独立 validator 要按 30 epochs、8 clusters、3 families 和 5 arms x 3 seeds 重算上限，并拒绝缺字段、额外未知字段和计数不一致。

### P1-06 checkpoint 没有把真实 model state 绑定到 arm registry 和 parameter count

**证据：** `CheckpointContract.from_mapping` 在 `217-256` 只要求 `arm` 非空、`parameter_count >= 1`，不导入或核对 `LGWO_A24_L1_ARM_REGISTRY`。`save_checkpoint` 在 `317-360` 使用合同里的 parameter count 写入 manifest，却没有将 model tensor 的总 numel、state schema 或 dtype 与该值比较；`load_checkpoint` 在 `367-416` 只验证 tensor records 和 aggregate digest，不验证这些 tensor 是否是合同声明的 2729/2585/2225/681/1000 参数 arm。

**科学后果：** 任意非空、甚至只有一个参数的 tensor state 都可能被标成 `full`/2729 并通过当前 loader。这样 resume 或结果归档可能看似复现，实际运行的模型架构已改变，消融和参数预算结论不可信。

**修复门槛：** checkpoint contract 必须携带 registry schema、arm state schema digest、实际 parameter count 和模型 state digest；save 时由已审计的 model 对象计算并强制匹配，load 时重新检查每个 name/shape/dtype/device/finite 状态。`arm`、seed 和 parameter count 必须来自 registry，不能由调用者任意填写；添加“伪造 1 参数 state 冒充 full arm”的负向测试。

### P1-07 checkpoint provenance 只有内部自洽校验，没有外部锁定根

**证据：** `source_commit` 在 `227-240` 被转成任意字符串，没有 commit 格式或 git object 校验；config/fit/normalization/envelope 只要求是合法 SHA256。`load_checkpoint` 在 `376-386` 验证格式，但没有外部 expected contract；manifest 本身也没有独立 manifest digest/signature。parent hash 只在下一个 checkpoint 被字符串比较，不能证明 parent 文件仍存在或整个链条完整。

**科学后果：** 如果 manifest 与 state 一起被替换，loader 能接受一套新的、内部自洽但不是预注册协议的 provenance。单独把 checkpoint 交给审稿人时，无法判断它是否来自已经 push 的 source/config/fit manifest。

**修复门槛：** runner 必须把外部锁定的 protocol/source/config/fit/normalization/envelope digests 传入独立 validator；checkpoint manifest 需要独立 checksum evidence，或使用提交中固定的 attestation。resume 前验证 parent artifact 的真实文件和完整链；不能只验证 manifest 中自报的 hash 字符串。

## P2：必须在比较和论文解释前处理

### P2-01 optimizer contract 允许额外 hyperparameter 漂移

`save_checkpoint` 在 `319-327` 只检查合同中列出的 key，未拒绝 optimizer param group 中的额外字段。若合同只列出 AdamW、lr 和 weight decay，betas、eps、amsgrad、maximize 等仍可变化，而 checkpoint 会被视为可恢复。修复门槛是 canonical optimizer group exact equality，并固定参数 id 顺序、单 group/所有 group 数量和完整 optimizer schema。

### P2-02 fixed-direction arm 的“固定方向”不一定固定在同一坐标基

`FixedDirectionProposal.forward` 在 `lgwo_a24_l1_arms.py:364-378` 每次根据 support 的数值执行 `topk`，再把 1000 个参数散射到这些索引。support 数值或布局改变时，参数到 voxel 的对应关系也会改变；平值 support 的 tie 行为还使这个消融不再是一个预先冻结的固定方向。这样得到的 fixed-direction margin 可能混合了 support 重排效应。

修复门槛是从 fit manifest 固定 support index map 和 digest，要求 binary support 与 canonical voxel order；forward 只能使用该冻结 index map，并增加跨 cluster/rig 改变 support 后的坐标一致性测试。

### P2-03 各 arm 参数量不同，不能直接解释为模型优越性

registry 明确使用 `2729`、`2585`、`2225`、`681` 和 `1000` 个参数。这个设置适合做输入消融，但不是容量匹配的模型比较。若后续把 full 的结果与 `g_only` 或 fixed direction 直接写成“算法更好/更差”，结论会同时反映容量、输入和结构差异。

修复门槛是增加参数量匹配的结构对照，或把这组结果严格限定为 capability/ablation evidence；论文表格同时报告参数量、训练时间、A/B 调用和每个 seed 的全量结果，禁止只挑最佳 arm/seed。

## 已确认没有问题的部分

以下内容在本次测试范围内没有发现与合同直接冲突的缺陷：

- arm 构造使用 CPU/float64 和隔离 RNG；注册 arm 的参数量、support 输出边界和输入屏蔽测试通过。
- loss 当前确实使用 `support > 0.5`、support-masked field relative-L2、物理 spacing 的 x/y/z forward difference 和 active-edge 规则；没有把这些实现误报为问题。
- checkpoint 使用 tensor-only NPZ/safetensors 路径，NPZ 读取明确 `allow_pickle=False`，常规 round-trip、篡改和连续 epoch 测试通过。
- fit manifest 的 8 clusters x 3 families、A/B pair contract、fit-only partition 和基础 deterministic replay 测试通过。

这些通过项只说明当前测试和局部合同实现成立，不等于训练成功、泛化成立或真实 OERF 数据上的物理有效性。

## 解封建议

在关闭全部 P1 前，不应 materialize 科学 fit case，也不应生成 early、route 或 fresh 结果。修复后至少需要重新运行独立 validator，并新增上述负向测试；随后才可更新 protocol 1.2 的 canonical hash、提交源码和测试，再按合同开启 fit-only。即便所有门通过，第一轮输出仍只能称为预注册 synthetic fit evidence，不能称为算法突破或论文结论。
