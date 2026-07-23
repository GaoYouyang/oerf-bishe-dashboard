# E73 工件运行器与 Phase-0 前置审计

更新：2026-07-19
当前结论：`DEVELOPMENT_ONLY_CALLER_FEATURES_UNTRUSTED_NO_CANDIDATE_AUTHORIZATION`
Phase-0 结论：`PHASE0_PREFLIGHT_NO_GO_CONTRACT_INCOMPLETE`

这里的两个 `NO` 都不是“算法已经失败”。它们表示证据链还没有资格让学习策略替代固定的 `CGLS k=4`，也还不能启动 16 个 phantom 的正式计分。

## 1. 为什么要先做运行器

E73 核心已经能把 action × metric 的 harm 做联合上界，但旧接口仍允许调用方同时递交：

- feature vector；
- predicted harm；
- 一个自称匹配的 predictor hash。

字符串相等不能证明预测真的来自冻结模型。新运行器因此只允许固定 JSON ridge/affine 公式，由进程自己读取 predictor 并计算

\[
\widehat h_{a,m}(x)=b_{a,m}+\sum_f W_{a,m,f}
\frac{x_f-\mu_f}{s_f}.
\]

它不加载 pickle、joblib、Torch checkpoint、动态 module、URL 或任意代码。

## 2. 本轮已经闭合的边界

### 2.1 静态模型包与 deployment 分离

静态 bundle 只含：

1. `feature_contract.json`；
2. `predictor.json`；
3. `scale.json`；
4. `certificate.json`；
5. `support.json`；
6. `policy.json`。

它由独立的 manifest SHA-256 锚定，同时绑定当前 runner 和 certificate core 源码哈希。每个新 deployment 是另一个单独 JSON，并由第二个 SHA-256 锚定。更换 deployment 不得顺便更换 predictor、policy 或 calibration envelope。

### 2.2 固定私有根与无覆盖发布

命令行只允许在仓库忽略的 `private_library/e73_jgtce` 下寻找：

- `bundles/<bundle_id>/`；
- `deployments/<deployment_id>.json`；
- `ledger/`；
- `results/`。

接口只接收小写 ASCII ID 和两个预先知道的哈希，不接收任意输入/输出路径，也没有 `--replace`。目录逐层通过持有的 file descriptor 和 `O_NOFOLLOW` 打开；成员还检查 regular file、单一 hard-link、大小、读取前后 inode/mtime/ctime 以及原始字节哈希。一次 `run` 从读取 bundle、读取 deployment 到发布 ledger/result 全程持有同一个 private-root descriptor，不再分阶段按路径重开根目录。

结果先写 mode `0600` 临时文件并 `fsync`，再用独占 hard-link 发布。目标已存在时拒绝，永不覆盖。

### 2.3 声明物理单元消费账本

deployment 必须同时声明：

- dataset namespace；
- physical unit ID；
- source manifest SHA-256；
- independent unit kind。

四者组成 unit key。正式写结果前，runner 先独占发布消费 marker；同一声明单元即使换 deployment 文件名或 result 名也不能再次使用。这个账本能阻止**声明身份完全相同**的重复消费，但不能识别人为给同一物理 flow 改名，因此 acquisition manifest 的真实性仍需实验流程证明。

### 2.4 JSON 与模型攻击面

当前回归门拒绝：

- 重复 key、NaN、Infinity、`1e309`；
- bool 或数字字符串伪装成数组值；
- 超深、超大、超长 JSON；
- action/metric/feature 重排与 shape 漂移；
- 非正 scale；
- `../`、绝对成员路径、父目录 symlink、最终 symlink、hard-link、FIFO；
- fit/calibration 重叠、camera 冒充独立单位；
- deployment 注入 prediction、truth 或 claimed predictor hash；
- runner/core 源码在 bundle 冻结后变化；
- 即使攻击者改写 policy 并重算所有 JSON 哈希，也不能把 `cgls_k5` 等任意字符串重新命名为 fallback；代码常量把 fallback 字面钉死为 `cgls_k4`；
- 已存在的 ledger 或 result。

### 2.5 第二轮独立安全复审

独立只读复审最初发现一个合同级 P0：旧 parser 只要求 fallback 不在 candidate 列表内，因此一个“内部哈希完全自洽”的恶意 bundle 仍能发布任意 fallback 名称。当前修复同时做了三件事：

1. 新增代码常量 `FORCED_FALLBACK_ACTION = "cgls_k4"`；
2. policy fallback 不是 `cgls_k4` 或 candidate 列表含 `cgls_k4` 时直接拒绝；
3. 增加重算 policy 与 manifest 哈希后的攻击回归测试，并补测 runner/core 两个源码绑定以及单根 descriptor 事务。

修复后的独立窄复审判定为 `3 CLOSED / 0 PARTIAL / 0 OPEN`，没有新增 P0/P1；这里的三项只指固定 `k4`、单根 descriptor 与 runner/core 双源码哈希测试。这关闭的是“任意动作冒充 fallback”的 P0，不是 formal trust launcher。两个外部哈希仍由可信操作者从命令行提供，因此它们能检测普通漂移，却不是面向恶意启动器的签名信任根。ledger 也只能阻止声明身份相同的重复消费；真实 flow 改名、marker 已写而 result 发布失败、raw feature provenance 与下游 `k4` 数组一致性仍分别标为 partial/open。

## 3. 为什么当前仍强制回退

当前 deployment 仍直接含 precomputed `feature_vector`。runner 能证明 prediction 来自冻结 predictor，却不能证明 feature 真由原始 BOST 的 pose、ray coverage 和前四步 residual 按冻结公式算出。

因此即使内部 diagnostic 会选中某个 candidate，最终结果仍固定为：

```text
selected_action = cgls_k4
reason = caller_precomputed_features_not_authorized
candidate_authorization_enabled = false
```

输出不会发布 diagnostic candidate 名称。`cgls_k4` 不是从 policy 自由读取的“某个 fallback”，而是 runner 代码硬编码并复核的唯一基线。只有完成 raw-BOST/source-manifest extractor、其代码哈希和输入 provenance 后，才允许重新设计 candidate authorization。当前 runner 也只返回动作合同，尚未证明下游真正执行的 fallback 三维数组与保存的 `k4` **逐字节相同**；Phase-0 必须补这道门。

这里的“当前 runner”专指 **development selector runner**：它仍接收 caller-precomputed feature，所以永远只授权 `k4`。另一个独立的 **Phase-0 data-foundation runner** 已在 fixture 内从 `y=A(x_proxy)`、同一条 CGLS history 与 `Aᵀr4` 内生构造 feature，并逐字节密封 fallback。现在又有一个独立的 fixture-only 24-metric scorer，但它尚未与 private finalizer 集成，没有打开真实 harm 或 candidate action 权限。三个组件的信任边界不能混写。

## 4. Phase-0 的 16 个解析代理

机器合同见 `demo_t16_operator/configs/e73_phase0_schema_pilot_v1.json`。八个实际代码 family 各出现两次：

| Block A | Block B（反序） |
|---|---|
| plume | multi_plume |
| wavy_front | vortex_pair |
| thin_front | oblique_shock |
| double_front | annular_kernel |
| annular_kernel | double_front |
| oblique_shock | thin_front |
| vortex_pair | wavy_front |
| multi_plume | plume |

16 个 uint32 seed 由 `SeedSequence(2026071901).spawn(16)` 一次性生成并显式保存。它们只是确定性 analytic proxy，不是 16 个统计独立实验 flow。

### 几何和观测

- 使用本地 PSU B0 九视角真实 ray geometry，网格 `16³`，范围 `[-0.11, 0.11]³ m`；
- float64 CPU，外一层 voxel 为零；
- 总 active rays `10,628,822`，九个 view slot 永不按重复 camera ID 合并；
- synthetic `y=A(x_proxy)`，无噪声，明确标记 `INVERSE_CRIME_SCHEMA_ONLY`；
- 禁止打开 source cache 中的 `observations_uv.npy`。

原有 `PSUCompactCachedRayStore` 构造时会打开 observations。新增 `PSUCompactGeometryOnlyRayStore` 已把 observation loader、属性与 chunk 值全部关闭；可信 manifest digest 现在是必填项。cache root 使用目录 FD，每个 geometry `.npy` 通过 `O_NOFOLLOW` 只打开一次，并在同一个 FD 上解析 header、校验 SHA-256、建立 copy-on-write mmap；它还拒绝与 observation 相同 inode 的 hard-link 以及 manifest 嵌套字段漂移。host audit 从 gitignored 私有 attestation 取得 digest 后，在真实 `10,628,822`-ray cache 上完成 constructor、329 个 chunk 的全量遍历以及一次 matrix-free forward/adjoint；只观察到 manifest 与五类 geometry 文件，`observations_uv.npy` open count 为 `0`。这关闭的是本次已执行路径，不是对其他进程或恶意启动器的全局保证；正式 Phase-0 runner 仍须绑定这些源码与 attestation 并重复审计。

## 5. Feature v0：29 维已冻结，但不能叫 geometry learning

29 维固定顺序分为：

- 1 个 view count；
- 4 个 rotation circular summaries；
- 1 个 `log1p(total active rays)`；
- 9 个逐 view ray fraction；
- 1 个 aperture sample count；
- 3 个 spacing 与 3 个 extent；
- 1 个 global valid aperture-sample fraction；
- `k0..k4` 的 5 个 relative measurement residual；
- `k4` 的 1 个 relative normal residual。

第 23 维原名 `global_valid_ray_fraction` 已在计分前更正为 `global_valid_aperture_sample_fraction`：底层 `valid[ray,sample]` 是 aperture sample 级布尔量，不是整条 ray 的布尔量。同一固定 geometry 下，前 23 维对 16 个 phantom 全部相同，只有最后 6 个 residual 特征会变化。因此 Phase-0 只能检查 schema、计算成本和 leakage，不能声称学会跨 geometry 泛化。

禁止 feature 包含 family、seed、ID/hash、truth、field/gradient error、harm、未来 checkpoint、真实观测值、exact operator mass/spectrum、oracle action、wall time 或数据角色。

producer 已用 canonical contract hash 冻结；独立 witness 不导入 producer，用另一套 `math` 实现重算 29 个值。它会拒绝“改 feature 后重算 feature hash”或“改 input 后保留旧 feature”的内部不一致；但若同时替换 input、按公开公式重算 29 个值并重算全部 hash，witness 会通过。这不是漏洞被掩盖，而是 witness 只证明内部一致、不能证明来源真实性。新增 data-foundation runner 已在微型 fixture 内从同一条 CGLS 的 `k0..k4` residual 与 `A^T r4` 内生构造 sufficient-stat record；真实 PSU private unit 尚未执行，candidate authorization 仍关闭。

## 6. 指标与调用成本

每个 candidate 相对 `k4` 定义 additive harm，共 `6 actions × 24 metrics`：

- 4 个全局 projection/macro/worst/p95 指标；
- 九个 view slot 各自的 relative-L2 与 p95，共 18 个；
- analytic truth 下的 field relative-L2 与 gradient relative-L2。

Phase-0 不用 front metric，因为 plume、vortex 与 shock 没有统一的单前缘物理定义。

一次最大深度 CGLS 轨迹保存 `k0,k1,k2,k3,k4,k6,k8,k12`。每个 unit 的逻辑成本是：

| 部分 | A | Aᵀ |
|---|---:|---:|
| synthetic truth | 1 | 0 |
| CGLS 到 k12 | 12 | 13 |
| 七个非零 checkpoint score | 7 | 0 |
| 每 unit 合计 | 20 | 13 |

16 units 加一次共享 dot test：`321 A + 209 Aᵀ`。由于 feature 必须看到 `k4` residual，最终选择 `k1/k2/k3` 也已经付过前四步成本；它们没有计算速度优势，只可能提供不同的隐式正则化。

## 7. 启动前三个硬阻断项

1. 将已经通过 fixture 测试的 24-metric scorer 与 private foundation finalizer 集成，只从同一进程的 observation/checkpoint 生成绑定 run/unit/source 的 metric/harm bundle；
2. commit 后执行不计分 private preflight：内部 attestation、source snapshot、scorer hash、dot product、精确调用序列、空 score ledger 与无重复 unit identity；
3. preflight 通过后仍需一次显式人工授权，不允许脚本因测试通过而自动打开 16-unit 分数。

runner 基础层已做到：通用 duck-typed 路径只能返回 `FIXTURE_ONLY`；生产路径要求精确 geometry-only adapter/operator，manifest digest 只从权限收紧且拒绝重复 JSON key 的 Git 忽略私有 attestation 读取；`unit_manifest.json` 绑定 run/unit、冻结 phantom 行、配置与 source manifest，根目录外部原子 claim 拒绝同一 unit 换名或已有目录互相重复；claim 先写 `PREPARING_UNPUBLISHED`，只有可逐字节证明未发布的崩溃 residue 才能在根锁内隔离到 quarantine 后重试，已有 finalization 但没有 anchor 时保持人工 NO-GO；finalizer 不接收 fallback bytes，只把密封 `k4` 写成 fallback，重读所有成员并与密封 bytes 全量相等，把 checkpoint 范数和 CGLS 系数递推对回 solver history，发布 `finalization.json` 后再写 bundle 外的 local final anchor。output root 只沿 `O_NOFOLLOW` 父链安全创建最后一级。完整攻击测试与边界见 `docs/e73_phase0_data_foundation_2026-07-19.md`。

这里的 capability 不是密码学权限。当前合同明确要求可信本地 Python 进程，`same_process_malicious_code_resistance=false`、`cryptographic_signature=false`；它关闭误调用、普通漂移与已覆盖的落盘篡改路径，不宣称能抵抗在同一解释器内任意执行恶意代码。

只有 16/16 unit 完整、dot error `<=1e-10`、调用数精确、全部数值有限、强制回退逐字节等于 `k4`、峰值 RSS `<=24 GiB`，才可写：

`PHASE0_SCHEMA_GO_ANALYTIC_PROXY_ONLY_NO_PERFORMANCE_CLAIM`

这仍不等于 predictor 训练、calibration、真实流场、CFD、泛化、算法优越性或论文结果。

## 8. 下一步顺序

1. geometry-only adapter、29 维 feature/witness、data-foundation runner、可信源绑定和逐字节 `k4` private finalizer 已在同一私有 commit 中闭合；
2. fixture-only 24-metric scorer 已精确验证 7-forward、9-view、24-metric 和 `candidate-k4` harm；下一步是 private integration，不是 predictor；
3. integration 提交后先做不计分 private preflight，再请求人工授权是否打开 16-unit schema pilot；
4. schema pilot 通过后再预注册至少 64 个 family-aware analytic units；
5. 最后才讨论真实独立 flow/session calibration 与 candidate authorization。

现在最有价值的不是立即训练 FNO，而是把这条数据链做成师兄可以逐项审核、任何失败都会自动回到 `k4` 的可复现实验合同。
