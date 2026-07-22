# E73 Phase-0 数据基础层：同轨迹特征、可信源绑定与 k4 落盘回退

日期：2026-07-19
当前状态：`PHASE0_PREFLIGHT_NO_GO_CONTRACT_INCOMPLETE`

这次完成的是 **Phase-0 的数据基础层代码与微型夹具证据**。没有运行真实 16-unit，没有计算任何 field/projection metric、harm、模型分数或 candidate takeover，也没有训练 predictor。它回答的是“将来一旦启动，数据从哪里来、CGLS 状态怎样产生、拒答时到底落盘哪一个数组”，不是“算法是否更好”。

## 1. 现在的数据链

```text
private attestation
  -> exact geometry-only adapter
  -> analytic morphology proxy x
  -> synthetic y = A(x)
  -> one CGLS trajectory to k12
  -> save k0,1,2,3,4,6,8,12
  -> build k0..k4 + A^T r4 sufficient statistics internally
  -> 29-feature producer
  -> independent formula witness
  -> private unit manifest binds run/unit/config/phantom/cache source
  -> root-level atomic unit claim writes PREPARING_UNPUBLISHED
  -> exclusive private finalizer copies saved k4 as fallback
  -> reopen every member byte-for-byte and publish finalization.json
  -> publish a bundle-external local final anchor last
  -> exact pre-finalization crash residue is quarantined before retry
```

正式入口不再接收调用方给出的 manifest digest。它只从 Git 忽略的私有 attestation 读取可信 digest，并要求运行时对象的精确类型分别是 `PSUCompactGeometryOnlyRayStore` 与 `PSUB0StreamingOperator`。九个 view slot 的 ray count、16 个 aperture samples、网格、dtype、边界与配置逐项对齐；只看总 ray 数不够。

通用 duck-typed 核心被明确降级为：

`E73_PHASE0_FIXTURE_ONLY_NOT_PRODUCTION_ATTESTED`

这是一个重要边界。独立审计构造过恶意 fixture：`iter_chunks()` 每次暗中调用 observation loader，同时虚报 open count 为 0。它能骗过 duck-typed ledger，却只能得到 fixture 状态，不能进入私有 finalizer，更不能产生生产级 finalization。

## 2. 一条 CGLS 轨迹，不重复前缀

运行器使用与现有普通 `cgls_solve` 相同的递推，只额外克隆冻结 checkpoint：

`k = 0, 1, 2, 3, 4, 6, 8, 12`

微型非立方夹具上，`k1/k4/k12` 最初逐元素相等测试已经扩展为全部非零 checkpoint 的同递推审计；当前聚焦测试确认保存值与独立调用原 CGLS 的结果完全相等。重新打开私有 bundle 时，验证器还会把每个保存 checkpoint 的 `L2/max-abs` 与同一步 history 重新计算对齐，并复核相对残差公式、measurement residual 非增性，以及 `beta_k=||s_k||²/||s_{k-1}||²`、`alpha_k=||s_{k-1}||²/||Ap_{k-1}||²`。history 因此额外保存 projected-direction norm，而不是只检查 `alpha/beta` 正负。每 unit 的基础层调用序列必须精确为：

```text
truth forward
initial adjoint
12 x (forward, adjoint)
```

也就是基础层 `13 A + 13 A^T`。每条 history row 还必须满足第 `i` 步累计 `i+1` 次 forward 与 `i+1` 次 adjoint。七个 checkpoint 的 projection scoring 尚未执行，所以完整 Phase-0 的 `20 A + 13 A^T` 仍未声称完成。

## 3. 29 个 feature 的来源现在怎样绑定

feature record 不由调用方填写。运行器直接从同一递推中取得：

- `||y-Ax_k||_2`, `k=0..4`；
- `||A^T(y-Ax_4)||_2`；
- geometry-only store 的九个 view counts、rotation、aperture count、bool valid mask；
- 冻结网格 shape、extent 与 spacing。

family、seed、unit ID、truth array、field/gradient error、任一 harm、`k6/k8/k12` metric、真实 observation、oracle action 和 wall time 都不进入 feature artifact。producer 生成 29 维向量后，独立 witness 重新计算公式。

这里仍要区分两层证据：fixture 已证明“运行器能内生构造 sufficient statistics”；真实 PSU private unit 尚未执行，所以不能写成“真实来源已运行认证”。

身份不进入 feature，不等于运行单元可以匿名。私有 `unit_manifest.json` 单独绑定：

- `run_id` 与 `unit_id`；
- 冻结的 block、family、uint32 seed 和整行 canonical hash；
- 当前配置 hash；
- 固定 cache alias 与只从私有 attestation 取得的 source-manifest hash。

finalizer 对输出根加进程间排他锁，并在创建目录前逐个重开已有 unit manifest、检查已有目录之间也没有重复。随后它在 bundle 外的 `.unit-claims/<unit_id>/` 以原子 `mkdir` 占用身份，保存原始 unit manifest 与不可变 `PREPARING_UNPUBLISHED` 状态；bundle 完成后再把 `finalization.json` 的 SHA-256 写入同一 claim 目录的 `final_anchor.json`。同一个冻结 `unit_id` 即使换 `run_id` 也会拒绝；把 bundle 合法重标成另一个冻结 unit，或把 checkpoint 保范数取负并重算 bundle 内 checksum，也会与外部锚冲突。

独立复审进一步指出：如果进程刚占用 claim 就断电，旧版会永久阻塞整个输出根。现在重试只能在根锁内隔离一个**可逐字节证明尚未发布**的 residue：claim 必须只有 manifest + preparing state，run 目录只能包含与本次密封 payload 完全相同的成员子集，且两边都不得出现 `finalization.json` / `final_anchor.json`。恢复记录先在 `<entry>.preparing` 目录中完整写入并同步，再在同一 quarantine 父目录内原子改名；在 entry 可见之前不移动 run/claim。完成标记也先写 `isolation_complete.json.preparing`，重读核验后才原子改名。空 stage、半写 record 和半写 completion 都能在确认源尚未移动或 prepared 记录仍完整后安全重建。源 run/claim 的成员轴会在 record 发布前显式同步；跨父目录 rename 后先同步目标 entry，再同步源父目录，优先避免 neither/data-loss 窗口。身份冲突、状态缺失、成员篡改，或“已有 finalization 但没有 anchor”都不自动修复，必须人工审计。

这里仍有一个明确边界：POSIX 不能把两个不同父目录的持久化合成一个真正原子事务。程序用 source/destination XOR 识别 both/neither 并 fail closed，但不能承诺在任意文件系统断电模型下自动修复两种歧义；要消除它，需要更大的 copy-verify + sibling tombstone 结构。本轮没有把这一工程加固写成密码学、断电完备证明或算法结果。

## 4. k4 逐字节回退不再靠内存自证

第一版只在内存里比较两份 bytes。独立审计指出，任何伪 payload 与自身比较都会通过，而且 frozen dataclass 内的普通 dict 仍可被替换。修订后：

1. checkpoint payload map 使用不可变 mapping；
2. caller 没有提交 fallback bytes 的参数；
3. finalizer 只取密封的 `checkpoint_npy_payloads[4]`，分别写成 `cgls_k4.npy` 与 `forced_fallback.npy`；
4. 两个文件均以 `O_EXCL` / `O_NOFOLLOW` 写入，不允许覆盖；
5. 写盘后重新打开，比较完整长度、SHA-256 与全部 bytes；
6. 所有成员复核完成后，最后独占写入 `finalization.json`；
7. 写盘后每个成员都必须与写入前的密封 bytes 完全相同；不是只对重读内容重新算一个新 checksum；
8. bundle 完成后才发布外部 local final anchor。只改成员并同步重算 bundle 内 `finalization.json`，仍会与外部锚不一致；
9. 把 `k1` 改成零场会先因 checkpoint/history 不一致被拒绝；把它整体取负虽然保留 L2/max，也会被外部锚拒绝。

这证明的是本地私有 finalizer 的文件级、跨成员语义和 bundle 外本地锚一致性。威胁模型明确要求**本地 Python 进程可信**：模块内 capability 只防正常调用路径误用，不抵抗在同一进程中任意执行恶意 Python。它也不是签名系统；拥有本机写权限的人若同时重造 bundle 和 `.unit-claims`，仍可能绕过本地锚。真实论文证据仍依赖受控 Git commit、权限为 `0700/0600` 的私有 attestation、只读归档与独立复算。

## 5. 本轮故障注入

新增测试覆盖：

- synthetic observation 数值为空；
- truth 含 NaN；
- source observation ledger 在启动前已非零；
- view 顺序漂移、valid mask 从 bool 偷换为 uint8；
- operator 不是 fresh ledger；
- 第一步 CGLS denominator breakdown；
- adjoint 注入 NaN；
- call record 顺序被伪造；
- fixture store 暗读 observation 26 次但仍不得生产状态；
- attestation 最终 symlink、hard-link、父目录 symlink、宽松目录/文件权限；
- attestation 重复 JSON key；
- fallback 单字节变化；
- finalized member 被篡改；
- 重复 run ID 试图覆盖既有私有目录；
- 改 checkpoint 后重算 checksum；
- 保范数取负 checkpoint 后重算 checksum；
- 改 `alpha/beta` 但保持正值并重算 checksum；
- 改 feature/history/unit manifest 后重算 checksum；
- 同一 `unit_id` 换新 `run_id` 重复发布。
- 已有目录之间预置重复 unit；
- 合法重标到另一个冻结 unit；
- 成员 mode 改成 `0644`、祖先目录 symlink、写后 bytes 与密封 payload 不同。
- claim 创建后、run 目录创建后、首个成员写入后三个崩溃点均可隔离并安全重试；
- orphan recovery record 的空 stage、完整 stage、entry rename 前后，run/claim rename、目标父目录同步、两个父目录均同步，以及 completion 临时文件/最终改名前后均可在同一事务目录恢复；
- `recovery_record.json` 与 `isolation_complete.json` 的半写临时成员会在不放宽身份/成员绑定的前提下丢弃并重建；
- `recovery_record.json` 逐成员绑定 SHA-256，`isolation_complete.json` 是唯一完成标记；任何无完成标记的跨 unit 任务都 fail closed；
- completion 篡改与旧版无完成标记隔离记录必须人工审计，不能自动升级；
- orphan manifest 冲突、claim state 缺失、部分成员篡改均拒绝恢复；
- 已写 `finalization.json` 但缺少外部 anchor 时保持人工 NO-GO；
- child `fstat` 与 verifier 第二次目录打开失败时不泄漏文件描述符；
- output root 的符号链接祖先不会在链接目标侧产生目录；父目录链必须预先存在，只安全创建最后一级。

当前验证：

- E73 code/protocol/config/scorer 聚焦：`156 passed`；
- data-foundation/scorer 相关链：`109 passed`；
- data-foundation runner 故障注入：`53 passed`；
- fixture metric scorer 定向：`19 passed`；
- Pages 发布构建攻击回归：`68 passed`；
- bounded fast matrix：`533 passed`。

这些数字是软件合同检查数，不是 156 个 flow，也不是算法成功率。新的 Pages 构建只允许从 clean `HEAD` blobs 生成发布包；输出经 held `dir_fd`、`O_NOFOLLOW` staging、整树 path/size/SHA-256 封印、同目录 rename 和安装后同 FD 复核才可见。正式输出名会在 seal 后和父目录 `fsync` 后再次绑定；安装后任何验证异常都恢复上一份 artifact。UTF-8/NFKC、HTML、URL、JSON/JS 多层转义、长/嵌套/换行 Base64、路径名中的账号/私有摘要、PDF、凭据、用户目录和私有摘要都进入 fail-closed 扫描。完整链接审计必须等本轮提交后对该提交重新构建，脏工作树会按设计拒绝。

## 6. fixture scorer 已闭合，仍未解除的三道启动门

现在已有独立的 fixture-only 24-metric scorer：它对 `k1/k2/k3/k4/k6/k8/k12` 各执行一次 forward，保留 9 个 view slot，严格计算 `candidate metric - k4 metric`，并只能向 owner-only 目录互斥发布 fixture JSON bundle。公式、信任边界和单元测试见 [E73 Phase-0 fixture metric scorer](e73_phase0_fixture_metric_scorer_2026-07-19.md)。

仍未解除的门是：

1. 将 scorer 与 private foundation finalizer 集成，从同一进程的 in-memory observation/checkpoint 生成绑定 run/unit/source 的私有 metric/harm bundle；不得接收 caller-precomputed metric。
2. commit 后执行不打开分数的私有 preflight：可信 attestation、source snapshot、scorer binding、dot product、精确调用序列、空 score ledger、输出目录无重复 unit；同时保留“可信本地进程、无密码学签名”的边界。
3. preflight 通过后仍要有一次显式人工授权，才能启动 private 16-unit schema pilot；不能由测试数量或自动化脚本自行解锁。

三道门关闭后，才允许启动 16 个 analytic proxy 的 schema pilot。即使 16/16 通过，唯一允许的结论仍是：

`PHASE0_SCHEMA_GO_ANALYTIC_PROXY_ONLY_NO_PERFORMANCE_CLAIM`

## 7. 下一步最窄实现

下一段代码只应实现 private scorer integration，不做 predictor：

- 在 foundation 生成后、进程释放 in-memory observation/checkpoint 前执行 scorer；
- 把 scorer hash、foundation finalization、run/unit/source manifest 和 metric/harm bytes 放进同一条不可换名的证据链；
- 任何半写、重复 unit、不同调用计数或非有限 metric 都只能进入隔离/人工审计；
- public summary 只允许 gate、聚合调用成本、峰值内存和 claim boundary，不带逐 unit 分数或私有哈希。

完成集成和 post-commit private preflight 后，才有资格请求人工授权运行 16-unit schema pilot；仍不需要租 GPU，也不该先训练 FNO/DeepONet。

## 8. 给师兄审核的四个问题

1. Phase-0 用无噪声 `y=A(x_proxy)` 只测接口，是否同意明确写成 inverse-crime schema pilot？
2. 九个 view slot 的 L2/p95 是否应全部保留，还是实验上还有更合理且不会隐藏单相机伤害的 group 定义？
3. `k4` 是否可以作为这条支线唯一经典 fallback，还是应先把 H1/Tikhonov、TV/Huber 或 Pyramid BOST 纳入 fallback 候选？
4. 真正组内数据到位后，什么才是一个独立统计单位：不同 flow realization、不同 run、不同 day/session，还是师兄已有更严格的划分？

## 9. 可复查命令

```bash
.venv/bin/python -m pytest -q \
  site_tools/test_e73_phase0_data_runner.py \
  demo_t16_operator/test_e73_phase0_schema_pilot_contract.py

.venv/bin/python site_tools/run_oerf_verification_matrix.py --tier fast
```

不要在 source commit 之前调用私有生产入口；它会因为绑定源码与 `HEAD` 不一致而 fail closed。
