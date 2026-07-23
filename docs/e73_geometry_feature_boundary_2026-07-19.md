# E73 geometry-only 与 29 维特征边界

更新：2026-07-19

当前状态：

- `REAL_CACHE_TRUSTED_MANIFEST_FULL_OPERATOR_OPEN_AUDIT_PASS`
- `FEATURE_CONTRACT_HASH_FROZEN_SOURCE_PROVENANCE_PENDING`
- `PHASE0_PREFLIGHT_NO_GO_CONTRACT_INCOMPLETE`
- `CANDIDATE_AUTHORIZATION_DISABLED`
- `NO_PERFORMANCE_CLAIM`

这一步解决的是“数据有没有偷看、feature 到底怎么算”，不是“候选算法是否更好”。尚未运行 16-unit Phase-0，也没有打开任何 candidate harm、field error 或论文性能分数。

## 1. 为什么必须单独写 geometry-only adapter

原来的 `PSUCompactCachedRayStore` 会在构造时同时 mmap 五类几何数组与 `observations_uv.npy`。Phase-0 的观测必须由解析代理经同一个离散算子生成：

\[
y=A(x_{proxy}).
\]

如果程序在生成 synthetic `y` 前顺手打开真实 observation，即使没有显式使用，也无法证明后续特征和选择没有受真实测量影响。新适配器 `site_tools/e73_geometry_only_cache.py` 因此只开放：

1. `base_indices.npy`；
2. `fractions_xyz.npy`；
3. `valid.npy`；
4. `projection_uv_xyz.npy`；
5. `ray_scale.npy`。

`load_observations()` 永远抛出 `PermissionError`，chunk 内的 `observation_uv` 固定为 `None`，对象也没有 `observations_uv` 属性。manifest 若把任何 geometry role 的 filename 改成 `observations_uv.npy`，即使记录和哈希自洽，也在文件打开前拒绝。

独立安全复审进一步指出：只检查 basename 仍可能被 symlink/hard-link 欺骗，任意嵌套 manifest 字段也可能夹带值。修订版因此强制调用者提供可信 manifest digest；cache root 以目录 FD 打开，每个 geometry `.npy` 用 `O_NOFOLLOW` 只打开一次，并在同一个 FD 上解析 NPY header、校验 SHA-256、建立 copy-on-write mmap，避免“检查一条路径、NumPy 又重开另一份字节”的窗口。它也会在读 payload 前拒绝与 observation 相同的 device/inode；manifest、source selection、view row、chunk、array record 与 claim boundary 均采用严格字段白名单。审计从 gitignored 私有 attestation 取得预先保存的 digest；公开摘要只报告“已绑定”，不泄露 digest 或本机路径。

## 2. 真实 cache 的宿主级打开审计

独立审计脚本使用 Python host audit hook 记录 cache 根下发生的文件打开，不只相信 adapter 自报 ledger。它在真实 16³ PSU B0 cache 上得到：

| 项目 | 结果 |
|---|---:|
| ray count | 10,628,822 |
| view slots | 9 |
| adapter ledger entries | 11 |
| `observations_uv.npy` open count | **0** |
| trusted source-manifest digest | 已私下绑定，公开值隐藏 |
| geometry 文件打开 | 每个 1 次；同一 FD 校验并 mmap |
| 实际覆盖路径 | constructor、iter_chunks、forward、adjoint |
| 完整遍历 | 329 chunks / 10,628,822 rays |
| 私有路径/manifest hash 是否进入摘要 | 否 |

五类 geometry `.npy` 均实际被 mmap 并做 SHA-256 校验；同一审计进程随后遍历全部 chunk，并各执行一次真实 matrix-free forward 与 adjoint。observation 文件即使被删除，小型攻击测试中的 geometry forward/adjoint 仍与完整 cache 逐值相同。公开安全摘要见 `docs/e73_geometry_only_real_cache_open_audit_2026-07-19.json`。

这项 PASS 只证明“本次被哈希绑定的 manifest 与 adapter，在 constructor/chunk/forward/adjoint 这四条已执行路径上没有打开 observation”。它不证明其他进程、未来 runner 或人工脚本绝不会打开，也不是对任意恶意本机启动器的密码学签名保证；正式 runner 仍必须绑定这些源码和私有 attestation，并在自己的事务中重复审计。

## 3. 29 维 feature 的精确定义

feature producer 是 `site_tools/e73_phase0_features.py`。输入不是任意字典，而是只有十二个字段的 sufficient-statistic record；额外加入 `family`、`seed`、unit ID、truth、harm、未来 checkpoint 或 data role 会直接拒绝。

29 维按固定顺序组成：

| 区间 | 内容 | 数量 |
|---|---|---:|
| 0 | view count | 1 |
| 1–4 | rotation 的 mean-sin、mean-cos、resultant、circular variance | 4 |
| 5 | `log1p(total active rays)` | 1 |
| 6–14 | 九个 view-slot ray fractions | 9 |
| 15 | finite-aperture sample count | 1 |
| 16–18 | spacing x/y/z，单位 m | 3 |
| 19–21 | extent x/y/z，单位 m | 3 |
| 22 | valid ray-aperture-sample pair fraction | 1 |
| 23–27 | `k0..k4` relative measurement residual | 5 |
| 28 | `k4` relative normal residual | 1 |

这里修正了一个会误导物理解释的旧名字：`valid` 数组逐项描述“某条 ray 的某个 aperture sample 是否落在网格内”，所以第 22 维叫 `global_valid_aperture_sample_fraction`，不再叫 `global_valid_ray_fraction`。

网格输入顺序冻结为 `shape_zyx`，但 spacing/extent 输出顺序冻结为 `xyz`；非立方 `4×5×6` 回归测试专门防止 x/z 被交换。rotation 以 view slot 等权，不按 ray 数加权。measurement/normal residual 的归一化分母使用预先固定的 `1e-12` floor；ray fraction 与 spacing 等结构量的分母由合同保证为正，不使用这个 floor。

## 4. 为什么另写一个 independent witness

`site_tools/verify_e73_phase0_feature_artifact.py` 不导入 producer，而是用 Python `math` 独立重写合同、轴顺序、单位变换和 29 个公式。它同时验证：

- feature contract SHA-256；
- sufficient-stat input SHA-256；
- canonical float64-le feature SHA-256；
- 29 个数与独立公式的最大绝对差不超过 `5e-15`；
- claim firewall 全部保持 false。

攻击测试把一个 ray fraction 改掉并重新计算 feature hash，witness 仍因独立重算不一致而拒绝；把输入 ray count 改掉并只重算 input hash，同样无法让旧 feature vector 通过。

这个 witness 的范围只是**内部一致性**，不是来源真实性：若攻击者同时替换 input record、按公开公式重算全部 29 个 feature，再重算两个 hash，witness 会通过，这是预期行为。当前 producer 接收的是小型记录，尚未证明 sufficient statistics 来自正式 runner。未来 Phase-0 runner 必须从自己持有的 geometry、synthetic observation、CGLS `k0..k4` residual 和 `A^T r_4` 内部构造记录，不允许调用者提交，并把该来源链绑定到可信 runner/config hash。

## 5. 当前只剩下哪些启动门

1. 实现 Phase-0 data runner，并绑定 geometry adapter、feature producer、independent witness 与机器 config 的源码/hash；
2. runner 的拒答结果必须与同一轨迹保存的 `cgls_k4` 三维数组逐字节相同；
3. 本轮 adapter、features、witness 与修订 config 必须先形成 clean protocol commit，之后才可打开任何 16-unit metric/harm。

因此当前最窄结论是：**在一次可信 manifest 绑定、完整 chunk/forward/adjoint 审计中 observation open count 为 0，29 维公式可以被独立复算；feature 来源真实性、算法性能、几何学习、泛化与候选接管仍完全没有证据。**
