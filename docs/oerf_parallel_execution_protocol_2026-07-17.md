# OERF 算子学习与三维重建并行执行协议

日期：2026-07-17
状态：`ACTIVE_WORKFLOW / NO_SCIENTIFIC_WIN_CLAIM`

这份协议解决的是反馈时延，不改变预注册、证据等级或科学门。能并行的是互不
依赖的代码、文献、数据和页面支线；正式性能开封、MPS 计时和共享结果目录仍
必须串行。

## 1. 本机与网络实测

| 项目 | 2026-07-17 实测 | 对当前工作的含义 |
|---|---:|---|
| 芯片 / 内存 | Apple M5，10 核，32 GB | 适合 4 个 CPU 测试 worker；只保留 1 个 MPS 数值任务 |
| 可用磁盘 | 431 GiB | 足够保存公开小基准和私有实验结果；不应复制 51 GB PSU 全量包 |
| 下行 / 上行 | 75.8 / 37.5 Mbps | 论文元数据、代码和中小数据下载不是当前主瓶颈 |
| 代理基础 RTT | 284 ms | 大量小网页逐个请求会慢，应批量查询并缓存 DOI/元数据 |
| GitHub raw TTFB | 0.96 s | 正常可用 |
| Crossref TTFB | 1.10 s | 正常可用 |
| arXiv TTFB | 0.79 s | 正常可用 |
| GitHub Pages TTFB | 1.29 s | 正常可用 |

测速显示所有请求经本机代理转发，因此 `networkQuality` 报告 loopback 接口；这
不代表只测试了本机。当前不建议切换 VPN。只有连续三次出现下行低于 10 Mbps、
常用一手来源连接超时率高于 10%，或下载公开数据时长时间低于 1 MB/s，才值得
调整线路。学校 WebVPN 的登录/授权失败属于访问控制问题，不是吞吐问题。

## 2. 依赖图，而不是无限加并发

```text
production operator map ----> exact |G|, P/P^T ----\
math red team --------------> tiny dense oracle ----+--> Gate A integration
exact composed |H R Q| -----------------------------/
active coordinates + zero ledger ------------------/
absolute TV factor --------------------------------/

primary literature ----------> innovation boundary ------> focused webpage
public datasets --------------> migration ladder --------> benchmark plan
test-latency audit -----------> fast/medium/full matrix --> every code loop

Gate A integration --PASS--> frozen Gate B --> one serial MPS run --> evidence
Gate A integration --FAIL--> fix implementation; do not interpret performance
```

主线程只守住 Gate A 的最近阻塞项。子代理只领取有清楚输入、输出和独占文件的
支线；不把下一步立即依赖的工作交出去后原地等待，也不让两个代理同时修改同一
文件。

## 3. 并发配额

| 资源 | 并发上限 | 原因 |
|---|---:|---|
| 只读文献/数据核验代理 | 3 | 主要受 284 ms 网络往返限制，批量请求有收益 |
| 相互独立的代码 worker | 3 | 必须拥有不重叠的文件集合 |
| pytest CPU workers | 4 | M5 的 4 个性能核与 6 个能效核下避免 BLAS 过度抢核 |
| MPS 训练或正式计时 | 1 | 多进程会争显存、温控和统一内存，破坏 wall-time 证据 |
| Pages artifact 构建 | 1 / 输出目录 | builder 会先删除目标目录；不同快照也不应混写 |
| science runner | 1 | 保持预注册顺序、共享报告、种子和开封边界 |

CPU 测试统一设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、
`VECLIB_MAXIMUM_THREADS=1` 和 `NUMEXPR_NUM_THREADS=1`。不能用 `-n auto`。

## 4. 三层验证反馈

### Fast：每次小编辑

- 新增模块自己的定向测试；
- Pages builder/public-summary 契约；
- `git diff --check`；
- 已存在 artifact 的本地链接审计。

目标时延小于 30 秒。不构建新 artifact，不启动浏览器，不运行任何科学 screen。

### Medium：一次功能切片完成

固定源码根：

```bash
.venv/bin/python site_tools/run_oerf_verification_matrix.py --tier medium
```

脚本固定四个 CPU worker，并自动 deselect 后串行运行 3 个 MPS case，避免新增 MPS
测试后忘记从并行阶段移出。

2026-07-17 的第一次固定 4-worker 实测为 `918 passed / 11.54 s`。正式 Gate A
合入后，最新统一 medium 门为 `1006 passed / 11.29 s`，随后 3 个串行 MPS case
为 `3 passed / 0.87 s`；fast 为 `199 passed / 1.46 s`。完整 medium matrix
总时长为 `15.88 s`。上一轮串行完整套件约为 `28.91 s`，在测试数继续增加后
反馈时延仍缩短约 45%。

### Full：提交与发布前

- medium 全部通过；
- 从固定 commit 构建唯一 `build/pages-audit-<sha>`；
- 链接、隐私、checkpoint/PDF 排除和 HTTP smoke；
- 桌面与移动端真实浏览器渲染；
- 构建期间不修改工作树。

Full 仍不自动运行 fresh/opened/postopen/held-out 科学实验。科学实验有自己的
clean-HEAD、输入 hash、测试节点 hash、结果目录和开封门。

## 5. 当前流水线状态

已完成并进入正式 Gate A mechanics attestation：

- tiny CPU/float64 signed-factor majorizer 与 6-step recurrence；
- production `P/P^T`；
- centered physical gradient 的 `|G_c|/|G_c|^T`；
- 组合后再取绝对值的 exact `|H R Q|/|H R Q|^T`；
- strict coordinate `E/E^T` 与 exact-zero constant ledger；
- production full-chain matrix-free ones-pass、camera/site shared step 与 exact-zero mask；
- production 1/6-step Huber recurrence 与 site-major dense oracle 逐步对齐；
- 4-worker 测试调度和依赖固定；
- config/input/test-node/code/expanded-node/environment fingerprint 与 stale 负例；
- setup/oracle/solver/scorer 的逻辑和物理账本；
- clean commit CPU/MPS attestation、独立 NumPy oracle、333 项 validator checks；
- validation report 与 release checksums 的第二次 `--no-write` 复核。

仍未完成的是 Gate B 性能比较。当前合法结论是 `Gate A mechanics attested`，不是
“算法有效”。本机环境虽记录完整 Torch/NumPy/pytest tree hash，仍不是隔离容器证明。

## 6. 后续顺序

1. 保留 Gate A 证据不变；后续源码修改必须重新 attestation。
2. 冻结 Gate B 数据、scalar/block/factor/graph-PCGLS 基线、调用预算和失败门。
3. Gate B 只跑 data-only；通过后才开放 TV/Huber 与 FM-CG-PDNO smoke。
4. 先报告 mean、p10、harm、worst、gradient/front 与 cold-start 成本，再决定是否
   值得转向 learned hybrid 或申请远端算力。
