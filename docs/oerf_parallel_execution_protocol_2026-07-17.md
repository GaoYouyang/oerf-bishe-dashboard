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
env PYTHONPATH=. \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -n 4 --dist=loadfile \
  demo_t16_operator site_tools demo_m3b paper_library/tools \
  -k 'not cpu_float32_and_mps_float32_match_on_fixed_short_run'

PYTHONPATH=. .venv/bin/python -m pytest -q \
  demo_t16_operator/test_psu_b0_primal_dual.py::\
test_cpu_float32_and_mps_float32_match_on_fixed_short_run
```

2026-07-17 的第一次固定 4-worker 实测为 `918 passed / 11.54 s`；随着本轮新增
模块并入，最新统一 medium 门为 `977 passed / 12.86 s`，随后串行 MPS
parity 为 `1 passed / 1.02 s`。fast 已包含 170 项页面/因子合同测试与 46,214 个
旧 artifact 链接审计；完整 medium matrix 总时长为 `16.81 s`。上一轮串行
完整套件约为 `28.91 s`，在测试数增加后反馈时延仍缩短约 41.9%。

### Full：提交与发布前

- medium 全部通过；
- 从固定 commit 构建唯一 `build/pages-audit-<sha>`；
- 链接、隐私、checkpoint/PDF 排除和 HTTP smoke；
- 桌面与移动端真实浏览器渲染；
- 构建期间不修改工作树。

Full 仍不自动运行 fresh/opened/postopen/held-out 科学实验。科学实验有自己的
clean-HEAD、输入 hash、测试节点 hash、结果目录和开封门。

## 5. 当前流水线状态

已完成但尚未等同于正式 Gate A attestation：

- tiny CPU/float64 signed-factor majorizer 与 6-step recurrence；
- production `P/P^T`；
- centered physical gradient 的 `|G_c|/|G_c|^T`；
- 组合后再取绝对值的 exact `|H R Q|/|H R Q|^T`；
- strict coordinate `E/E^T` 与 exact-zero constant ledger；
- production full-chain matrix-free ones-pass、camera/site shared step 与 exact-zero mask；
- production 1/6-step Huber recurrence 与 site-major dense oracle 逐步对齐；
- 4-worker 测试调度和依赖固定。

仍未完成：

- 冻结 config/input/test-node/code fingerprint 及 stale 负例；
- setup/solver/scorer 的正式机器可读账本与独立 validator；
- CPU/MPS 固定 fixture 的完整 Gate A attestation；
- Gate B 性能比较。

因此当前合法结论仍是：实现链取得了可验证增量，网络不是主瓶颈，Gate A 与
算法性能均尚未通过。不得把测试提速或因子正确性写成重建优越性。

## 6. 后续顺序

1. 冻结 Gate A 配置、测试 node 列表、代码与输入 hash。
2. 在 clean commit 上生成 CPU/MPS attestation 及独立 validator；失败只修合同。
3. Gate A 全通过后，才运行 scalar/block/factor 与 graph-PCGLS 的同调用 Gate B。
4. 先报告 mean、p10、harm、worst、gradient/front 与 cold-start 成本，再决定是否
   值得转向 learned hybrid 或申请远端算力。
