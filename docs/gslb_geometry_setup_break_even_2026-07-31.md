# GSLB 几何 setup 摊销与调用数 break-even

> 日期：2026-07-31  
> 角色：结果无关的精确调用账分析。  
> 当前状态：未测 wall/RSS，`algorithm_breakthrough=false`。

## 1. 为什么必须单独计算 setup

GSLB 候选的在线账是每个 cell `2A+2A^T`，Zero-CGLS K4 是 `4A+4A^T`。只比较
这两个在线数字会隐藏几何 basis 的一次性构造成本。

对 rank `k`：

```text
geometry setup = (128+k) A + k A^T
accepted online cell = 2A + 2A^T
Zero-K4 cell = 4A + 4A^T
```

这里 128 次 `A` 构造完整样条父空间的 observation，另外 `kA+kA^T` 构造所选模式的
`A U_k`、`A^T A U_k` 与后续 reduced evaluation 所需量。

## 2. 一般 break-even 公式

设一次 `A` 的时间成本为 `c_A`，一次 `A^T` 为 `c_AT`，同一冻结几何下有 `N` 个
全部接受的 cell。GSLB 相对 Zero-K4 的最好情况 break-even 为：

```text
N >= ((128+k)c_A + k c_AT) / (2c_A + 2c_AT).
```

若只有比例 `p` 的 cell 在不消耗精确调用的 pre-exact gate 后被接受，其余直接走
Zero-K4，则：

```text
N >= ((128+k)c_A + k c_AT) / (p(2c_A + 2c_AT)).
```

所以接受率下降会按 `1/p` 放大摊销所需帧数。若先运行候选的精确调用再 fallback，上式
不成立，成本会更差，必须重新完整计账。

## 3. `c_A=c_AT` 时的精确表

| rank | setup `A` | setup `A^T` | 等权 setup 总调用 | 最少 break-even cells |
|---:|---:|---:|---:|---:|
| 8 | 136 | 8 | 144 | 36 |
| 32 | 160 | 32 | 192 | 48 |
| 127 | 255 | 127 | 382 | 96 |

### 不同复用长度的总调用

表中 candidate 已包含一次 geometry setup，并假定所有 cell 接受、没有 fallback、没有把
便宜 factor 或网络计算折算成精确调用。

| rank | 同几何 cells | Zero-K4 总调用 | GSLB 总调用 | GSLB/Zero-K4 | 等权调用变化 |
|---:|---:|---:|---:|---:|---:|
| 8 | 25 | 200 | 244 | 1.2200 | +22.0% |
| 32 | 25 | 200 | 292 | 1.4600 | +46.0% |
| 127 | 25 | 200 | 482 | 2.4100 | +141.0% |
| 8 | 101 | 808 | 548 | 0.6782 | -32.2% |
| 32 | 101 | 808 | 596 | 0.7376 | -26.2% |
| 127 | 101 | 808 | 786 | 0.9728 | -2.7% |
| 8 | 505 | 4040 | 2164 | 0.5356 | -46.4% |
| 32 | 505 | 4040 | 2212 | 0.5475 | -45.2% |
| 127 | 505 | 4040 | 2402 | 0.5946 | -40.5% |

## 4. 对 v78 的直接判决边界

v78 每套几何只含 25 个已经开封的 development cell。因此即使 GSLB32 达到 75/75，
本次 formal 在等权精确调用上也不是资源胜利：每套几何 `292` 次完整调用，高于 Zero-K4
的 `200` 次。

v78 的唯一作用是判断 rank-32 表示是否存在全 cell headroom。未来只有满足以下条件，
才能把在线 `2A+2A^T` 转化为完整调用优势：

1. 同一几何至少复用到 break-even 以上的 cell；
2. setup 只构造一次且不因 trajectory、噪声或标定变化重复；
3. pre-exact gate 在拒绝时不会先花候选的精确调用；
4. 接受率足够高；
5. 网络、factor、缓存和数据搬运没有抵消 wall/RSS；
6. 最终 field/gradient/observation matched-accuracy 仍成立。

## 5. 对 GSLB127 的策略含义

rank-127 在等权成本下需要 96 个同几何 cell 才 break even。一条 101-frame trajectory
即使全部接受也只剩约 2.7% 的理论完整调用优势，极容易被 predictor、factor、setup 和
缓存开销吞没。

所以若 v78 失败，GSLB127 仍可作为最终 parent-space 表示上界诊断，但即使它找到
truth-aware witness，也不能直接成为部署候选。它必须证明 setup 能跨更多同几何数据稳定
复用，否则论文价值只剩“表示上界负/正诊断”，没有可信的端到端加速故事。

## 6. 可复算公式

```python
import math

def break_even_cells(rank, adjoint_to_forward_cost=1.0, acceptance=1.0):
    q = float(adjoint_to_forward_cost)
    p = float(acceptance)
    setup = (128 + rank) + rank * q
    saving_per_accepted_cell = 2 + 2 * q
    return math.ceil(setup / (p * saving_per_accepted_cell))
```

这只是 exact-call break-even，不是 wall-time 或 memory 证明。

