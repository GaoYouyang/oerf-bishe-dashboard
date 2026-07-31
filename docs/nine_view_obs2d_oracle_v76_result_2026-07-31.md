# v76 同预算二维可行性结果：17/75 个单元有严格不可行证书

> 日期：2026-07-31  
> 科学判决：`FAIL_OBS2D_ORACLE_SPAN_INSUFFICIENT_V76`  
> 独立验证：`PASS_INDEPENDENT_VALIDATION_OBS2D_ORACLE_V76_1`  
> 突破状态：`algorithm_breakthrough=false`

## 先说结论

v75 证明固定 PoolFire `q8-K1` 原样跨到 BLASTNet vitiated H2-air
Case 3 时，完整外部合同只有 `5/75`。v76 继续问一个更硬的问题：

> 不增加一次精确 `A/A^T`，如果让一个知道真值的 oracle 在当前两条方向里
> 选择最好的两个系数，能不能把 75 个单元全部救回来？

答案是不能。

```text
严格可行见证        58 / 75
严格不可行证书      17 / 75
数值不确定           0 / 75
```

每档冻结几何都存在不可行单元：

| 几何 | 可行 | 严格不可行 |
|---|---:|---:|
| F12+ | 20/25 | 5/25 |
| F15+ | 18/25 | 7/25 |
| F30+ | 20/25 | 5/25 |

因此，任何只负责预测这两个系数的 MLP、FNO、DeepONet 或其他网络，都不可能
通过当前 75 单元合同。继续把网络做大只会浪费训练时间，不能改变表示空间。

## 二维表示是什么

部署可见的 loaded-q8 detector proposal 记为 `z`。先做精确提升：

```text
h = A^T z
```

再做一次精确残差和伴随：

```text
r_h = y - A h
n   = A^T r_h
q   = A n
```

v76 检查整个二维场空间：

```text
x(c) = c1 h + c2 n
A x(c) = c1 A h + c2 A n
```

生成 `h,n,Ah,An` 的账仍是 `2A+2A^T`；搜索系数只在已经生成的二维数组上
运算，额外精确调用为 0。oracle 可以看真值，所以它不是部署算法，只是当前
表示能力的上界。

## 八个约束

每个单元必须同时不劣于两条参考：

1. 相对 Zero-K4 的 field / full-gradient / interior-gradient /
   observation 不得超过 `1.01`；
2. 相对同调用 Zero-K2 的上述四项不得超过 `1.00`。

把每项写成关于 `c=(c1,c2)` 的凸二次不等式：

```text
g_j(c) = ||F_j c - t_j||_2^2 / scale_j^2
         - (tau_j / scale_j)^2 <= 0
```

v76 求解 `min_c max_j g_j(c)`。判决留有 `1e-10` 双侧弃权带：

- `max_j g_j(c) <= -1e-10` 才是可行见证；
- 只有精确有理数 simplex 权重使加权二次函数的全局最小值
  `> +1e-10`，才是不可行证书；
- 两边都不到就必须写 `INCONCLUSIVE`。

本轮没有单元落入不确定区。

## 为什么这次不是“优化器没找到”

对 17 个失败单元，保存的不是“搜索若干次仍没找到”，而是：

```text
lambda_j >= 0
sum lambda_j = 1
Q_lambda is positive definite
inf_c sum_j lambda_j g_j(c) > 0
```

若存在一个 `c` 同时满足所有 `g_j(c)<=0`，它们的任意非负加权和也应
`<=0`；但证书证明该加权和对所有 `c` 都严格大于 0，产生矛盾。因此二维
交集确实为空。

17 个不可行单元的精确 dual lower bound 最小为
`1.7168886e-4`，最大为 `9.3063048e-3`，都远离 `1e-10` 判决边界。

## 失败机制

truth-aware oracle 确实把完整通过数从原候选的 `5/75` 提高到 `58/75`，
说明两个系数不是毫无价值；但它仍无法闭合梯度与观测之间的冲突：

| v76 oracle 门 | 通过数 |
|---|---:|
| field / Zero-K4 | 75/75 |
| full-gradient / Zero-K4 | 75/75 |
| interior-gradient / Zero-K4 | 75/75 |
| observation / Zero-K4 | 58/75 |
| field / Zero-K2 | 75/75 |
| full-gradient / Zero-K2 | 67/75 |
| interior-gradient / Zero-K2 | 63/75 |
| observation / Zero-K2 | 75/75 |

在全部 17 个不可行单元的 minimax 点上，K4 observation 门都仍为正；
其中 12 个同时违反 K2 interior-gradient，8 个同时违反 K2
full-gradient。最大约束有 12 个来自 K4 observation、4 个来自 K2
interior-gradient、1 个来自 K2 full-gradient。

这说明不是简单把当前 correction 再放大或缩小就能解决。为了改善局部梯度，
二维方向必须牺牲已经守住的 K4 观测残差；为了守住观测，又无法达到同调用
K2 的梯度水平。

不可行帧集中在演化中段，frame 10 和 16 在三档几何上都不可行。这支持
“流场形态变化使固定二维 Krylov 表示失配”的解释，但仍只是当前公开代理上的
机理判断，不是对真实实验的证明。

## 独立验证与一次验证器修复

正式 runner 在结果前冻结的干净 detached worktree 上生成 75 行原子
输出。第一版独立 validator 立即停止，因为它错误要求：

> 独立重建浮点二次式后，有理数证书的分子和分母必须逐字相同。

不同运算顺序会产生约 `1e-15` 的末位浮点差，因此转成精确
`Fraction` 后文本分数必然不同。这个要求过严，不代表证书失效。v76.1
只修验证口径，不改正式输出、二维表示、门槛、单元或判决：

1. 正式分数自身必须 canonical、行列式为正；
2. 独立路径用同一整数权重、独立重建的二次式再次做精确有理数证书；
3. 对不可行单元，两条路径的 exact lower 都必须严格大于 `1e-10`；
4. 浮点重建差和 determinant 相对差仍受预写容差约束；
5. 修复后的 validator 从另一份干净 detached commit 运行并绑定自己的
   source closure。

最终：

```text
formal/independent 最大绝对差             4.7962e-14
exact dual lower 最大重建差               6.2970e-16
exact determinant 最大相对重建差          2.7871e-15
formal payload unchanged                  true
raw payload unchanged                     true
```

这次修复没有把失败改成通过；它把一个错误的“文本完全相同”要求，换成了
真正需要的“双路径都给出严格正证书”。

## 关闭什么，不关闭什么

现在正式关闭：

- 只在 `span{h,n}` 里预测两个系数；
- 用更大的网络挽救同一个二维表示；
- 在 v76 之后为同一表示打开 Case 4/6 补考；
- 为失败表示运行 wall/RSS 资源门。

没有被否定：

- 产生空间变化 correction 的 field-space 网络；
- 跳出 `span{h,n}` 的多尺度或局部梯度方向；
- 在同一 `2A+2A^T` 预算下先构造新的 `x0`，再运行未修改 CGLS K1；
- curved-ray 与真实 BOST。

下一条有效路线必须先改变表示，而不是先增大模型。最直接的候选是：

```text
x0 = h + u_theta(h, geometry-visible features)
r0 = y - A x0
x1 = one unchanged CGLS step from x0
```

其中 `u_theta` 是空间变化的三维修正，能够直接修复局部梯度，不再被限制为
两个全局标量。精确账仍可保持 `2A+2A^T`。训练前仍要先做新的 truth-aware
representation oracle；Case 3 只能作为已开封 development，另一个未打开
工况必须保留为一次性外部门。

## 证据边界

```text
same_budget_obs2d_oracle_pass=false
span_h_n_closed=true
larger_network_on_same_span_authorized=false
field_space_representation_not_yet_tested=true
resource_stage_authorized=false
case4_or_case6_opened=false
real_bost_result=false
algorithm_breakthrough=false
paper_success=false
```

脱敏摘要与图表：

- `docs/nine_view_obs2d_oracle_v76_public_summary.json`
- `assets/nine_view_obs2d_oracle_v76.png`
