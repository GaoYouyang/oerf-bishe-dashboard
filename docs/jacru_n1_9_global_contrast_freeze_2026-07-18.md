# JACRU N1.9 三相机 contrast/global-K 旧开发集设计冻结

日期：2026-07-18

状态：`POSTOPEN_DESIGN_SCREEN_NO_NEW_GEOMETRY`

证据等级：`E1_SYNTHETIC_OPENED_DEVELOPMENT_HYPOTHESIS_DESIGN_ONLY`

## 1. 要回答的窄问题

N1.8 显示 Camera-Block-6 与 Pose-Fourier-K6 在有限步 field/H1 上优于全局 Krylov-4，
但前者只保留 `57.071%` 的阻尼后额外 headroom，二者的 support-adjoint gain 也只有约
`8%` 至 `9%`。N1.9 只检验一个运行后已经明确提出、尚未重放的结构问题：

> 相机局部 contrast 与全局 `K=A P A^T` 响应组合后，能否同时保留 Camera/Pose 的有限步收益
> 和全局 Krylov 的 solver consistency？

它不是确认实验，不打开新 geometry，不训练网络，也不能更改 N1.7 或 N1.8 的判决。

## 2. 冻结的相机 contrast

当前 surrogate 固定为三台相机，按公开给重建器的 `camera_index = 0,1,2` 定义两个中心化、
单位 L2、彼此正交的 Helmert contrast：

```text
C1 = ( 1/sqrt(2), -1/sqrt(2),          0)
C2 = ( 1/sqrt(6),  1/sqrt(6), -2/sqrt(6))
```

contrast 只按 camera label 逐 ray 乘权，不读取真值、exact mismatch、family 或误差。若相机数
不是 3、标签不连续、向量不是有限值或最终 rank 不是 6，运行必须 fail closed。

## 3. 两个且仅两个候选

记：

- `d`：部署可见的 component-damping correction；
- `r=(b-Ax_warm)/scale`：部署可见 warm residual；
- `K=A P A^T`：与有限步求解器 support 一致的 measurement normal map；
- `orth(...)`：按给定顺序做两遍 modified Gram-Schmidt。

冻结向量顺序如下：

1. **Residual-Contrast Global-K6**

   `orth(d, r, C1 r, C2 r, Kd, Kr)`

2. **Damping-Contrast Global-K6**

   `orth(d, r, C1 d, C2 d, Kd, Kr)`

不允许加入 per-case exact mode、fit PCA、额外相机块、额外 normal probe、偏置向量或运行后混合。

这个二选一只隔离“camera contrast 乘在 `r` 还是 `d`”这一窄变量。它不是 Camera-Block、
Pose-Fourier 与 Krylov 的纯机制消融：Camera-Block 同时拆分 `d/r`，Pose-Fourier 把调制后的
residual 送入 normal probe，而本设计只计算全局 `Kd/Kr`。与 Krylov-4 的比较也只匹配算子调用
预算、不匹配 rank；任何结果都不得写成三种机制的因果优劣。

## 4. 完全匹配的成本与半径

两候选均固定：

- warm CGLS：`12F/12A^T`；
- warm projection：`1F`；
- `Kd,Kr` setup：`2F/2A^T`；
- correction 后 refine：`10F/10A^T`；
- 总部署成本：`25F/24A^T`；
- high-order teacher 调用：`0`（teacher 只作 evaluator control，不能进入候选）。

整个 correction 的可见 trust radius 保持：

```text
R = min(2 ||d||, max(||d||, 16 ||r||))
```

半径不读取 exact target；measurement projection oracle 只用于回答“该表示是否有 headroom”，
不是可部署算法。

## 5. 冻结的 17 项重建门与机制分级

所有阈值逐项继承 N1.8，runner 必须验证配置完全相同：

1. mean field gain vs CGLS-24 `>=5%`；
2. mean H1 gain vs CGLS-24 `>=3%`；
3. mean field gain vs damping `>=1%`；
4. 比冻结 N1.7 field gain 再高 `>=0.5` 个百分点；
5. worst-case field gain vs CGLS-24 `>=0`；
6. worst-geometry field gain vs CGLS-24 `>=0`；
7. 相对 CGLS-24 的 `>1%` harm rate `=0`；
8. 相对 damping 的 `>1%` harm rate `=0`；
9. 每个 phantom family 相对 damping 的 mean field gain 均严格为正；
10. exact-oracle field gain retention `>=70%`；
11. 阻尼后额外 headroom retention `>=60%`，定义为
    `sum(E_damping-E_candidate)/sum(E_damping-E_exact)`；
12. low forward calls `=25`；
13. low adjoint calls `=24`；
14. high-order forward calls `=0`；
15. high-order adjoint calls `=0`；
16. 每个 case basis rank 精确为 `6`；
17. 最大正交缺陷 `<=1e-10`。

只有 17 项全过才进入内部角色分级：`P A^T` gain `<0` 一律 NO-GO；`[0,50%)` 的内部代码标签
仍沿用 `SOLVER_AWARE_REPRESENTATION_ELIGIBLE`，但由于门槛只是非负且诊断读取 evaluator exact
mismatch，对外只能写 `NONNEGATIVE_SUPPORT_ADJOINT_SCREEN_ONLY`；`>=50%` 才可称
forward-correction representation。前两种 eligible role 都可以授权一个 representation 去做新
split 预注册，但 `REPRESENTATION_NO_GO` 永不在白名单中。

另外冻结两项本机配对 solver-path 成本非劣门。对每个相同 geometry/family，以 component damping
为参考，计算 `shared warm + basis setup + refinement` 的计时比值：median 必须 `<=1.25`，P90
必须 `<=1.50`。二者任一失败均不得授权。measurement projection oracle 的系数求解没有进入计时，
所以这不是可部署方法的 end-to-end runtime；它只描述当前 Mac 上、oracle 系数给定后的求解路径，
也不外推到其他硬件或内存成本。

当前候选没有 covariance head 或 majorizer，因此 Schur 门为
`NOT_APPLICABLE_NO_COVARIANCE_OR_MAJORIZER`，不得伪填“0 次违反”。若后续引入 covariance、
metric 或 preconditioner，必须另行冻结 SPD/Schur 审计。

## 6. 运行与停止规则

1. 先提交本冻结、配置、实现和测试，再运行完整 6 geometry/12 paired fields。
   配置同时冻结 T0、N1.5、N1.7 与 N1.8 的关键 config/summary/manifest，以及实际复用的
   fixture、operator、teacher、N1.5--N1.8 Python module/runner 的 SHA-256；source path 或 hash
   任一漂移都必须在 case preparation 前终止。完整运行还必须拒绝未提交或相对 HEAD 有修改的
   N1.9 config、freeze、model、runner 与 tests。
2. 允许用 `--seed-limit` 做接口 smoke，但 smoke 必须强制输出 `N1_9_SMOKE_NONDECISIVE`，
   不得授权候选、关闭分支或改变候选、阈值与顺序。
3. 完整运行后不根据结果改 contrast、rank、半径、门槛或预算。
4. 若恰有候选通过全部门，只授权为新 split 预注册一个候选；仍不训练 learner、不声称增益。
5. 若两个候选都失败，状态固定为
   `N1_9_RANK6_CAMERA_GLOBAL_K_BRANCH_CLOSED`，关闭 rank-6 camera/global-K 拼接分支，
   不继续用同一开发集堆 basis 或网络。这里关闭的只是这两个预冻结、rank-6、三相机 synthetic
   复合表示候选，不代表排除了所有 camera/global-K 方法。

## 7. 允许与禁止的论文表述

允许：这是在已打开 synthetic development 上、相同算子预算下对两个预先冻结 representation
的 mechanism-oriented falsification screen。

禁止：把 oracle 系数称为模型；把均值优势称为泛化；把 synthetic surrogate 称为真实 BOST；
把通过旧开发集称为确认；把失败后改门或改结构的结果并入同一次预注册。

当前 fixture 每台相机使用相同 detector grid，Helmert camera-level contrast 扩展到 ray space 后才
保持对称。真实 BOST 若存在不同 ROI、mask、缺失像素或置信度，必须按有效 ray measure 重新加权
并重新做正交性审计，不能直接复制这里的系数。当前实现也会在三相机有效 ray 数不相等时直接拒绝。
