# BLASTNet v45-v47：三种修复都没有救活单 curved-adjoint 路线

日期：2026-07-29  
数据角色：BLASTNet 预混 H2-air DNS，`phi=0.5`，外部门开封后的机理开发  
综合状态：`POST_OPEN_MECHANISM_CHAIN_V45_V47_BOUNDED_NEGATIVE`

## 1. 为什么连续做这三次实验

v44 已经找到一条真正离开旧 straight-CGLS span 的 curved-adjoint 方向。它在四个
快照上都改善 field，却没有同时守住 gradient 和 observation 的 `1.01 x Direct-K4`
兼容门。

接下来最便宜、最容易排除的三个解释依次是：

1. **频谱不对：**方向高频过强，固定 Sobolev 平滑也许能保住 gradient；
2. **整体步长不对：**方向本身没错，只是 correction 放大或缩小错了；
3. **五个分量比例不对：**整体缩放不够灵活，但旧四方向与新 curved 方向独立调权
   后也许存在完整门候选。

v45、v46、v47 分别检验这三个解释。它们没有换数据、放松门槛或打开新测试集。

## 2. 三次实验实际做了什么

### v45：固定 Sobolev 方向族

对同一个 raw curved-adjoint 方向施加固定 `lambda = 0.25, 1, 4, 16` 的
Sobolev 型滤波，同时保留 raw、four-direction 和 straight-continuation controls。
候选生成不读 field truth，但这只是 scratch 诊断，没有独立数值重放。

结果：固定 lambda、observation 选择的 lambda、以及事后看真值选出的 family member
都没有让任何快照通过完整门。

### v46：整条 correction 的幅度路径

冻结 v44 场，扫描

```text
x(alpha) = Direct-K3 + alpha * (x_v44 - Direct-K3)
alpha = 0, 0.025, ..., 2
```

总共 `81 x 4 = 324` 个候选，加上参考重放共 `332 F`；没有 JVP/VJP。独立程序重新
构场、重跑 curved forward 和门禁，指标与误差比最大差分别为 `3.33e-15` 和
`2.33e-15`。

结果：四帧都没有通过网格点，也不存在固定 alpha 通过四帧。由于只检查了冻结网格，
不能写成连续域数学不可能。

### v47：五个系数独立变化

搜索空间为四个 frozen straight-CGLS directions 加一个 v44 curved-adjoint
direction。field 和 gradient 必须严格满足冻结兼容约束，objective 只最小化精确
curved observation residual。

每帧使用两个相关但不同的可行起点：

1. v43 的四维可行点加 `w5 = 0`；
2. v44 权重从零点径向收缩到 truth-feasible 区域。

每个起点最多运行 8 次 JVP trust-region 外循环。总共精确评分 48 个候选，账本为：

```text
straight basis: 16 A + 16 A^T
curved work:    52 F + 310 JVP + 0 VJP
```

独立程序不导入 v47 runner，重新构造五个方向、候选场、curved prediction、三项指标
和调用账。最大差为：

```text
metric       3.33e-15
ratio        2.33e-15
joint metric 1.11e-15
```

报告在验证前后未改变。

## 3. 核心结果

下表给出每种机制在每个快照上能找到的最小“最差指标比”。小于等于 `1.01` 才算通过。

| 机制 | S1 | S2 | S3 | S4 | 完整门 |
|---|---:|---:|---:|---:|---:|
| v45 最佳固定 Sobolev | 1.0155 | 1.0302 | 1.0252 | 1.0427 | 0 / 4 |
| v46 最佳幅度网格点 | 1.0147 | 1.0208 | 1.0113 | 1.0423 | 0 / 4 |
| v47 五维受约束搜索 | 1.0165 | 1.0458 | 1.0111 | 1.0434 | 0 / 4 |

v47 的逐指标比最能解释失败：

| 快照 | field / K4 | gradient / K4 | observation / K4 | 精确候选数 | 通过数 |
|---|---:|---:|---:|---:|---:|
| S1 | 0.978542 | 1.010000 | 1.016493 | 12 | 0 |
| S2 | 0.983386 | 1.010000 | 1.045775 | 7 | 0 |
| S3 | 0.979858 | 1.010000 | 1.011126 | 17 | 0 |
| S4 | 0.978027 | 1.010000 | 1.043375 | 12 | 0 |

四个最优点的 field 都有余量，gradient 却全部贴到允许上限，observation 仍越线。
因此当前矛盾不是 field 轮廓太差，而是**在这组方向和局部搜索中，继续改善 curved
observation 会伤害 gradient**。S2、S4 的缺口较大；S3 只高 `0.001126`，所以仍不能
草率宣布五维空间无解。

## 4. 成功了什么，失败了什么

成功的是问题定位：

- 固定平滑不能救活方向；
- 整体放缩不能在冻结网格上救活方向；
- 两起点局部五维调权也没有找到通过候选；
- 48 个 v47 候选的指标和账本已经独立重算；
- 因此现在没有依据训练一个网络去预测这五个系数。

失败的是算法门：

```text
five_coefficient_learning_target_authorized=false
deployable_algorithm=false
matched_accuracy=false
speedup=false
real_BOST=false
algorithm_breakthrough=false
paper_success=false
```

这不是“没有产出”。它关闭了三个很容易浪费数周训练时间的解释，并把新模型必须新增
的信息限定为：要打破当前 gradient-observation 冲突，而不是继续平滑或缩放同一组
方向。

## 5. 不能夸大的边界

v47 只用了两个相关起点，每帧留下 7 到 17 个精确候选。它证明的是固定预算局部搜索
没有找到见证，不是五维可行域的全局不可行证明。

因此当前不允许写：

```text
the five-dimensional space is mathematically infeasible
the curved-adjoint direction is useless
a six-dimensional method is already superior
the method generalizes to PoolFire or real BOST
```

## 6. 由结果决定的下一次真实实验

在增加第六个方向前，只再运行一次固定预算的五维全局反例搜索：

1. 五个方向、truth constraints、评价门和数据保持不变；
2. 在精确二次可行域内生成固定、可复现的全局覆盖点；
3. 每点只做一次 exact curved forward；
4. 仅从 observation 最好的少量点启动短程 exact-gradient refinement；
5. 先跑缺口最大的 S2；仍失败就停止无边界五维调参。

如果四帧都找到完整门候选，才说明“五系数学习目标存在”；如果任一帧仍失败，只能写
“固定预算全局反例搜索未找到可行点”，然后关闭五系数路线，转向一个真正改变 span 的
二阶 curved Gauss-Newton/Krylov 方向。不能通过继续加起点或放松 `1.01` 门救结果。

## 7. 讲人话

我们先试了“把新方向磨平一点”，再试“把它整体调小或调大”，最后允许五个旋钮分别
调。三种办法都没把四个快照救过线。最关键的现象是：场的大轮廓已经更好，但一逼近
相机观测，梯度细节就卡到红线。

所以此刻训练网络不是勇敢，而是让网络背一个还没有证明存在的答案。下一次只花一笔
封顶预算确认五维空间里是否漏掉了远处的可行点；确认没有后，才新增真正不同的物理
方向。

公开图表：
[v45-v47 机理链总图](../asset_viewer.html?asset=assets%2Fblastnet_h2air_phi05_curved_followups_v47.png)

机器可读脱敏摘要：
[v45-v47 public summary](../docs/blastnet_h2air_phi05_curved_followups_v47_public_summary.json)
