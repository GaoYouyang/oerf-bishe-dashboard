# N2-PVGR N4.1 执行修正预注册

日期：2026-07-18

## 1. 修正范围

N4 v1 的科学协议、16 对样本、H256/H512/H1024/H2048 路线、阈值、指标、topology gate、图表和停止规则已经在提交 `1761ff4eca74a904350ecd90ddfe2e83fffc47a9` 中冻结。首次正式执行在第二格触发 H2048 分支时因控制流缺陷退出，详见 `n2_pvgr_n4_v1_aborted_execution_record_2026-07-18.md`。

N4.1 不重新定义科学协议。它只修复以下执行顺序：

1. 运行 H256、H512、H1024；
2. 独立计算完整的 inherited H1024 gate bundle；
3. 若全部通过，不运行 H2048；
4. 若任一门失败，运行 H2048；
5. 无论是否升级，最终都调用冻结 v1 runner 的 `_cell_decision` 生成逐格结论；
6. 断言预判 gate bundle 与冻结最终函数中的 H1024 gates 完全相同。

除此之外不允许任何变化。

## 2. 已知观测披露

在 N4.1 封存前已经知道：

- v1 第一格 `smooth-s1729-orientation_58-wide__stress_1` 的控制台状态为 H1024 PASS；
- 第二格 `smooth-s1729-orientation_58-narrow__stress_1` 进入了 H2048 升级路径；
- 尚无任何 H2048 checkpoint、正式结果 bundle 或最终 machine decision。

这些信息没有用于修改样本或阈值。即便如此，N4.1 仍应被标为 post-N3 selected mechanism audit，不能冒充独立外部验证。

## 3. 数据隔离

N4.1 使用新的 formal work 和 formal output 路径。v1 的 6 个 checkpoint 不复制、不链接、不读取、不复用。N4.1 checkpoint 的 preregistration SHA-256 来自 N4.1 amendment config，因此 v1 payload 即使被误放入新目录也会因哈希不一致而拒绝。

## 4. 继承门槛

N4.1 从 frozen base config 直接读取，不在 amendment config 中复制：

- 16 对 failure/control；
- 256 common Sobol rays；
- H256/H512/H1024 与条件 H2048；
- output 与 matched-residual relative-L2；
- contraction ratio 与 machine-zero fallback；
- finite/domain/stencil/direction/topology/frustum gates；
- N3 指标重现门；
- mixed reference 与 tiny field-JVP/VJP 权限边界。

amendment validator 必须同时验证 frozen base attestation 和 N4.1 attestation。任一 base file 漂移均拒绝运行。

## 5. machine decision

机器判定字符串及含义完全继承 N4 v1：

- 32 格最终 reference 均通过：`EVALUATOR_CONVERGENCE_CLEARED_FOR_TINY_FIELD_JVP_VJP_GATE`；
- 否则：`FAIL_CLOSED_EVALUATOR_REMAINS_UNAUTHORIZED`。

前者只授权下一步 tiny field-JVP/VJP gate；不授权算法优越、三维重建、真实 BOST、跨分布泛化或论文主张。

## 6. attestation 顺序

1. 提交 amendment config、runner、测试、validator、本协议与 v1 中止记录；
2. 确认 N4.1 formal output/work 均不存在；
3. 生成并单独提交 N4.1 attestation；
4. `validate-only` 通过后才启动正式 N4.1；
5. 正式结果完成后运行独立 validator，复算门、计数、H2048 路由约束、CSV、查询账本与 manifest 哈希。
