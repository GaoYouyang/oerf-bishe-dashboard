# PoolFire C 路线 v9 三角色代码门

更新时间：2026-07-26

## 一句话结论

Cross14 现在已经被拆成 `fit -> deployment -> score` 三个文件角色，并在两帧合成
fixture 上完成 fresh-process 执行和数值逐数组重算。当前可信状态只到：

```text
PASS_SYNTHETIC_V9_THREE_ROLE_CODE_GATE_ONLY
```

这证明 synthetic plumbing、几何与模型身份、exact K2 调用账、评分公式和原子输出
在代码层可执行；它不证明正式 v9 编排、跨轨迹精度、速度、内存、真实 BOST、泛化或
论文成功。当前 worker 会明确拒绝任何 `formal_development` 请求。

## 为什么要拆成三个角色

如果拟合、预测和评分都在一个进程里，代码即使“声称没看真值”，也很难排除误读
heldout teacher 或 truth 的路径。v9 因此固定三种精确文件集合：

| 角色 | 允许看到 | 明确看不到 |
|---|---|---|
| fit | 训练轨迹的 raw BP、equalized BP、K4 teacher | heldout observation、heldout teacher、truth、fresh holdout、test |
| deployment | 冻结模型、fit 凭据、heldout observation、冻结 geometry | 任何 truth、teacher、训练 BP |
| score | 已原子发布的 initializer/K2 candidate、heldout teacher/truth、observation、geometry | 模型选择、lambda 选择、重新拟合 |

每个请求目录只允许事前列出的普通文件。额外文件、目录、符号链接、多硬链接文件、
重复 JSON key、非有限 JSON、payload SHA 漂移、源码 SHA 漂移或外层 request digest
不一致都会在输出前拒绝。

## 绑定了哪些身份

每次运行同时绑定：

1. v9 协议、Cross14 核心、三角色合同、worker、CGLS/PCGLS、equalizer 和 straight-ray
   operator 的源码 SHA；
2. Python、NumPy、BLAS、体系结构和线程环境；
3. trajectory 角色、frame count 和 execution scope；
4. geometry 坐标及其 pair identity；
5. 从同一个 inverse operator 解析重建的 `diag(A^T A)` equalizer；
6. fit receipt、模型 payload、模型数值 digest 和训练 trajectory 列表；
7. 上一阶段 checksums 与最后写入的 `READY.json`。

后实现审计发现，原版本虽要求 `formal_development=101` 帧，却没有实现冻结协议要求
的 nested complete-trajectory LOTO、完整 lambda 网格和 one-standard-error rule，
因此可能把调用者手选的 lambda 误标成 formal PASS。修复后，本 worker 只接受
1–4 帧 `synthetic_fixture`；即使给足 101 帧也会 fail closed。`p22-s01`、预注册但
未释放的 `p45-s03` 和两条 untouched test 继续锁住。

## 部署主账与评分旁账

单帧 Cross14-K2 部署主账保持：

| 阶段 | A | A^T |
|---|---:|---:|
| raw/equalized BP | 0 | 1 |
| 非零初值 residual | 1 | 0 |
| exact CGLS K2 | 2 | 2 |
| 合计 | 3 | 3 |

score 会从同一 observation、同一 geometry 重新运行一次 Zero-K4，并逐数组核对
外部 teacher，然后再投影 candidate 和 teacher。因此离线旁账为每帧
`6A+4A^T`：K4 复算 `4A+4A^T`，两次评分投影 `2A`。它只用于防伪与离线评价，不能
混进或移出 deployment 的主账。

三项逐帧指标为：

- gauge-centered field relative-L2；
- 按真实粗网格间距计算、拼接三个方向 forward difference 的 gradient relative-L2；
- observation relative-L2。

p90 固定采用 `method="higher"`。score 同时保存 joint matched、overall harm 以及
field/gradient/observation 三个独立 harm 标记，但本代码门没有执行跨轨迹论文判决。

## 本轮验证结果

```text
源码提交                              12ea0d5
Ruff                                  PASS
Python byte compilation               PASS
三角色定向测试                         19 passed
与 Cross14/baseline/protocol/page 联合  112 passed
git diff --check                      PASS
后实现只读复审                         P0=0 / P1=0
```

数值 validator 不导入正式 Cross14 核心、三角色合同、worker、baseline CGLS 或正式
评分 helper。它重新实现并计算：

1. Cross14 scaler、14 维特征、Gram、float64 EVD ridge；
2. geometry、equalizer 和模型预测；
3. exact CGLS K2；
4. field、gradient、observation、matched 与 harm。

fit、deployment 和 score 三阶段的最大绝对差均为 `0.0`，且验证前后请求与结果树
逐字节不变。但 validator 仍复用了正式的
`ProjectionFirstInteriorStraightRayOperator` primitive，因此这不是 operator 独立
重写，也没有重算 nested folds、全部 controls、runtime/RSS 和最终论文判决。准确状态
为：

```text
PASS_SYNTHETIC_V9_THREE_ROLE_NUMERICAL_RECOMPUTATION_WITH_SHARED_OPERATOR_PRIMITIVE_ONLY
```

## 现在仍然没有证明什么

```text
formal_v9_scientific_gate_implemented=false
formal_101_frame_run_completed=false
trajectory_split_proven=false
process_truth_free_proven=false
clean_detached_worktree_proven=false
independent_operator_recomputation_proven=false
independent_full_protocol_validation_proven=false
development_LOTO_completed=false
matched_accuracy_gate_passed=false
wall_time_speedup=false
whole_pipeline_peak_memory_measured=false
fresh_v9_holdout_opened=false
untouched_test_opened=false
neural_training_authorized=false
algorithm_breakthrough=false
paper_success=false
```

`process_truth_free_proven=false` 是刻意保留的严格边界：当前证明的是三个普通
Python 进程收到的请求目录成员不同，没有证明受限文件系统、网络拒绝或全文件系统
不可见性。源码虽然逐文件 hash 绑定，也尚未在 clean detached worktree 的单一提交上
运行。输入检查后仍按路径打开，面对主动并发换包时还存在 TOCTOU 残余风险。

后实现审计原先报告的三个 P0 分别是：formal 选参可绕过、fit 数组缺语义 provenance、
score observation/teacher 可换包。当前处理是：

1. 彻底禁止 formal scope，避免把不完整编排包装成正式结果；
2. deployment receipt 新增 observation SHA，score 必须逐字节匹配；
3. score 要求 truth/teacher/candidate/initializer 都满足冻结 gauge，并从同一
   observation/geometry 复算 Zero-K4；
4. fit 的 trajectory-axis、pair registry、teacher generation provenance 留给下一层
   formal orchestrator；在其完成前不得生成正式凭据。

## 下一步唯一有效门

1. 实现单独的 formal orchestrator，机器绑定五个 outer fold、每 fold 的 inner
   complete-trajectory LOTO、完整 lambda grid、one-standard-error rule 和禁止事后
   改网格；
2. 为每个 fit 数组绑定 trajectory-axis、pair registry、observation、geometry、
   equalizer 与同源 K4 teacher 的生成 receipt，并在 clean detached commit 上运行；
3. 通过外层编排器为五条 fit trajectory 生成 101 帧请求，对 heldout trajectory
   分别运行 deployment/score，p14 只作 mandatory development veto；
4. 在完全相同 geometry、solver、K 和 metric 下跑 Zero-K3、Zero-K4、BP、
   normalized BP、line-search、DCT、dual-ridge 与 geometry-PCGLS；
5. 逐 trajectory 报 field/gradient/observation p50、p90、worst、matched、三项 harm、
   `A/A^T`、端到端 wall 和 fresh whole-pipeline peak RSS；
6. 五条 outer LOTO 与 p14 veto 全过，才允许冻结一次性 `p45-s03` release；在此之前
   不训练 3D U-Net、FNO、UNO 或 DeepONet。

**突破监测：没有算法突破。** 本轮新增的是可信的软件与证据边界，不是性能结果。
