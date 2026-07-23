# D0.6 三参数化 matched-budget 设计预检

> 日期：2026-07-23
>
> 主状态：`D0_6_DESIGN_PREFLIGHT_PASS_EXECUTION_CLOSED`
>
> 机械预检：136 / 136
>
> 训练执行：0
>
> 突破监测：**没有突破**

## 先说人话

这一步没有训练模型。我们先把 `S0_VOXEL`、`S1_FOURIER` 和 `S2_BOUNDED_RESIDUAL` 放进同一套物理前向、数据拆分、参数量级和投影工作预算里，再让独立审计专门找协议漏洞。

审计确实抓到一个会破坏公平性的错误：第一版 LR 合格规则要求在第 4 步 checkpoint 上重新计算 1,920 条 fit 射线，但账本漏掉了四个候选各一次的完整 forward。如果直接开跑，实际工作量会比声称的预算大。这个错误在任何训练前被修复，第一版仍保存在 commit `01ce64d`，没有被覆盖。

协议修复版 commit `f721eca` 通过 136 / 136 项只读预检。第一次保存包随后被 Pages 隐私门拒绝，因为环境详情包含本机绝对用户路径；它没有上线，现保留在私有隔离区。验证器只把该详情改成相对标识 `.venv`，没有改协议、输入、预算或阈值。可公开 v2 仍为 136 / 136，并绑定 publish-safe commit `1f0136c`。这些检查证明协议常量、公开输入身份、确定性拆分、参数计数、预算算术、角色访问合同和 claim closure 相互一致，不证明任何模型有效。

## 三条 arm 如何对齐

| arm | 参数量 | 前 80 步 | 后 30 步 | 当前作用 |
|---|---:|---|---|---|
| `S0_VOXEL` | 31,875 | 体素参数 | 继续体素参数 | 最低复杂度控制组 |
| `S1_FOURIER` | 31,873 | trunk + base | 继续 trunk + base | 神经隐式基线 |
| `S2_BOUNDED_RESIDUAL` | 31,970 | 与 S1 完全相同 | 冻结 base，只训练 97 参数有界残差头 | 候选改进 |

最大参数差是 97，按最小参数量归一为 0.3043%。三条 arm 都先生成同一个 `27 x 53 x 27` 栅格，乘同一个解析边界包络，再进入同一个 128 点直线射线 forward。这里比较的是三种离散参数化，不允许 Fourier MLP 绕过共同栅格获得额外积分分辨率。

`S1` 与 `S2` 的 LR trial 和正式前 80 步必须逐步一致。每一步都绑定 batch SHA、loss 的 float64 十六进制值、trunk/base 梯度范数、参数 SHA 和 Adam 状态 SHA。第 81 步前丢弃 base optimizer，为 residual head 新建 step-0 Adam，避免隐藏的 bias-correction 差异。

## 修正后的 matched projection-work 账本

每个 `arm x seed` 的主账本为：

| 阶段 | forward calls | VJP calls | forward rays | VJP rays |
|---|---:|---:|---:|---:|
| LR 四候选训练 | 16 | 16 | 7,680 | 7,680 |
| LR step-4 fit-union 复评 | 4 | 0 | 7,680 | 0 |
| LR 完整 dev 复评 | 4 | 0 | 2,688 | 0 |
| 110 步正式训练 | 110 | 110 | 52,800 | 52,800 |
| **合计** | **134** | **126** | **70,848** | **60,480** |

总 ray evaluation 为 131,328；乘 128 个积分点后是 `16,809,984 RQWU`。RQWU 只匹配物理 ray-quadrature 工作，不代表 MLP/体素端到端 FLOPs 相同。因此未来还必须单列 wall time、峰值 RSS、参数阶段和全部 postseal scoring 成本。

## audit 和真值怎么隔离

协议现在定义四个互斥角色：

1. `SPLIT_BROKER` 是唯一能读取 manifest、二维 mask 和 12 张公开观测图的预处理进程，只输出带哈希的 fit/dev/audit shard；
2. `TRAINER` 只能挂载 fit/dev，不能看到原图、audit、GT、CGLS-TV 或三维 support；
3. `AUDITOR` 只有在九个 checkpoint 与全部 manifest 封存后才能读取 audit shard；
4. `GT_SCORER` 最后才读取 GT、CGLS-TV 和三维 support，且不得把结果反馈给训练。

这一结构已经写入输入身份与协议，但尚未实现成进程隔离和负向挂载测试。因此 `split_broker_implemented=false`，也正因为如此，正式训练仍然关闭。

## 136 项预检检查了什么

- 协议与输入 SHA 锁；
- 外部公开 release commit `a385cce...f5604` 和 18 个输入文件哈希；
- 14 个 broker 输入、3 个 postseal GT scorer 输入和 1 个 provenance 文件的角色互斥；
- 12 x 512 射线选择、每视角 fit/dev/audit 哈希与 10 个 batch 哈希；
- 三个网络/体素参数量和 16,809,984 RQWU 算术；
- LR 合格条件、S2 状态机、G0/G1/G2/G3 阈值和失败路由的精确语义；
- 删除或修改关键授权、预算、门槛、环境、边界包络时必须失败的 mutation tests；
- `.venv`、arm64、NumPy 2.4.6、SciPy 1.17.1、PyTorch 2.13.0；
- 协议与输入身份来自 post-audit commit `f721eca`；可公开验证器与三项 source-binding 均绑定到 commit `1f0136c`；
- 所有训练与科学主张保持 false。

准确状态是 `VALID_D0_6_FROZEN_DESIGN_PREFLIGHT_ONLY`。验证器没有调用物理 forward、没有执行 optimizer，也没有生成 checkpoint。

发布边界也单独过了一道门：v1 虽然数值上 136 / 136，但保存了本机 `.venv` 的绝对路径，因此过滤构建 fail-closed。v1 没有公开，移动到私有隔离区；v2 只改变环境详情的序列化，输出中不再包含 `/Users/`、用户名、凭据或私有原始资产。这个修复不是新的研究结果，也没有重新定义任何科学 PASS 条件。

## 现在允许与不允许的事

当前唯一为 true 的字段是：

`frozen_design_consistent=true`

仍为 false：

- split broker 已实现；
- runner 已实现；
- mechanical dry-run 通过；
- 正式训练或参数化筛选完成；
- 三维重建、算子学习、跨 field / geometry 泛化；
- 真实 BOST / OERF；
- 优于 DeepONet、FNO、NeRIF 或 NIRP；
- 新算法、论文成功和突破。

## 下一步

1. 实现只接收 14 个 allowlist 文件的 split broker，并为 shard 互斥、并集 6,144 rays、内容 SHA 和越权路径写负向测试；
2. 实现三条 arm 与共同 forward 的 runner，先做参数梯度、step-80 prefix、预算计数和失败保存测试；
3. 把 broker 与 runner 提交到 Git 后做一次不产生研究结果的 mechanical dry-run；
4. dry-run 也通过后，才另开正式执行授权；
5. 单公开 Phantom 的筛选无论结果多好，都只能选择下一阶段候选，不能替代 fresh 多 field 与跨几何实验。

## 证据入口

- 设计说明：`docs/open_nir_bos_d0_6_matched_budget_design_2026-07-23.md`
- 机器协议：`learning_labs/open_nir_bos_d0_6_matched_budget_protocol.json`
- 输入身份：`learning_labs/open_nir_bos_d0_6_input_identity.json`
- 136 项可公开预检：`learning_labs/results/open_nir_bos_d0_6_protocol_preflight_v2/validation.json`
- 当前授权覆盖层：`docs/open_nir_bos_d0_6_preflight_authorization_overlay.json`

## 最终证据边界

**真实增量：独立审计在训练前发现预算漏项；v1.1 修复了预算、S1/S2 轨迹身份、分支门、共同边界先验和四角色访问合同，并通过 136 / 136 项 source-bound 预检。**

**没有得到：任何训练结果、重建指标、算法优势、算子学习、真实 OERF、泛化、论文成功或突破。**
