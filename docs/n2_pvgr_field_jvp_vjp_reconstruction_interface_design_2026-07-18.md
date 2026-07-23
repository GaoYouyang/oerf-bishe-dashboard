# N2-PVGR field JVP/VJP 到三维重建的最小接口设计

> 日期：2026-07-18
>
> 状态：`DESIGN_ONLY_NO_96_CELL_NO_RECONSTRUCTION_RESULT`
>
> 证据边界：本文只定义一个待实现、待测试的最小接口。没有运行正式 96-cell，没有三维重建结果，没有真实 OERF 数据结果，也不声称优于 NeRIF、TDBOST、DeepONet、FNO 或 FFNO。

## 1. 目标与最小边界

目标是把同一个离散 BOST 曲光线 forward 闭合成下列链路：

```text
active field parameters theta
  -> values_zyx
  -> 8-view curved-ray BOST forward F(theta)
  -> field JVP: J(theta) v
  -> field VJP: J(theta)^T q
  -> 6-view data term / regularizer
  -> matrix-free 3D reconstruction update
  -> 2 held-out view reprojection after freeze
```

第一版只支持一个 `float64` 体素场、固定的中央差分、固定 RK4 步数和固定视角/光线 manifest。可微核心不负责画图、存文件、记帐或做 Python 路由决策。神经场、cone-ray、OCBH 快速路由和 4D 时间/张量参数化都是后续扩展，不是首个重建 smoke 的一部分。

## 2. 现有四个模块的可训练性审计

### 2.1 `field_dependent_ray.py`

`trace_field_dependent_rays` 的 RK4 位置和方向本身保留场依赖。`path_integrated_deflection(..., detach_path=False)` 同时保留：

1. 场对积分核的直接导数；
2. 场经过 RK4 轨迹再影响观测的导数。

因此它是最小完整曲光线 field JVP/VJP 的物理核心。下列 detach 只应作诊断，不应进入 loss：

- `_ray_rhs` 对 `position.detach()` 求 domain/stencil margin；
- `trace_field_dependent_rays` 对轨迹和方向 detach 后生成 Python 浮点诊断；
- `path_topology_diagnostics` 对场、位置、方向和投影轴全部 detach；
- `relative_l2` 明确是 detached metric。

`path_integrated_deflection(..., detach_path=True)` 不能用作完整曲光线训练 forward；它故意删掉 field-to-trajectory 导数，只适合分解直接项与轨迹项。首版还应固定 `gradient_mode="central"`，避免 `coupled_automatic_spatial_gradient` 内嵌 `autograd.grad` 和 `requires_grad` 分支同 `torch.func` 组合在一起。

### 2.2 `discrete_rk4_jvp_predictor.py`

这是对弯曲同伦标量 `epsilon` 的 forward-mode JVP 教师，不是对场参数的 JVP：

- `_validate_inputs` 返回 `values.detach()` 和 `states.detach()`；
- `_bend_parameterized_discrete_map` 只把 `bend_strength` 留在图中；
- `DiscreteRK4JVPResult` 的 `base_output_uv`、`residual_prediction_uv`、轨迹、切线、risk 和 mask 全部 detach；
- `stop_gradient_applied=True`，记帐也明确为零次 field VJP。

所以当前公开函数不能从重建 loss 反传到 `values_zyx`。若未来要优化“场依赖的弯曲同伦导数”，需要计算混合导数 `d/dtheta [dY(theta,epsilon)/depsilon]`，不能只删除最后一个 `.detach()` 就声称完成。

### 2.3 `operator_consistent_homotopy_predictor.py`

当前 OCBH 也是 detached development baseline：

- `_validate_inputs` detach 场和 pupil states；
- `_operator_consistent_coefficients` 用 `positions.reshape(...).detach().requires_grad_(True)` 构造坐标导数点；直线 base path 本来与场无关，这一点不是主要问题，但它必须与场图的保留分开说明；
- `central_jacobian` 的 `autograd.grad(..., create_graph=False)` 不保留后续 field VJP 所需的高阶图；
- `refractive_index`、`forcing`、位置/方向 Jacobian 在 helper 返回前 detach；
- 入口又 detach 几何量，最终 result 的所有物理张量再次 detach；
- `stop_gradient_applied=True`、`reverse_mode_field_vjp_evaluations=0`。

因此它现在可以核对离散 `epsilon` JVP，不能作为可训练三维重建 renderer。后续改造必须保留同一 central operator 对 field 的图，并独立过 field FD/JVP/VJP 门。

### 2.4 `shared_straight_state.py`

`build_straight_path_state` 没有 detach `values_zyx`，`scalar_values -> gradients -> refractive_indices -> curvatures -> projected_outputs` 这条物理链保留反传，可作为 straight-ray field JVP/VJP 基线。

`_cell_ids_xyz` 中的 `positions.detach()` 只用于生成离散 cell-id 诊断，不会截断 `projected_outputs` 的场梯度。但 `bool(torch...)` 验证和 Python 数据类构造会带来 host sync/函数变换风险；它们应位于纯张量 forward 核外的 preflight/audit wrapper，不应成为 loss 的可微输入。

## 3. 冻结的数据布局

观测行顺序固定为 `view-major -> ray-major -> (u,v)`。不允许根据当前场、残差或 held-out 误差重排/删除光线。

```python
@dataclass(frozen=True)
class BOSTRayBatch:
    view_ids: tuple[str, ...]                 # exactly 8 unique IDs
    rigs: tuple[SyntheticRayRig, ...]         # one frozen rig per view
    pupil_states_by_view: tuple[Tensor, ...]  # each [R_v, 2], float64
    observed_uv_by_view: tuple[Tensor, ...]   # each [R_v, 2]
    valid_uv_by_view: tuple[Tensor, ...]      # each [R_v, 2], bool, frozen
    inv_std_uv_by_view: tuple[Tensor, ...]    # each [R_v, 2], frozen
    train_view_ids: tuple[str, ...]            # exactly 6
    heldout_view_ids: tuple[str, ...]          # exactly 2
    manifest_sha256: str
```

`valid_uv` 只能由冻结的标定/数据质量流程产生。不能在优化中用大残差作理由丢行。`inv_std_uv` 定义训练内积，但 raw forward 始终返回未白化的 detector-plane deflection；像素/物理单位转换必须作为冻结标定的线性层显式记录。

## 4. Field parameterization

首版不直接优化整个 `[Z,Y,X]` 网格，而优化 support 内、非边界的活跃体素向量：

```python
@dataclass(frozen=True)
class VoxelFieldSpec:
    shape_zyx: tuple[int, int, int]
    active_linear_indices: Tensor  # [P], int64, z-major, unique, frozen
    max_abs_value: float
    refractivity_scale: float
    refractive_index_floor: float
    manifest_sha256: str

def decode_field(theta_p: Tensor, spec: VoxelFieldSpec) -> Tensor:
    # theta_p: [P], float64
    active = spec.max_abs_value * torch.tanh(theta_p)
    flat = torch.zeros(math.prod(spec.shape_zyx), dtype=theta_p.dtype,
                       device=theta_p.device)
    return flat.scatter(0, spec.active_linear_indices, active).reshape(spec.shape_zyx)
```

这与当前 kernel 的定义一致：

\[
n(x)=1+\kappa\,\mathrm{values}(x),
\]

其中 `kappa = refractivity_scale`。配置预检必须满足

\[
1-\kappa\,\text{max_abs_value} > n_{\min}.
\]

活跃索引不包含最外一层和 support 外体素，因而实现零边界/Dirichlet gauge，避免在纯梯度 BOST 观测中优化不可辨的常数偏移。`theta_p=0` 是零扰动场。

神经场后续只能通过替换 `decode_field(theta, spec) -> values_zyx` 接入；forward、row layout、JVP/VJP 和预算合同不变。首版不同时改 field representation 和 ray physics。

## 5. Forward signature

对 `torch.func` 可见的正式签名只返回张量：

```python
def n2_curved_bost_forward(
    theta_p: torch.Tensor,
    batch: BOSTRayBatch,
    field_spec: VoxelFieldSpec,
    *,
    view_ids: tuple[str, ...],
    difference_step: float,
    step_count: int,
) -> torch.Tensor:
    """Return unwhitened deflection in frozen view/ray order, shape [M, 2]."""
```

冻结合同：

- `theta_p` 必须是 `[P]` `torch.float64`，且在同一 device；
- `view_ids` 必须是 batch 中的有序子序列；训练传 6 个 train IDs，最终审计传 2 个 held-out IDs；
- `gradient_mode` 首版隐式冻结为 `central`，不对外暴露可变选项；
- 每个 view 使用自己的 rig 和 pupil states，返回后按 `view_ids` 顺序 `torch.cat`；
- 物理路径等价于 `trace_field_dependent_rays(..., create_graph=True)` 后调用 `path_integrated_deflection(..., create_graph=True, detach_path=False)`；
- loss 不能使用 `RayTraceResult` 里的 Python 浮点诊断，也不能读 detached topology 作可微特征。

实现时应从现有 trace 抽出一个只返回 `positions/directions/projection_u/projection_v/step_size` 的 tensor-only 内核。当前 `_validate_points`、`_ray_rhs` 和 dataclass 构造中的 `bool/float/detach` 应留在图外 preflight/audit wrapper；不改变 RK4 stage、方向归一化、中点求积或 central-difference 算子。

## 6. Field JVP/VJP 与 dot test

JVP 和 VJP 必须由上述同一 forward closure 生成，不单独训练 adjoint：

```python
def field_jvp(theta_p, tangent_p, forward_closure):
    y, jv = torch.func.jvp(
        forward_closure, (theta_p,), (tangent_p,), strict=True
    )
    return y, jv

def field_vjp(theta_p, cotangent_uv, forward_closure):
    y, pullback = torch.func.vjp(forward_closure, theta_p)
    (jtq,) = pullback(cotangent_uv)
    return y, jtq
```

`tangent_p` 与 `theta_p` 同形，`cotangent_uv` 与 forward 输出 `[M,2]` 同形。若重建使用白化残差，`W` 必须显式放在 closure 外：数据项梯度为 `J^T W^T W(F-y)`，不得把不对称的白化藏进另一个 adjoint。

对每个冻结 test context，用非零随机 `v` 和 `q` 检查：

\[
e_{dot}=\frac{|\langle Jv,q\rangle-\langle v,J^Tq\rangle|}
{\max(|\langle Jv,q\rangle|,|\langle v,J^Tq\rangle|,10^{-12})}.
\]

首版 CPU float64 工程门为 `e_dot <= 1e-8`。同时必须做中心方向有限差分：

\[
e_{fd}(h)=\frac{\|[F(\theta+hv)-F(\theta-hv)]/(2h)-Jv\|_2}
{\max(\|Jv\|_2,10^{-12})}.
\]

用多个 `h` 检查先收敛后受舍入误差影响的趋势，预注册有效 `h` 上 `e_fd <= 1e-4`。若 `theta +/- h v` 改变了冻结 mask、cell/support crossing 签名、frustum 合法性或进入了 floor/domain 失败，该方向不记为“通过”，而是标记 `DERIVATIVE_CONTEXT_CHANGED_FAIL_CLOSED`。

dot test 只证明 JVP/VJP 闭合，FD 只证明局部程序导数；两者都不证明曲光线 forward 代表真实成像物理。

## 7. 6 train + 2 held-out views

8 个 view 在数据 manifest 生成时一次性固定：

```text
train:    view_00 ... view_05  -> loss, VJP, optimizer and train diagnostics
held-out: view_06 ... view_07  -> sealed until code/config/stopping rule freeze
```

实际 ID 不必连续，但必须有 6/2 唯一划分并记录 SHA-256。两个 held-out view：

- 不进入 reconstruction loss、regularizer 选择、早停、路由阈值或超参整定；
- 不用于选择 field resolution、RK4 steps、迭代次数或方法；
- 只在 field parameters、代码 hash、停止规则和预算上限冻结后运行一次最终 reprojection；
- 不允许看完两个结果后重训；如果重训，原 held-out 自动降级为 development evidence。

合成小测试可以使用 8 个微型视角验证数据隔离，但不得把它写成已完成 6+2 三维实验。

## 8. 到三维重建的最小闭环

训练数据项为

\[
L_d(\theta)=\frac12\|W_{tr}(F_{tr}(\theta)-y_{tr})\|_2^2.
\]

最小目标再加一个已冻结的三维 smoothness 正则：

\[
L(\theta)=L_d(\theta)+\lambda\|D\,\mathrm{decode}(\theta)\|_2^2.
\]

反向步不显式构造 Jacobian：

```text
values = decode_field(theta)
y_hat = F_train(theta)
r = valid_mask * inv_std * (y_hat - y_train)
g_data = VJP(theta, valid_mask * inv_std * r)
g = g_data + grad(regularizer)
propose optimizer / line-search step
run detached validity and topology audit on the trial field
accept, shrink, or fail closed
```

首个 smoke 可用有严格迭代上限的 Adam/L-BFGS；它只需 forward+VJP。第二门才使用 JVP/VJP 定义 matrix-free Gauss-Newton 正规算子：

\[
v\mapsto J^T W^TWJv+\lambda H_Rv.
\]

这是 field JVP/VJP 与三维重建的最小接口，不需要 materialize `J`，也不允许 forward 和 adjoint 分别训练。

首个重建门必须使用 `fidelity="full_curved_central_rk4"`。`straight` 只是对照；当前 `discrete_rk4_jvp_predictor` 与 OCBH 因 field detach 不得出现在优化图里。它们只能在完成第 12 节改造并单独过门后成为可选 fidelity。

## 9. 预算账本

预算单位必须同时记“物理场点查询”、“向量化 dispatch”和“导数 sweep”，不能用一个 `forward_calls` 覆盖全部成本。

```python
@dataclass
class FieldOperatorBudgetLedger:
    train_forward_calls: int
    train_jvp_calls: int
    train_vjp_calls: int
    heldout_forward_calls: int
    failed_or_rejected_forward_calls: int
    logical_scalar_grid_point_queries: int
    interpolation_dispatches: int
    rk4_stage_evaluations: int
    saved_state_bytes: int
    host_sync_count: int
    wall_time_seconds: float
    peak_rss_bytes: int
    peak_device_bytes: int
```

对某批次总光线数 `R=sum_v R_v`、步数 `S`，当前 central-difference 离散规则的逻辑下界为：

| route | scalar point queries | interpolation dispatches | derivative ledger |
|---|---:|---:|---|
| straight shared state | `7 R S` | `1` for its bundled implementation | one forward, plus requested JVP/VJP |
| full curved central RK4 | `35 R S` | `35 S` when rays are batched per view | 4 RK4 RHS + 1 midpoint-output RHS per step |
| discrete epsilon-JVP teacher | `35 R S` | `35 S` | one bend JVP; currently zero field VJP |
| OCBH | `7 R (2S+1)` | `7` batched coefficient dispatches | 4 coordinate reverse sweeps; currently zero field VJP |

每个一阶重建迭代至少记 `1 train forward + 1 train VJP`。JVP dot/FD test、Gauss-Newton/CG 内循环、line-search 重算和因失败拒绝的 trial 都单独记帐。VJP 可以复用前向保存的图，但必须报 saved-state/peak-memory，不能把“没有新发起 field query”写成“VJP 成本为零”。

预算至少分成四个 namespace：`derivative_gate`、`train_reconstruction`、`heldout_audit`、`failed_or_fallback`。等 VJP 对比先冻结训练 VJP 总数，等 wall-time 对比再冻结独立进程墙钟上限；两种预算不能混为一个胜负。

## 10. 失败关闭

以下任一情况禁止继续该步反传或将结果填成零：

1. 场、轨迹、观测、JVP 或 VJP 出现 non-finite；
2. `n <= refractive_index_floor`；
3. 任一 RK4 stage/中点离开 `[-1,1]^3` 或 central stencil 域；
4. 方向归一化误差超过预注册阈值；
5. trial 相对 base 改变了 mask、row order、support/cell crossing 或 frustum/topology 签名；
6. `view_ids` 跨越 train/held-out 边界，或 manifest/hash 不匹配；
7. dot test/FD 未过，却请求进入 reconstruction；
8. 预算超过冻结的 VJP、wall-time 或 memory 上限。

对完整曲光线路由，失败意味着缩小 optimizer/line-search 步长；重试仍失败就关闭当前 case，不回退成 straight ray 继续训练。

未来若启用 OCBH/Picard 候选路由，路由 mask 必须在每个 outer iteration 的 base state 上 detached/frozen；unsafe rays 回退到同配置 full curved forward，候选查询与 full fallback 查询都记帐。路由决策不反传，不声称它的离散切换有普通导数。

## 11. NeRIF/TDBOST 接口边界

本地 NeRIF/2026 neural-primitive 文档已表明：神经折射率场、自动/中央差分梯度、路径采样、mask 和多视角重建都有直接先例。所以本接口的作用是确保同一个离散曲光线 forward 可以被 JVP/VJP 审计并接到重建，不是声称首次神经 BOST 或首次可微 ray tracing。

本地 TDBOST 对账文档显示公开代码与正文在 rank/frame/decoder 等配置上仍有待确认项，公开结构也没有给出可直接复用的 field `Jv/J^Tq` 合同。因此：

- 6 train + 2 held-out 是本项目的预注册协议，不归因为 TDBOST 原始配置；
- 不把 TDBOST 的 time/tensor factor 直接当成 field tangent parameter；
- 接入组内实现前必须确认 input/output tensor、view/frame layout、distortion module、ray model、sampling/mask、单位、loss 和可用许可；
- 没有师兄确认的最小调用链和独立对账，不声称与 TDBOST distortion module 等价或不重叠；
- 外部 NeRIF/TDBOST 代码只做来源审计；本接口保持 clean-room，不复制无明确许可的实现。

## 12. 最小实现步骤

1. 新建 `VoxelFieldSpec/BOSTRayBatch` 和严格 preflight，冻结 8-view row order、6/2 split、support、单位与 hash。
2. 实现 `decode_field(theta_p)`，验证活跃索引、零边界、幅值界和折射率 floor。
3. 从 `field_dependent_ray.py` 抽出不做 Python scalar conversion 的 tensor-only central RK4 trace/integral kernel，保持当前离散物理不变。
4. 实现第 5 节 `n2_curved_bost_forward`，先只支持 `full_curved_central_rk4`，并与现有直接逐 view 调用做数值对账。
5. 用同一 closure 实现 field JVP/VJP，先过 tiny float64 dot test、FD 和 scalar-loss gradient 对账。
6. 新建图外 diagnostics/audit wrapper，返回 domain/stencil/floor/norm/topology 状态；与纯 forward 输出分开。
7. 实现四 namespace 预算账本，用冻结公式对账 logical queries，用实测记录 wall/memory/host sync。
8. 做一个小网格、少光线、8 微型视角的合成 reconstruction smoke，只验证 loss 下降、参数更新和 held-out 隔离，不报算法结果。
9. 只在 full curved field 接口全部过门后，改造 OCBH：移除 field/input/output detach，为所需的坐标导数保留 `create_graph=True`，再独立做 field FD/JVP/VJP。
10. OCBH 过门后才加 frozen routing/full-curved fallback；cone-ray、神经场、强基线和正式 96-cell 继续关闭。

## 13. 必须实现的测试

### A. 参数与 forward

1. `decode_field` shape/dtype/device/index-order 测试；
2. support 外和最外层恒为零，`n` 严格高于 floor；
3. `theta=0` 的零扰动场输出为数值零；
4. multi-view cat 与逐 view direct call 在 float64 容差内一致；
5. 更换 view 输入顺序时输出只做对应 block 置换；
6. tensor-only 核与当前 full curved central RK4 相同配置对账。

### B. 导数

7. 至少 3 个随机种子、2 个非零 field states 的 JVP central-FD 趋势测试；
8. 相同 contexts 的 JVP/VJP dot defect `<=1e-8`；
9. `VJP(q)` 与 `autograd.grad(sum(F*q), theta)` 一致；
10. `detach_path=True` 与 `False` 的 VJP 在非零弯曲 case 上可检测地不同，防止轨迹项被静默删除；
11. 一个明确测试确认当前 `predict_discrete_rk4_jvp_residual` 和 `predict_operator_consistent_homotopy_residual` 的返回值不能对 field 反传，防止误用；
12. 将 JVP/VJP 精度门在 CPU float64 固定，不用 MPS float32 结果放宽它。

### C. split、重建与预算

13. manifest 必须恰好 6 train + 2 held-out，无重复、无遗漏；
14. 改变 held-out observation 不能改变任一次训练 loss、gradient、optimizer state 或 stopping decision；
15. 微型重建 smoke 中 `theta.grad` 有限且存在非零活跃成分，训练 loss 在冻结小步数内可下降；
16. 每次重建迭代恰好记录 forward/VJP，JVP/line-search/rejected trial 不得漏记；
17. straight/full-curved/epsilon-JVP/OCBH 的 logical query 公式与 tiny instrumented counts 一致；
18. held-out audit 账本与 train 账本分开，不能重置后只报局部费用。

### D. fail-closed

19. non-finite field/output/JVP/VJP、折射率 floor、domain/stencil 越界均抛出类型化失败；
20. topology/cell/frustum 签名改变时 FD 和 trial step 不得报 pass；
21. train 调用传入 held-out ID 必须拒绝；
22. 预算上限触发时停止且保留完整失败账本；
23. 候选路由未过 field derivative gate 时，配置不能将其设为 reconstruction fidelity。

## 14. 进入正式实验前的关闭清单

只有下列项全部有独立产物和测试证据，才可设计正式 96-cell：

- 同一 full curved forward 的 field FD/JVP/VJP 通过；
- 6/2 view split 和数据单位冻结；
- 微型三维重建 smoke 通过；
- 预算账本在查询、sweep、墙钟和内存上闭合；
- failure injection 全部 fail closed；
- OCBH 若要进入优化，已独立去除 field stop-gradient 并通过同样的导数门；
- NeRIF/TDBOST 最小输入、输出、评价和许可边界已与师兄对账。

在此之前，本文只是实现设计，不是重建成功、算法有效或论文新意的证据。
