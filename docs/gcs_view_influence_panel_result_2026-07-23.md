# Exact Leave-One-View 逐视角影响面板：有信息，但没有超过旧观测的增量价值

## 0. 一页结论

正式状态：`VIEW_INFLUENCE_POSTOPEN_INFORMATION_NO_GO_CLOSE_BRANCH`。

本轮用 exact leave-one-view（LOO）回答一个很窄的问题：将六个 BOST 视角中的一个移除，并从同一 seed 重新训练 base 与 residual 后，三维修正方向的幅值、余弦、有符号投影和准入稳定性，能否比上一轮已有的 observable energy/control 特征更好地排序“真改善”与“中性/有害”修正。

结果是一个严格 NO-GO：

- 144 条 full path 和 768 条 exact LOO path 全部完成，总计 912 次 base+residual 拟合；
- 144 条 full replay 与冻结源观测的最大差为 `0`；
- 74/144 条网络路径可完整计算 LOO 特征，合并网络重复后只有 27/72 个条件单元完全可观测；不可观测单元全部 fail-closed，没有从分母中删除；
- source observable control 的 grouped out-of-fold ROC-AUC 是 `0.962637`；
- view influence only 的 AUC 是 `0.729670`；
- source + view influence 的 AUC 是 `0.909890`；
- 组合特征相对 source control 的增量为 `-0.052747`，而结果前冻结的通过门是 `>= +0.05`。与门的总差距为 `-0.102747`。

独立 validator 没有调用 runner 的 rank-AUC 函数，而是显式枚举 7 个正例与 65 个非正例的 455 个样本对。source control 正确排序 438 对，combined 只正确排序 414 对；独立复算与主报告完全一致，状态为 `VALID_NO_GO_DECISIVE_INCREMENTAL_GATE_FAILURE`。

**突破监测：没有突破。** LOO 特征确实含有一些 post-open synthetic 排序信息，但它没有提供超过已有 observable control 的增量价值。因此不授权把 exact LOO 近似成 JVP/VJP influence，不训练 set encoder，不打开 Stage B，也不申请新 synthetic calibration 数据。

## 1. 实验在回答什么

上一轮 energy gate 证明，总场 L2/H1/Huber 和一个留出投影不足以稳定判断三维修正方向。逐视角影响的工作假设是：

> 如果一个修正只被少数相机支持，移除关键相机后的重训修正应当出现更大幅度变化、方向翻转或准入不稳定；这些部署可见信息可能揭示投影近零空间中的脆弱修正。

实验故意使用 exact LOO 而不是遮住一个 residual block。每次移除视角后，base 与 residual 都用与 full path 相同的 seed 从头拟合。它是信息上界与成本上界，不是可部署算法。

## 2. 结果前冻结的分母与对照

| 层级 | 数量 | 角色 |
|---|---:|---|
| morphology family | 2 | smooth plume / wrinkled density interface；仍是同 family post-open |
| phantom group | 12 | 独立分组单位 |
| noise | 2 | 2% / 8% |
| rig | 3 | nominal 6-view / sparse 4-view / coarse-ray 6-view |
| condition unit | 72 | `phantom x noise x rig` |
| network repeat | 2 | nuisance repeat，不当独立物理样本 |
| full fit | 144 | 对冻结源观测做 exact replay |
| leave-one-view fit | 768 | 4-view rig 每条 4 次，6-view rig 每条 6 次 |
| total fit path | 912 | 全部完成 |

特征在真值 join 之前落盘。固定三路对照使用完全相同的 leave-one-phantom-out folds、ridge 强度与闲值：

1. `source_observable_control`：上一轮能量、holdout 和粗糙度特征；
2. `view_influence_only`：14 个 LOO 幅度/方向/准入特征；
3. `source_plus_view_influence`：前两者的固定并集。

主门不是“combined AUC 是否好看”，而是“combined 是否比 source control 至少多 `0.05`”。这一设计防止把重复信息包装成新算法。

## 3. 三路信息模型的冻结结果

| feature set | grouped OOF AUC | smooth AUC | wrinkled AUC | cluster bootstrap 95% | eligible |
|---|---:|---:|---:|---:|---:|
| source observable | 0.962637 | 0.959596 | 0.976562 | [0.891414, 1.000000] | 27/72 |
| view influence only | 0.729670 | 0.666667 | 0.789062 | [0.631579, 0.816901] | 27/72 |
| source + view | 0.909890 | 0.919192 | 0.875000 | [0.820111, 1.000000] | 27/72 |

四个信息门的结果：

| 门 | 冻结要求 | 结果 |
|---|---:|---|
| combined grouped AUC | >= 0.75 | PASS |
| combined 每 family AUC | >= 0.65 | PASS |
| combined phantom-bootstrap lower | >= 0.50 | PASS |
| combined - source AUC | >= +0.05 | **FAIL: -0.052747** |

因为四门是 AND 合同，最后一门失败已充分否定信息增量。不允许事后删特征、换 lambda、换 threshold 或重新定义主门。

## 4. coverage 不能被 precision 遮住

| feature set | p=0.5 | p=0.8 | p=0.9 |
|---|---|---|---|
| source observable | 1 TP / 0 FP / 1.4% coverage | 1 / 0 / 1.4% | 0 / 0 / 0% |
| view influence only | 0 / 2 / 2.8% | 0 / 0 / 0% | 0 / 0 / 0% |
| source + view | 1 / 1 / 2.8% | 1 / 0 / 1.4% | 1 / 0 / 1.4% |

combined 在 `p>=0.8` 时看起来是 100% precision，但它只接受 72 个条件中的 1 个，仍漏掉 6/7 个真改善。这与上一轮 energy ridge 的问题一样：近乎全拒绝可以得到表面 precision，却不能证明安全或有用。

## 5. 独立复算与 CSV 恢复事故

全量拟合完成后，旧 CSV writer 因异构行表头中止：按准入协议，不可观测的 fail-closed path 不含 LOO feature 列，可观测 path 含这些列；`csv.DictWriter` 只使用第一行字段作表头，在后续观测行遇到新列后报错。

恢复没有重训：

- 报错前 `path_records.jsonl` 已有 144 条唯一 full record 和 768 条嵌套 LOO record；
- 恢复器将 `_fit_path` 替换为“任何调用立即失败”，只允许原 runner 从既有记录汇总；
- 恢复前后 fit-record SHA-256 都是 `1236c64b53959f412b0709552ced3aa9e5344c38db4050e802e5ab42db228e31`；
- `new_fit_record_count=0`，原 runner SHA-256、protocol SHA-256 和 source commit 全部绑定；
- 原始总 wall-clock 因在 summary 前报错而不可恢复，不伪造时间；912 条 record 内部的 base/residual training time 求和分别为 876.45 s 与 6085.84 s。

独立 validator 进一步检查了结果目录校验和、三路各72条分母、不可观测概率强制置零、observable CSV 无真值列、独立 pairwise AUC、Stage B 密封和所有 claim authorization 为 false。16 项检查全部通过。

## 6. 能学到什么，不能推断什么

可以说：

1. exact LOO 不是完全无信息；view-only AUC 0.7297 表明它对这批已开 synthetic 条件有部分排序能力。
2. 这些信息在冻结的低容量模型中没有超过 source observable control，与 source 合并反而降低 held-out 排序。
3. 45/72 个条件因两个网络重复不能都生成完整 LOO 特征而 fail-closed，说明这套特征在当前 reconstruction policy 下 coverage 太低。

不能说：

- LOO 必然在所有 BOST 或真实相机中无用；当前只是同 family post-open synthetic 面板；
- AUC 下降证明了某一个唯一物理原因；14 个新特征、7 个正例和 12 个独立 group 不足以做稳定归因；
- source control 是可部署的安全 gate；它的 AUC 很高，但 `p>=0.8` 也只接受 1/72；
- 已优于 NeRIF、DeepONet、FNO 或任何现有三维重建方法。本轮根本没有与它们做公平性能对比。

## 7. 对算子学习路线的直接判决

V0--V5 路线在 V1 停止：

| 层 | 原计划 | 现在的判决 |
|---|---|---|
| V0 | source observable / energy control | 保留为一个很强但低 coverage 的开发对照，不是成功 gate |
| V1 | exact LOO information upper bound | **NO-GO，支线关闭** |
| V2 | JVP/VJP influence approximation | 不授权；没有理由近似一个无增量价值的上界 |
| V3 | grouped ridge on view statistics | 不另行调参；冻结 ridge 已给出 NO-GO |
| V4 | permutation-invariant set encoder | 不授权；不用大模型容量追小样本事后结果 |
| V5 | operator correction + exact fallback | 不授权 |

这不是放弃算子学习，而是把算子的输出从“一个在 synthetic 上堆特征的 gate”转回真实任务。下一个有效门不是继续挖这 12 个 phantom，而是获得师兄的真实 callable、geometry 和物理 endpoint。

## 8. 下一步只做三件事

1. **拿到真实数据合同。** 确认 straight/curved-ray forward 的输入输出、逐 camera residual、JVP/VJP、几何标定、单位、mask、切分层级和可用的 truth/proxy。
2. **选一个真实物理终点。** 优先级为 known-target/phantom field，其次是 PIV-BOST 补偿后 velocity error，再其次是不依赖同一投影链的质量/动量/能量或时序输运 residual。
3. **先用经典算法定义强基线。** 在真实数据上固定 CGLS/TV/Huber/NeRIF 等计算预算、场/梯度/重投影/尾部指标，再决定算子应学 warm start、preconditioner、bounded correction 还是 uncertainty/abstention。

当前最重要的师兄联系稿仍是 [N5/D5 首次接口确认](n5_d5_advisor_first_contact_2026-07-19.md)。在这些回答到位前，继续在已开 synthetic 面板上发明 gate 不会使毕设更接近真实成果。

## 9. 复现与主产物

```bash
cd /path/to/oerf-bishe-dashboard

PYTHONPATH=. .venv/bin/pytest -q \
  learning_labs/test_finalize_gcs_view_influence_panel.py \
  learning_labs/test_run_gcs_view_influence_panel.py \
  learning_labs/test_analyze_gcs_view_influence_panel.py \
  learning_labs/test_validate_gcs_view_influence_panel_result.py

shasum -a 256 -c \
  learning_labs/results/gcs_view_influence_panel_v0_run1/checksums.sha256
shasum -a 256 -c \
  learning_labs/results/gcs_view_influence_analysis_v0/checksums.sha256
shasum -a 256 -c \
  learning_labs/results/gcs_view_influence_independent_validation_v0/checksums.sha256
```

主产物：

- `learning_labs/results/gcs_view_influence_panel_v0_run1/summary.json`
- `learning_labs/results/gcs_view_influence_panel_v0_run1/path_records.jsonl`
- `learning_labs/results/gcs_view_influence_panel_v0_run1/recovery_provenance.json`
- `learning_labs/results/gcs_view_influence_panel_v0_run1/information_predictions.csv`
- `learning_labs/results/gcs_view_influence_analysis_v0/view_influence_information_audit.png`
- `learning_labs/results/gcs_view_influence_independent_validation_v0/validation.json`

## 10. 证据等级与禁止主张

当前证据等级仅为 `E1_POSTOPEN_SYNTHETIC_INFORMATION_UPPER_BOUND_ONLY`。

只授权三句话：

1. 在12个已开 phantom group 上，exact LOO 特征本身具有部分 grouped ranking 信息。
2. 将它加到冻结 source observable control 后，grouped OOF AUC 降低 0.052747，未达到预写增量门，该支线按协议关闭。
3. 912 条拟合记录的分母、哈希、恢复过程和 pairwise AUC 已被独立验证。

不允许声称：新算法成功、安全 gate、算子学习成功、优于 NeRIF/DeepONet/FNO、真实 BOST/PIV-BOST 验证、跨 rig 或跨形态泛化、论文成功或研究突破。
