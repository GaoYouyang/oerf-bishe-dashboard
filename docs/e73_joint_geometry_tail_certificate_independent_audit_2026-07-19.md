# E73-B JGTCE 核心独立代码复审

> **复审日期：2026-07-19**
> **范围：只审查证书核心、攻击回归测试与声明边界；未运行真实数据、未打开新 flow 分数、未编辑仓库文件。**
> **结论：本轮指定的 3 项核验全部 `CLOSED`，`0 PARTIAL`，`0 OPEN`。**

## 1. 为什么要做第二轮复审

第一轮复审已经关闭了错误 strict-improvement 参照点、直接构造负尺度、action/metric 顺序漂移、deployment unit 重用等问题，但留下两个 P1：

1. NumPy 数组即使执行过 `setflags(write=False)`，如果底层内存仍由数组自己所有，调用方可以重新打开 `WRITEABLE`；
2. `scale_artifact_sha256` 能记录来源文件，却还没有把实际 scale 数值绑定到证书。

这两项都可以把一个本应回退的候选伪装成“严格改善”，因此不能当成一般工程细节。

## 2. 复审结果

| 核验项 | 状态 | 现在的机制 | 攻击结果 |
|---|---|---|---|
| 核心数组不能恢复写权限 | `CLOSED` | `_immutable_array()` 把 canonical `<f8` 字节放入不可变 `bytes`，再建立 NumPy view | `predicted_harm`、`scales`、`lower`、`upper`、`_calibration_scores` 重开 `WRITEABLE` 全部抵达 `ValueError` |
| scale 实际数值与合同绑定 | `CLOSED` | 构造时按 dtype、shape、C-order bytes 计算 `scale_values_sha256`，使用时再算一次 | 替换 scale 但保留旧 hash，构造和上界计算都 fail closed |
| prediction 绑定后篡改 | `CLOSED` | `upper_bounds()` 在使用前重算 prediction value hash | 绑定为 `+0.04` 后强制替换为 `-1.0`，selector 在 strict gate 前拒绝 |

## 3. 攻击脚本的可见结果

```text
IMMUTABLE predicted_harm: ValueError
IMMUTABLE scales: ValueError
IMMUTABLE lower: ValueError
IMMUTABLE upper: ValueError
IMMUTABLE _calibration_scores: ValueError
SCALE_CONSTRUCT_FAIL_CLOSED: scale value hash does not match scales
SCALE_USE_FAIL_CLOSED: scale value hash no longer matches scales
PREDICTION_TAMPER_FAIL_CLOSED: prediction value hash no longer matches table
```

实现对应 `demo_t16_operator/joint_geometry_tail_certificate.py`，回归反例在 `demo_t16_operator/test_joint_geometry_tail_certificate.py`。终审时核心测试为 `23 passed`；连同机器协议与聚焦页合同共为 `35 passed`。

## 4. 没有被这次复审授权的事

- 这不是 predictor 性能证据，当前没有 predictor；
- 这不是真实 BOST 安全证据，当前没有真实独立 calibration units；
- 这不是 conditional、OOD、anytime 或多未来单位保证；
- hash 是完整性绑定，不是 Python 进程内的密码学信任根。如果恶意代码能同时用 `object.__setattr__` 替换数值和对应 hash，它仍能伪造普通 Python 对象；
- 因此正式 runner 必须从只读、SHA-256 绑定的 artifact 自行加载 predictor 和 scale，自行计算 feature/prediction，不接受外部同时递交的“值 + 自称正确的 hash”。

## 5. 下一个真正的门

核心对象现在可以进入 formal runner 和 Phase-0 schema pilot，但仍须按顺序做：

1. 固定 artifact loader、feature function、数值 hash 和单位 manifest；
2. 先运行 16 个 analytic phantom 检查接口、成本与强制 fallback；
3. 再决定是否扩到 `24 fit + 20 calibration + 20 sealed audit`；
4. 任何 analytic 结果都不升级为真实反应流场或算法优越性主张。
