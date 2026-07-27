# PoolFire C v15.1 独立审计

## 审计结论

最终复审状态：

```text
P0 = 0
P1 = 0
P2 = 1
VALIDATED_NEGATIVE_NO_SHARED_MODE_V15_1
algorithm_breakthrough=false
```

权威运行绑定提交 `b6a63cd`。更早的 `2d64577` 预运行因红队发现四个 P1 已明确
作废，不能引用其科学判决。

## 第一轮红队发现

1. 全零正超额也会因稳定排序获得三个 top-3 位置；
2. 一致性票和 top-3 票可以来自不同轨迹，仍可能拼成通过；
3. 负科学结果被包装在 `PASS_INDEPENDENT_RECOMPUTATION_FAIL...` 状态中；
4. source roster 漏掉运行时载入的 `learning_labs/__init__.py`；
5. 历史 checkpoint 主要依赖同目录 report/READY 自证；
6. “truth-free”措辞强于实现能证明的范围；
7. 多个模式通过时没有冻结唯一选择规则；
8. 回归测试没有覆盖上述反例。

前四项定为 P1，后三项定为 P2。第一次预运行因此不具权威性。

## v15.1 修复

- 严格正超额才有资格进入 top-3；
- 同一轨迹必须同时满足 median 正、正帧比例至少 60% 和 top-3；
- 科学结果使用 `POSITIVE/NEGATIVE`，独立复算完整性单列布尔值；
- runner 与 validator 独立冻结同一 20 文件 source roster；
- release 绑定旧 report、READY、v14 gate、独立验证 seal、checkpoint 和 observation；
- 协议加入公开 observation-tail 数值锚点，运行时重新投影并逐项复现；
- 声明改为“未请求 raw pair truth；读取了旧 truth-derived report；
  filesystem-wide nonaccess 未证明”；
- 多模式选择按支持轨迹数、median 正超额份额和 mode ID 确定；
- 新增零能量、支持交集和多模式确定性选择回归测试。

复审人工反例全部转为正确负判，运行时加载模块均在绑定闭包内。

## 独立数值复算

独立 validator 没有导入正式 runner 或正式残差分析模块。它重新：

1. 加载十个 LOTO checkpoint；
2. 生成两个模型在五条轨迹上的最终 K1 场；
3. 重新投影并计算 observation residual；
4. 重写 18 模式 FFT 分解；
5. 重写逐轨迹 top-3、P45 富集和共享模式 gate。

结果：

```text
maximum array absolute difference          = 0
maximum nested summary absolute difference = 0
maximum Parseval absolute                  = 8.326672684688674e-17
formal evidence unchanged during validation = true
```

科学判决为负，独立复算本身通过。两者没有再混写成一个 PASS 状态。

## 剩余 P2

当前 `8 passed` 已覆盖 FFT/Parseval、零能量 top-3、support 交集、P45-only
反例和多模式确定性选择。尚未把每一种 public anchor、independent seal 和 source
roster 篡改组合都做成自动 mutation test。

这些边界已由当前运行时检查和人工红队确认，因而不影响本次冻结判决；它仍是未来维护
时的回归覆盖风险，不是算法或物理证据。
