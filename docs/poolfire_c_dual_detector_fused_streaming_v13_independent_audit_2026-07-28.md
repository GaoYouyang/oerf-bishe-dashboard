# v13 融合流式正式预检：独立只读审计

日期：2026-07-28
结论：`FAIL_SYNTHETIC_PREFIT_FUSED_STREAMING_V13`

## 审计判决

- P0：0
- P1：3
- P2：3
- 未发现数据伪造、算术错误、调用账错误或足以推翻本轮结果的实现缺陷。

## 独立重算

```text
v13 calls       = 202A + 202A^T
reference calls = 404A + 404A^T

field relative-L2 p90 / worst = 7.668e-8 / 8.429e-8
steady-state measured wall reduction = 42.68%
v13 RSS reduction vs v12.5 = 22.41 MB

higher-p90 v13 RSS       = 60,571,648 bytes
higher-p90 reference RSS = 57,622,528 bytes
ratio                     = 1.051179983
frozen cap                = 1.05
```

15 个 worker 的逐帧 receipt 全部正确，三条 arm 的五次字段输出各自确定性一致。

## P1

1. 按冻结 validator，RSS 使用 `method="higher"`，比值确实超过 1.05，正式 FAIL
   不可争议。
2. 协议正文没有显式写 p90 插值法。5 次测量时 higher-p90 就是最大值；事后计算的
   linear-p90 比值约 `1.048859`。这不能用于改判，只说明 RSS 统计稳健性有限，
   不能宣称 v13 已被证明内存必然更差。
3. 42.68% 是预热后的 proposal + solver + 输出写入区间，不含启动、模型加载和
   native context 创建，不能称完整冷启动部署加速。

## P2

1. 字段误差可从保存数组独立重算；proposal parity 由 validator 再调用同一生产
   wrapper，不是第二套独立实现。
2. validator 没有逐项强制核对 worker report 的全部模型、几何和环境身份字段；
   本轮外部复核一致，不影响当前裁决。
3. `VALIDATED_READY` 直接绑定汇总验证文件，没有直接绑定全部 worker artifact；
   本轮重新核对 15 份 report 和字段文件后未发现变化。

这些 P2 是未来证据封印改进点，不是继续扩建基础设施的理由，也不改变本轮科学结论。

## 允许公开的最小结论

在固定 101 帧合成代理预检中，v13 与 v12.5 保持数值一致，把完整算子调用从每帧
`4A+4A^T` 降至 `2A+2A^T`；稳态 measured wall 下降约 42.7%，相对旧 v12.5
的峰值内存减少约 22.4 MB。但保守 higher-p90 whole-process RSS 比值为 1.05118，
略高于预注册 1.05 门，因此正式结果为 FAIL，不授权进入 fit 数据实验。

## 禁止声明

- 算法突破、论文成功或 SOTA；
- 真实 BOST、真实 PoolFire 重建或物理同精度；
- 跨工况泛化或完整冷启动部署加速；
- 已证明该路线没有内存潜力或数学上不可行；
- 用其他通过项覆盖 RSS 失败；
- 事后更换 p90 算法把本轮改判为 PASS。
