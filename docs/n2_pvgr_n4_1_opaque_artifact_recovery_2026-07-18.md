# N2-PVGR N4.1 不透明 checkpoint 后处理恢复协议

日期：2026-07-18

## 1. 失败位置

N4.1 正式执行完成了全部 32 格的数值层级和逐格 gate 判定。随后在生成三联图的第二个 panel 时，冻结 base `_plot` 调用了：

`axes[1].bar(counts, counts.values(), ...)`

当前 Matplotlib 将 `counts` 这个字典视为单个不可哈希 category，抛出：

`TypeError: unhashable type: 'dict'`

因此正式 output 目录仍不存在。错误发生在所有数值 checkpoint 完成之后，不涉及射线积分、指标、阈值或 machine decision。

## 2. 封存时已知与未知

已知：

- console 已输出全部 32 条 H1024/final PASS/FAIL 聚合状态；
- work directory 含 105 个 level checkpoint；
- 其中 H256/H512/H1024 各 32 个，H2048 共 9 个；
- 正式 output 不存在；
- 日志 SHA-256 为 `4b617023d6df446a7ac1c133cf89d1403f0585194ed46778d8d5fd87f1eb7560`。

在本恢复协议提交前，恢复作者不解析 checkpoint JSON 内的输出数组、相对误差、contraction、topology 或 cost 数值。因为 console 聚合状态已经可见，本恢复称为“opaque checkpoint recovery”，而不声称结果状态仍完全盲。

## 3. 冻结恢复动作

恢复程序只能：

1. 验证恢复 attestation、checkpoint 数量、H2048 数量与 Merkle root；
2. 以原 N4.1 amendment config SHA-256 验证并加载全部 checkpoint；
3. 让原 N4.1 runner 重新执行纯 Python gate 汇总与 artifact staging，不运行任何缺失 level；
4. 临时替换 plot 函数中柱状图的 x 输入为 `list(counts.keys())`，y 输入为 `list(counts.values())`；
5. 保持散点、计数、成本曲线、颜色、阈值线和所有数据来源不变；
6. 在 result、manifest、summary 中写入 recovery protocol commit、attestation SHA 和 checkpoint Merkle root；
7. 原子发布正式 output 后运行原 N4.1 validator 与 recovery validator。

禁止：

- 重跑任何 H256/H512/H1024/H2048 数值层；
- 修改或删除 checkpoint；
- 修改 16 对样本、阈值、指标、gate、逐格结论或 machine decision；
- 排除 final FAIL 格；
- 把 evaluator audit 写成算法、三维重建、真实数据或泛化成功。

## 4. attestation

attestation builder 在不解析 JSON payload 的前提下，对排序后的 `relative_path + file_sha256` 计算 checkpoint Merkle root，并记录：

- formal output 在创建时不存在；
- 105 个 checkpoint 与 9 个 H2048；
- recovery config 和所有恢复代码已在 protocol commit 中；
- `checkpoint_payloads_parsed` 为 false。

恢复 runner 每次启动都重新计算集合 Merkle root。数量、路径或任一字节变化都 fail closed。
