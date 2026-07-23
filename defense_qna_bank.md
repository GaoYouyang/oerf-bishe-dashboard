# 开题答辩问答库

用途：准备学院开题、组会汇报或和蔡老师/何远哲讨论时可能被问到的问题。回答原则是：不夸大、不泛 AI、每个回答都回到 BOST 物理问题、可复现代码和误差分析。

## 1. 题目与创新性

### Q1：你的题目创新点在哪里？

建议回答：

本课题不以提出一个全新大模型为创新点，而是围绕 BOST 这一具体实验流体诊断问题，系统量化神经隐式/坐标场方法在少视角、含噪和有限表示容量条件下的适用边界。已有预研显示，coordinate regularizer 在 3/5 视角 sparse-view 情况下更稳，但在 7/9 视角干净数据中传统 baseline 更强。因此本文的创新点是建立可复现的误差图谱、数据接口和鲁棒性分析流程，而不是简单声称神经方法一定优于传统方法。

### Q2：为什么不是直接复现 NeRIF？

建议回答：

完整 NeRIF 涉及真实九视角系统、位移估计、复杂光线模型、网络结构、训练策略和真实实验验证。本科阶段直接完整复现风险较高。我会先拆成合成数据、传统 baseline、coordinate-field / simplified neural field 和参数扫描四个可控模块；如果组内能提供真实数据，再向 NeRIF 真实接口迁移。

### Q3：为什么题目不叫“基于深度学习的流场重构”？

建议回答：

那样太泛。我的问题不是一般图像识别，而是 BOST 光学观测到折射率/密度场的反问题。神经网络在这里是连续场表示和可微优化工具，必须和光学 forward model、重投影验证、物理量单位和误差来源绑定。

## 2. 物理与方法

### Q4：BOS/BOST 的基本物理链条是什么？

建议回答：

反应流温度和组分变化导致密度变化，密度通过 Gladstone-Dale 关系影响折射率。折射率梯度使经过流场的光线发生偏折，背景图案在相机图像中出现表观位移。BOS 测量这种位移，BOST 进一步利用多视角位移反演三维折射率场。

### Q5：你如何验证重构结果？

建议回答：

合成数据中有 ground truth，可直接计算 relative L2、correlation、SSIM proxy 和 PSNR。更重要的是做 re-projection validation：用重构场重新生成多视角 deflection，与输入观测位移比较。真实数据中如果没有 ground truth，重投影误差、leave-one-view-out 验证和物理一致性会更重要。

### Q6：传统 baseline 选什么？

建议回答：

早期阶段用简化层析/逐层 stack reconstruction 和正则化方法作为 baseline，重点是公平比较视角数、噪声和容量的影响。后续如果何远哲师兄指定组内 baseline 或 UBOST/体素法作为标准，就以组内约定为准。

## 3. 已有工作

### Q7：你现在已经做了什么？

建议回答：

已经完成五个本地可运行 demo。M0 是 2D BOST / coordinate-field toy；M1 是 3D-stack sparse-view BOST toy；M2 做了 noise-view-capacity robustness scan；M3A 是 PIV-BOST velocity compensation toy；M3B 是 4D BOST low-rank temporal toy。这些 demo 不等于最终论文结果，但证明了我已经能把问题拆成数据、forward model、baseline、指标和图表。

### Q8：M2 的主要结论是什么？

建议回答：

M2 显示 coordinate regularizer 不是永远更好。在当前 synthetic proxy 中，3/5 视角下 coordinate regularizer 全部降低 relative L2；7/9 视角下传统 stack baseline 全部更好。容量扫描也显示 Fourier feature 太少会过度平滑，容量提高后才可能超过 noisy baseline。这支撑了“适用边界”而不是“神经方法绝对优越”的叙事。

### Q9：M3A 和 M3B 会不会让主线太散？

建议回答：

不会，因为它们是备选升级线，不是同时承诺完成的主线。主线仍是 BOST/NeRIF 鲁棒性分析。M3A 用于如果师兄希望贴 PIV-BOST，M3B 用于如果师兄希望贴 4D BOST。最终会根据数据和组内需求选择 A/B/C 中一条。

## 4. 数据与可行性

### Q10：如果课题组不给真实数据怎么办？

建议回答：

保底路线仍能成立：用 synthetic phantom 和公开 BOS/TBOS 数据完成完整 pipeline，包括 forward model、baseline、coordinate-field 重构、鲁棒性扫描、可视化和报告。真实数据作为升级项。如果真实数据不能公开，则论文图可以主要使用 synthetic/开源数据，真实数据只用于内部验证或趋势确认。

### Q11：你需要课题组提供什么数据？

建议回答：

最低需要一组九视角 BOST 样例：flow-off 背景图、flow-on 扰动图、mask、视角/标定参数、位移场或位移估计结果，以及一份参考重构切片或指标。还需要确认哪些图能用于本科论文公开展示。如果做 PIV-BOST，还需要 PIV 图像对、速度场、时间间隔和同步方式。

### Q12：算力是否足够？

建议回答：

当前 M0-M3B 都是轻量 numpy/matplotlib 版本，能在本地运行。后续如果做 PyTorch neural field，初期也可以控制分辨率和采样数，先完成小规模实验。只有完整 NeRIF 或 4D BOST 高分辨率训练才需要更强 GPU，因此不会把毕业设计成败完全压在算力上。

## 5. PIV-BOST 与 4D BOST

### Q13：PIV-BOST 部分你准备怎么做？

建议回答：

如果选择 PIV-BOST 路线，第一阶段不直接做复杂 stereo-PIV，而是明确补偿层级：图像层、速度向量层还是误差传播层。当前 M3A 已经在向量场层面展示 BOST-style compensation 可降低速度 RMSE。后续可以升级到粒子图像 toy 或真实 PIV-BOST loader。

### Q14：为什么不完整复现 4D BOST？

建议回答：

完整 4D BOST 涉及时空张量分解、畸变校正、ray tracing、混合精度和大规模真实数据，超出本科毕设的可控范围。我会只拆一个子问题：低秩时序先验是否能减少逐帧重构抖动。当前 M3B 已经显示 rank 3 能降低 mean L2 和 temporal smoothness，但也明确低秩不能修正系统性几何偏差。

## 6. 论文与成果

### Q15：最终论文能写成什么结构？

建议回答：

第 1 章写 OERF 背景和 BOST 需求；第 2 章写 BOST 物理模型和反问题；第 3 章写 neural/coordinate field 方法；第 4 章写 M0-M2 合成数据和鲁棒性分析；第 5 章根据师兄数据选择真实 BOST、PIV-BOST 或 4D low-rank toy；第 6 章总结和展望。

### Q16：如果实验结果没有明显提升怎么办？

建议回答：

这仍然是有价值的结果。我的研究问题不是保证神经方法永远更好，而是找出何时更好、何时失败。如果结果显示传统 baseline 在某些视角数或噪声条件下更强，就可以写成适用边界和方法选择建议。这对课题组也有价值，因为真实实验中知道什么时候不该用复杂模型同样重要。

### Q17：你的毕业设计最终交付物是什么？

建议回答：

至少包括：可复现实验代码、合成数据 pipeline、baseline 和 coordinate-field/NeRIF-style 方法、鲁棒性扫描图、论文图表、数据 manifest 模板和开题/论文文档。若数据条件允许，还会交付真实 BOST/PIV-BOST/4D 子问题接口或初步结果。

## 7. 和导师/师兄沟通时的安全边界

### Q18：哪些话不要说？

不要说：

- 我想做 AI。
- 我都可以，看师兄安排。
- 我想完整复现 4D BOST。
- 我现在还没想好，先学学。

更好的说法：

- 我已经有 NeRIF/BOST 保底主线，也准备了 PIV-BOST 和 4D BOST 两条备选升级线，希望根据组内数据和需求收束到一条。
- 我不承诺完整 4D BOST 复现，但可以做低秩时序先验、参数扫描和可视化子模块。
- 我希望真实数据接口从第一周就按组内格式设计，即使暂时先用 synthetic 保底。

## 8. 最后一分钟总结

如果答辩老师只给一分钟，可以这样说：

本课题面向背景纹影层析的三维折射率场重构问题，目标是建立可复现的 BOST/NeRIF 重构与误差分析流程。已有预研完成了 2D toy、3D sparse-view toy、鲁棒性扫描、PIV-BOST 补偿 toy 和 4D low-rank toy。下一步将根据课题组数据条件，优先选择真实 BOST 数据迁移、PIV-BOST 补偿或 4D 低秩时序子问题中的一条作为毕业设计主线。本文的重点不是泛化深度学习，而是光学观测、物理 forward model、神经/坐标场表示和误差验证的结合。

