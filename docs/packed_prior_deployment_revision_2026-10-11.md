# 无损打包后的条件先验部署成本

## 中文

精确经典逆因子采用标准下三角打包，不修改任何float64系数、先验、核模型、物理、精化深度或门槛。原主序存储管线的失败保留；这是另行冻结的表示修订，不是新神经算法或事后改判。

固定一条完整101帧留出轨迹，三种方法、两套实现、各三轮，共18个新进程。全部终点复现原合格输出，最大归档场差2.33e-12、跨实现差1.36e-12。112项执行检查与61项终态/重复核验通过。

| 实现 | 方法 | 101帧墙时中位数 | 三轮最大峰值RSS |
| --- | --- | --- | --- |
| 正式 | 条件先验与打包解析初值 | 4.067秒 | 1.043GiB |
| 正式 | 无先验打包解析初值 | 3.640秒 | 0.786GiB |
| 正式 | CGLS128 | 16.689秒 | 0.605GiB |
| 独立 | 条件先验与打包解析初值 | 4.141秒 | 1.064GiB |
| 独立 | 无先验打包解析初值 | 3.613秒 | 0.785GiB |
| 独立 | CGLS128 | 16.744秒 | 0.612GiB |

**受限部署时间优势成立，联合资源门仍FAIL。** 先验墙时为配对CGLS128的0.244/0.247倍，约4.10/4.04倍时间优势；但峰值RSS仍为1.725/1.738倍。不能用时间通过掩盖内存失败。相对同精度无先验初值，先验另增加11.73%/14.62%时间与32.79%/35.54%RSS；两者都为5A+5A^T，主要省调用与时间优势来自经典解析求解，不是学习归因。

几何因子由约571MB无损降至285MB；先验完整数值payload仍约506MB，对照CGLS128约24MB，payload不是进程RSS。统计场收益保持，下一瓶颈是完整因子/源库依赖及相对强经典控制的学习价值。

**范围：**已知五相机、干净输入、一条预固定轨迹、8线程。计入导入、输入/产物加载、顺序单查询、日志和输出至退出；文件系统可能温热。已有几何/QR/源标签/核拟合/CFD与新增离线打包仍非免费，未计入部署计时。两种打包三角接口共享标准BLAS/LAPACK，独立核推理/CGLS和各自合格因子保持。不是十一折资源、准备/训练至部署、噪声、新几何、神经、外门或真实BOST成功。旧失败与各项科学边界不变。

## English

This separately frozen revision stores every float64 lower-Cholesky coefficient in standard packed form. It changes no prior, kernel model, physical operator, refinement depth, loading or threshold. The earlier row-major deployment FAIL remains intact; this is an exact representation revision, not a new neural algorithm or a retrospective verdict change.

One fixed complete 101-frame held trajectory, three methods, two implementations and three fresh-process rounds give 18 runs. All endpoints reproduce qualified archives: maximum archived-field and cross-implementation differences are 2.33e-12 and 1.36e-12. All 112 execution and 61 terminal/repeat checks pass.

Prior/ridge/CGLS128 median wall times are 4.067/3.640/16.689 seconds formally and 4.141/3.613/16.744 seconds independently. Worst-of-three peak RSS values are 1.043/0.786/0.605 GiB and 1.064/0.785/0.612 GiB.

**Limited prepared-deployment wall advantage is measured; the joint resource gate still FAILS.** Prior wall ratios to paired CGLS128 are 0.244/0.247, about 4.10/4.04 times faster, but RSS ratios remain 1.725/1.738. Prior premiums over equally accurate no-prior ridge are 11.73%/14.62% wall and 32.79%/35.54% RSS. Both use 5A+5A^T: classical analytic solving, not learning, accounts for the principal call/time reduction. Established statistical field gains remain.

Geometry-factor storage falls losslessly from about 571 MB to 285 MB. Total packed numerical payload remains about 506 MB for the prior versus 24 MB for CGLS128; payload is not process RSS. Reducing full-factor/source-bank dependence and proving value beyond strong classical controls remain priorities.

Scope is one known clean five-camera geometry and one pre-fixed complete trajectory, 8 threads, potentially warm filesystem. Imports, artifact/input loads, sequential queries, logs, output writing and process exit are timed. Geometry/QR/source labels/kernel/CFD preparation and new offline packing remain nonfree and outside this timer. The packed wrappers share standard BLAS/LAPACK; separate kernel/CGLS implementations and qualified numerical factors are retained. Not all-fold resources, preparation-plus-training-to-deployment, noise/new-geometry, neural, external or real-BOST success. Earlier failures and scientific boundaries remain unchanged.
