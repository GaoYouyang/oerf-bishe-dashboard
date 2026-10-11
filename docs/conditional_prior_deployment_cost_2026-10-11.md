# 条件先验新进程部署成本

## 中文

五相机条件先验已有1111/1111查询、11/11完整留轨迹的四指标匹配与场收益。现在补测真实部署代价：保持算法和已训练模型不变，固定第一条完整留出轨迹101帧，条件先验、同精度无先验解析初值和CGLS128各做两实现、三轮独立新进程，共18次。

全部输出复现已合格归档，逐帧最大场相对差2.33e-12；独立实现与三轮重复一致。计时包括Python导入、输入/几何/模型加载、顺序单查询推理、日志和输出写入，未丢第一帧、未批量查询或选择异常轮次。

| 实现 | 方法 | 101帧墙时中位数 | 三轮最大峰值RSS |
| --- | --- | --- | --- |
| 正式 | 条件先验加解析初值 | 19.82秒 | 1.837GiB |
| 正式 | 无先验解析初值 | 18.99秒 | 1.560GiB |
| 正式 | CGLS128 | 17.86秒 | 0.605GiB |
| 独立 | 条件先验加解析初值 | 20.72秒 | 1.821GiB |
| 独立 | 无先验解析初值 | 20.28秒 | 1.560GiB |
| 独立 | CGLS128 | 17.55秒 | 0.604GiB |

**冻结部署资源门FAIL**：条件先验的墙时为CGLS128的1.110/1.181倍，峰值RSS为3.034/3.018倍；要求分别不超过0.95与1.0。5对A/A^T回调以外仍有完整几何逆因子、核求解和源场收缩。无先验解析初值同为5对，因此省回调不能归因于学习。

先验相对无先验初值只增加4.36%/2.20%墙时与17.79%/16.79%RSS。原有场收益不变；这次说明当前部署没有资源优势，不说明先验信息无价值，也不是充分优化的数学成本下界。

**范围**：只是一条完整轨迹、已知五相机、干净输入、8线程的预制产物部署。文件系统缓存可温热。此前几何构建、QR、标签生产、核拟合和CFD生成仍非免费，未重做、未计入这次计时。不是准备/训练至部署全流程、十一折成本、噪声、新几何、神经、外部或真实BOST成功。保留原正负结论，不改变参数或重跑取更好数字。

## English

The unchanged five-camera conditional prior already has 1111/1111 four-error matches and 11/11 complete held-trajectory passes, with field-information gains. This deployment screen fixes the first complete 101-frame held trajectory and compares the prior, an equally accurate no-prior analytic initializer, and CGLS128. Two implementations and three fresh-process rounds produce 18 runs.

Every endpoint reproduces its qualified archive; maximum per-frame field discrepancy is 2.33e-12. Startup/imports, geometry/input/model loading, sequential single-query inference, logs, and output writing are included. No first-frame discard, query batching, warmup or selective rerun.

Formal median wall times are 19.82/18.99/17.86 seconds for prior/ridge/CGLS128; independent times are 20.72/20.28/17.55 seconds. Their worst-of-three peak RSS values are 1.837/1.560/0.605 GiB and 1.821/1.560/0.604 GiB.

**The frozen limited resource gate FAILS**: prior wall ratios to CGLS128 are 1.110/1.181 and RSS ratios 3.034/3.018, exceeding the 0.95/1.0 limits. Full classical factors, dense inverse actions, kernel solves and source-bank contractions remain nonfree. Both prior and no-prior initializer use 5A+5A^T, so their classical call reduction is not a learned contribution. Prior premiums over ridge are 4.36%/2.20% wall time and 17.79%/16.79% RSS; its established field gain remains valid.

This is prepared-artifact deployment at one known clean five-camera geometry and one 101-frame trajectory, with 8 threads and potentially warm filesystem caches. Geometry/QR/label/kernel/CFD preparation is inherited, nonfree and outside the timer. It is not preparation-plus-training-to-deployment, all-fold, noise/new-geometry, neural, external or real-BOST resource evidence, nor a mathematical cost lower bound. No parameter change or rerun promotes the failed gate.
