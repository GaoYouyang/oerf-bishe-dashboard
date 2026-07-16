# 实验配置模板

这里放毕业设计代码仓库可直接借用的配置模板。当前模板服务 BOST/NeRIF 主线，用来统一 dataset、geometry、method、metrics 和 output。

## 文件

- `bost_nerif_experiment_template.json`：单次 BOST/NeRIF 实验配置模板。
- `experiment_registry_template.csv`：实验登记表模板，用于记录每次实验的核心图、支持的论点、失败边界和下一步。

## 使用方式

复制到你的代码仓库：

```bash
cp experiment_templates/bost_nerif_experiment_template.json /path/to/bost-nerif-bishe/configs/synthetic_m1_3d.json
```

每次实验至少改：

- `experiment.id`
- `dataset.source`
- `geometry.num_views`
- `forward.noise_std`
- `methods`
- `output.root`

## 关键原则

- 配置文件描述实验，不要把关键参数只写在脚本里。
- 实验输出目录必须保留一份 resolved config。
- 真实 OERF 数据不要写绝对私有路径到公开报告里。
