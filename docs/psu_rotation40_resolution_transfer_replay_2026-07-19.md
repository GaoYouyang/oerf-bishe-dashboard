# PSU rotation-40 分辨率迁移：独立 clone 完整 replay

这份命令用于**新的独立 clone**。必须 checkout attestation commit `7cc9c3c`，因为该 commit 已固定协议和 attestation、但尚不含正式结果目录；直接在当前 result commit 上运行会按设计因输出已存在而拒绝。

## 1. 准备

```bash
git clone https://github.com/GaoYouyang/oerf-bishe-dashboard.git oerf-r40-replay
cd oerf-r40-replay
git checkout 7cc9c3c

# 指向一个已安装 Python 3.11 + torch/numpy/matplotlib 的解释器。
export PYTHON=/absolute/path/to/python

# 指向本机私有 PSU 派生库根；不要把它提交或上传。
export PSU_PRIVATE_ROOT=/absolute/path/to/psu_bost_flight_body
```

私有根必须包含冻结的 geometry binding、rotation-40 payload、16³ baseline 和 32³ cached reference。runner 会逐文件复核哈希、shape、dtype 与 manifest；路径正确但内容不同也会拒绝。

## 2. 正式重放

```bash
"$PYTHON" site_tools/run_psu_rotation40_resolution_transfer.py \
  --config demo_t16_operator/configs/psu_rotation40_resolution_transfer_preregistered_v1.json \
  --attestation demo_t16_operator/configs/psu_rotation40_resolution_transfer_attestation_v1.json \
  --geometry-root "$PSU_PRIVATE_ROOT/rotation40_geometry_binding_v1" \
  --geometry-private-report "$PSU_PRIVATE_ROOT/rotation40_geometry_binding_v1/geometry_binding_private_report.json" \
  --geometry-public-summary docs/psu_rotation40_geometry_binding_public_summary.json \
  --payload-root "$PSU_PRIVATE_ROOT/rotation40_development_h2_v1" \
  --payload-private-report "$PSU_PRIVATE_ROOT/rotation40_development_h2_v1/payload_private_report.json" \
  --payload-public-summary docs/psu_rotation40_cell_payload_public_summary.json \
  --resolution-public-summary docs/psu_b0_streaming_resolution_public_summary.json \
  --volume-16 "$PSU_PRIVATE_ROOT/b0_streaming_baseline_v1/reconstruction_16cubed.npy" \
  --volume-16-private-report "$PSU_PRIVATE_ROOT/b0_streaming_baseline_v1/private_report.json" \
  --volume-16-public-summary docs/psu_b0_streaming_baseline_public_summary.json \
  --volume-32 "$PSU_PRIVATE_ROOT/b0_cached_reference_v1/reconstruction_32cubed_cached.npy" \
  --volume-32-private-report "$PSU_PRIVATE_ROOT/b0_cached_reference_v1/private_report.json" \
  --volume-32-public-summary docs/psu_b0_cached_reference_public_summary.json \
  --output-dir demo_t16_operator/results/psu_rotation40_resolution_transfer_public_v1
```

预期机器判决：

```text
SUPPORT_RESOLUTION_GAIN_DID_NOT_CLEAR_NUMERICAL_TRANSFER_GATE_NO_GO
```

预期核心数值：

```text
16 cubed concatenated-all-ray relative-L2 = 0.8432631430215097
32 cubed concatenated-all-ray relative-L2 = 0.9595912440290707
16 minus 32                              = -0.11632810100756097
```

## 3. 完整性复核

```bash
cd demo_t16_operator/results/psu_rotation40_resolution_transfer_public_v1
shasum -a 256 -c checksums.sha256
```

注意：attestation 没有预先绑定四个传递依赖和 requirements；完整缺口及本次运行后核验哈希见 [独立审计](psu_rotation40_resolution_transfer_independent_audit_2026-07-19.md)。重放一致不会把这一预注册缺口倒推成已修复。
