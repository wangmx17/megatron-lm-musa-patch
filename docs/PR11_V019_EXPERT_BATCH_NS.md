# v0.19：TE expert 跨参数 batched Newton–Schulz

## 改动目的

TEGroupedMLP 将各 expert 的权重保存为独立 Parameter。原版 v0.19 Muon 对每个二维参数分别执行五轮 NS，产生大量小矩阵乘法与 host 提交。本补丁只把本 rank、同 shape、同参数组的独立 expert 矩阵临时 stack 成三维 batch；不改变持久参数、checkpoint 布局或 expert 划分。

默认不生效。启用方式：

```bash
export MUON_TE_EXPERT_BATCH_NS=1
export MUON_TE_EXPERT_BATCH_SIZE=8
```

入口在 musa_patch/__init__.py；实现为 musa_patch/muon_expert_batch_v019.py。
只拦截 TensorParallelMuon 的精确类型，要求 expert group、expert_tp、二维 FP32 主参数、稠密梯度、blockwise TP。QKV、GTP remat、不符合条件的参数走原 orthogonalize，其他子类保留原 step。chunk size 小于 2 时整个 step 回退原实现。

归一化按每个矩阵分别执行，沿用本环境 Emerging Optimizers 的 NS 系数表、顺序、scale_mode、extra_scale_factor、动量、Nesterov 和 weight decay。BF16 baddbmm 与逐矩阵 addmm **不是逐 bit 等价**，本补丁不宣称数值完全相同。

## v0.19 的特殊兼容点

Megatron 的 _kwargs_from_config 通过 inspect.signature(TensorParallelMuon.__init__) 决定传入哪些配置。包装构造函数必须使用 functools.wraps 保留签名；否则日志虽然显示 blockwise 等配置，实际构造器却可能收到默认值。已增加 test_muon_v019_factory_signature.py 回归检查签名、参数映射及重复安装。

首次 B10 因这一问题作废，以下数据不包含该轮。不是把无效实验当作收益证据。

## 单机完整 A/B 与基线复测

环境：worker31012 / wmx_megatron_test，MTT S5000 8 卡，Megatron 0.19.1，本目录适配后的 Emerging Optimizers。
TP2/PP1/CP4/EP8/DP1，seq65536，GBS16，真实 indexed 数据，seed1234，warmup1000。
基线叠加 PR8+PR10。A/B 均 full block recompute2，为 v0.19 FP32 CE 工作区留显存；不是关闭重计算的双机配置。

| 试验 | 步数 / rc | steps6–10 平均 iter |
|---|---|---|
| A，PR11 关闭 | 10/10 / 0 | 48.37398 s |
| B，签名修复后 PR11 开启 | 10/10 / 0 | 48.17488 s |
| A，B 后关闭 PR11 复测 | 10/10 / 0 | 48.34776 s |

B 比前后 A 分别缩短 0.4116% / 0.3576%。这是小幅短跑收益，不是统计显著性或长训加速保证。计时排除启动、step4 profiler/export；只使用 steps6–10。

首轮有效 A/B：最大 loss 绝对差 0.00032，grad norm 差 0.009。
复测 A/B：最大 loss 绝对差 0.00027，grad norm 差 0.016。
两组比较 LR 都完全相同，所有步骤 skip/NaN 为零；没有观测到 OOM/hang，各 rank 正常结束并释放 GPU。不以此宣称长期收敛通过。
首轮全程 allocator 峰值（8 ranks 最大值）：allocated A 77876019200 bytes、B 77871011328 bytes；reserved 两组均 78982938624 bytes。不是五秒设备采样值。

## Trace 证据与口径

两份 trace 均采集 step4、rank0，with_stack=1。分析按 Muon Python step 包含的 host/runtime 事件，再通过 correlation 关联 device kernel。

| 指标 | A | B |
|---|---:|---:|
| Muon host step inclusive | 1.024558 s | 0.395865 s |
| 逐参数 newton_schulz_tp 次数 | 1131 | 51 |
| batched_ns Python 调用 | 0 | 136 |
| bmm / baddbmm 调用 | 0 / 0 | 680 / 1360 |
| 关联 kernel 数量 | 31542 | 10044 |
| 关联 kernel 时长之和 | 0.572273 s | 0.382537 s |

host/device annotation 同名标记共 272 个，不等于 272 次 batch；独立 batch 为 136。嵌套 host 时长不能相加；kernel 总时长不是关键路径 wall time，也不证明通信 overlap。

服务器证据根目录：/mbzz_ssd/modelbest_v0.19/experiments/evidence_20260917。
A trace：output_log/pr11_single/a10_20260917_recompute2/trace/iteration_4/rank0.1789609811662.pt.trace.json。
B trace：output_log/pr11_single/b10_20260917_signature_fixed/trace/iteration_4/rank0.1789613706281.pt.trace.json。
上述 trace 路径相对于 /mbzz_ssd/modelbest_v0.19；未上传完整 trace。
分析脚本 analyze_pr11_trace.py、完整数值 pr11_fixed_ab10_full_comparison.json、复测 pr11_repeat_a_vs_fixed_b.json 均在证据根目录。

测试 test_muon_expert_batch_v019_probe.py 对相同输入做十次更新，覆盖 Nesterov 开/关、零梯度、缺失梯度，要求动量逐值相同、参数差小于 1e-5 且有限。实测通过，最大参数差约 5.53e-6；这是小型更新探针，不代替训练验证。

本 PR 仅包含 PR11 v0.19 实现与测试文档，不包含 PR8/PR10，也不修改训练启动器或默认开启优化。测试栈中的 PR8/PR10 为性能对照前提。
