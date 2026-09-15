# TE expert parameters batched Muon Newton-Schulz

## 改动

TEGroupedMLP 的各 expert 权重可以表现为彼此独立的二维 Parameter。已有的
`MUON_BATCH_NS=1` 主要处理单个 Parameter 内可还原的 expert 轴，不能自动把多个
独立 Parameter 合成一个 batch，因此这些权重仍会逐参数执行 Newton-Schulz（NS）。

`musa_patch/muon_expert_batch.py` 新增 `MuonExpertBatch`。它在同一 expert Muon
group 内按 shape、dtype 和 device 分组，用临时 `[batch, rows, cols]` tensor 调用
现有 batched NS，再把更新分别应用回原 Parameter。每个矩阵独立归一化，不合并
持久 Parameter 或 checkpoint 布局；单元素尾组回退到原更新。

该适配默认关闭，需要同时设置：

```bash
export MUON_BATCH_NS=1
export MUON_TE_EXPERT_BATCH_NS=1
```

本次验证使用的最大 batch 为8。

## 支持与安全边界

- 只处理 expert-parallel Muon group 中完整、未分片的二维NS输入；QKV排除；
- distributed 元数据需满足 `dist_world_size=1`、`tp_split_dim=-1`、完整shape及
  `local_range == global_range`；
- 不兼容的 shape、分片参数和单元素尾组回退到原单参数路径；
- 依赖当前兼容版 Muon 的输入准备、NS计算、更新应用和分布式重建接口，不声称支持
  任意上游 Megatron Muon 版本；
- 临时 `torch.stack` 会增加短期显存占用，需要用真实训练观察全rank峰值。

## 正确性与性能摘要

CPU/MUSA 三次完整 optimizer update 对照覆盖混合 expert shape、Muon与Adam参数；
更新后的参数最大差为0，状态一致，资格guard检查通过。真实训练实际命中
`(2048,512)`、`(1024,2048)` 两类矩阵的batch8和尾batch4。

真实 indexed 数据、随机初始化、8 x MTT S5000、BF16、THD/span、TP2 PP1 CP4
EP8 DP1、GBS16、MBS1、seq65536 的独立10-step A/B结果为：

| 指标 | 原Muon路径 | 跨Parameter batched NS |
|---|---:|---:|
| step 2-10 平均 | 48.955411 s/iter | 48.510700 s/iter |
| 全 rank 最大 allocated | 69908.889 MiB | 69915.309 MiB |
| 全 rank 最大 reserved | 74182 MiB | 74076 MiB |

两组均完成10/10 step，launcher rc=0，无skip或NaN，8个rank峰值记录齐全。
观察到iteration time下降0.908400%；短A/B未证明统计显著或长期稳定收益。最大逐step
loss相对差为0.001335392%，gradient norm相对差为0.011734105%；gradient norm
对照不等于全模型逐元素梯度证明。

运行时机制统计显示单参数NS由1089次降至9次，batched NS由0增至136次，说明跨
Parameter批处理确实命中。临时stack/concat仍可能形成局部空泡，本项也不处理
attention或其它同步路径。

本 PR 不包含原始或裁剪后的 trace、evidence JSON、运行 manifest 或其它性能附件，
只保留本 README 中的必要验证摘要。最终是否采用由审查者决定。
