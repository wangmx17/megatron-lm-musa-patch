# TE 独立 expert 参数：跨参数 batched Muon NS

## 为什么需要这一项

TEGroupedMLP 的各 expert 权重可以是彼此独立的二维 Parameter。已有“单个 Parameter 内还原 expert 轴”的 batched NS 路径无法自动把这些独立参数合成一个 batch，因而会继续逐参数执行 Newton–Schulz（NS）。本项解决的是这个参数组织差异，不要求修改持久参数存储布局。

## 具体实现

新增 `musa_patch/muon_expert_batch.py` 中的 `MuonExpertBatch`，由 optimizer 工厂在 `MUON_TE_EXPERT_BATCH_NS=1` 时显式选用，默认不启用。仍需现有 `MUON_BATCH_NS=1`；本次 max batch 为 8。

继承兼容版 Megatron Muon，复用其 momentum/Nesterov 输入准备、单参数 fallback、权重更新与缩放 helper；step 中保留对应版的分布式重建和 Adam 参数更新流程。并非只调用父类 step 后再改结果，也不能假定任意上游 Muon API 均兼容。

对同一 optimizer group 内符合条件的独立 expert NS 输入，按矩阵 shape、dtype、device 分组，每最多 8 个用 `torch.stack` 建立临时 `[batch, rows, cols]`，调用既有真正 batched NS 实现，再逐参数应用更新。每个矩阵分别归一化，不使用跨 expert 的联合范数，不合并持久 Parameter 或 checkpoint 权重。单元素尾 batch 退回原实现。

## 安全边界

- 仅 expert-parallel Muon group、二维 NS 输入；需拆分的 QKV 参数排除。
- distributed 模式仅接受 `dist_world_size=1`、`tp_split_dim=-1`、完整 shape、`local_range == global_range` 的完整矩阵。
- 不满足条件走原单参数更新；已有 DP1 low-memory 分支不由本项扩展。
- 当前适配依赖兼容版的 `_prepare_muon_input`、`_compute_muon_update`、`_apply_muon_update` 等接口；缺少接口时明确报错，max batch 必须为正。
- 临时 stack 会增加峰值显存，因此必须看完整训练的全 rank 峰值，不能只依赖小矩阵单测。

本轮基线 `megatron/core/optimizer/muon.py` 的 SHA256 为 `d4429129cf3837cbc68c43198f64019d95c5fc4b8885a4765b8fef3139e1b8ac`。这是当前优化栈的兼容实现，不是仅凭仓库主 README 中通用 Megatron 安装步骤就能保证获得的接口。本文不声称已验证任意上游 commit；使用其它 Muon 实现前必须检查上述接口及 NS/更新语义，不能只忽略 import 错误强行启用。

归档的历史200-step B实现与本PR的 `_can_batch_te_expert_param`、`_apply_te_expert_batches`、`step` 三个方法经Python AST逐一比较完全一致。本PR将这些已运行逻辑从直接修改Megatron `muon.py` 改为默认关闭的patch adapter；这证明代码来源对齐，不把历史B-only运行冒充为当前A/B证据。

## 验证范围

已完成 CPU/MUSA 三次完整 optimizer update 对照，覆盖混合 expert shape 与 Adam 参数；参数最大差为 0，状态一致，资格 guard 检查通过。该结果只证明被测算子/更新样例，不代替真实训练 A/B 或全模型逐元素梯度比较。

## 10-step A/B 性能

真实 indexed 数据、随机初始化；8 张 MTT S5000，BF16、THD/span、Muon，TP2 PP1 CP4 EP8 DP1、GBS16、MBS1、seq65536。仅打开本项，原有 `MUON_BATCH_NS=1` 与 max batch 8 在 A/B 两组均保持一致；未叠加其它三个新优化。

| 完整非 profiler 运行 | A | B |
|---|---:|---:|
| step 2–10 平均 s/iter | 48.955411 | 48.510700 |
| 全 rank 全程最大 allocated，MiB | 69908.889 | 69915.309 |
| 全 rank 全程最大 reserved，MiB | 74182 | 74076 |
| 完成步数 / launcher rc | 10/10 / 0 | 10/10 / 0 |

均值下降 **0.908400%**。最大逐步 loss 相对差 **0.001335392%**，grad norm 相对差 **0.011734105%**；无 skip/NaN，8 rank 全程 allocator 峰值齐全，源码/helper 恢复且无残留 rank。短 A/B 尚不证明统计显著性，最终保留仍由审查者决定。

B 训练日志实际打印两类 expert 矩阵 `(2048,512)`、`(1024,2048)` 的 batch=8 及尾 batch=4，证明不是仅设置开关而未命中。

## Trace 机制证据

独立A/B trace各6步，rank0第4步单步stack采集；均6/6、rc0、8rank峰值和恢复通过。trace最大loss相对差0.000744491%，grad norm相对差0.006068452%；profiler iter不计性能收益。

| 选定Muon frame内 | A | B |
|---|---:|---:|
| 顶层step次数 / inclusive，s | 2 / 1.585536 | 2 / 0.494428 |
| 单参数NS次数 | 1089 | 9 |
| batched NS次数 | 0 | 136 |
| runtime correlation数量 | 59946 | 15780 |
| aten::mm次数 / host，s | 16335 / 0.496770 | 135 / 0.004163 |
| aten::bmm次数 / host，s | 0 | 2040 / 0.078774 |
| aten::norm次数 | 1089 | 9 |
| aten::stack次数 / host，s | 0 | 136 / 0.006150 |

单参数16335次mm=1089×15；B中2040次bmm=136×15，fallback 135次mm=9×15，直接证明真正跨Parameter batched NS生效。device跨度49.997583→48.872686s，busy并集46.202596→45.974518s，>100µs空泡合计1.561267→1.452551s；profile与独立性能运行不同，不能将约1.125s跨度差直接当成iter收益。

B仍有一个11.494ms空泡，其下一KernelConcatNy在空泡后才host launch，栈位于`_apply_te_expert_batches`；推断临时stack/concat仍有优化空间。全trace同步次数未变，等待累计也未改善，本项不解决FA等其它同步。

[A审查trace](A.review.trace.json.gz) · [B审查trace](B.review.trace.json.gz) · [逐步数值、全rank峰值、运行时batch标记、哈希与裁剪声明](evidence.json)

附件是裁剪后的Perfetto JSON，不是完整原件：保留全部非Python事件、相关Python frames、原时间戳和correlation/stream，移除无关Python frames和绝对路径等args。完整原件哈希、大小与事件数随证据提供，原件另存；缺失frame不表示没有调用。公开附件已做敏感信息扫描。

完整四项20-step A/B仍在运行，保持draft，不自动合并；最终是否保留由审查者决定。
