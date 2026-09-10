# TE padded-metadata no-sync：独立验证

本改动仅去掉 MUSA 上用于判断 padded metadata 是否相同的两类 `torch.equal` 读取，保守选择 TE 已有的 padded-compatible 路径。它不改变 FA kernel、THD/span 语义或 CP P2P 调度，也不声称去掉所有 FlashAttention equal。

## 实现和启用

`MUSA_TE_PADDED_METADATA_NO_SYNC=1` 显式启用，默认关闭。`musa_patch/te_padded_metadata.py` 在进程内对三个已知 TE 入口进行精确 AST 匹配，保留外层 None 判断、非 MUSA equal 和函数装饰器，并同步一个 MUSA adapter alias。源码布局、closure 或 alias 不兼容时抛错，不做部分安装。

不改写镜像安装的 TE 文件，不替换全局 `torch.equal`。四项 CPU 合约单测通过；目标运行环境的入口绑定与幂等安装检查通过。这是针对实测 TE 接口的兼容实现，不承诺任意 TE 版本均适用。

## 测试口径

- 8 张 MTT S5000，BF16，TP2 PP1 CP4 EP8 DP1；GBS16、MBS1、seq65536、THD/span、Muon。
- 真实已分词数据，随机初始化；不是 Megatron checkpoint 恢复。
- A/B 各 10 step，仅此项不同。均值为非 profiler 运行 step 2–10 的算术平均。
- A/B trace 分别独立运行 6 step，rank0 第 4 step 单步采集，保留 stack；不以 profiler 运行的 iter 时间计算收益。
- 训练梯度检查是逐步 grad norm 对照，不是全模型逐元素梯度等价性证明。

| 非 profiler 完整运行 | A | B |
|---|---:|---:|
| 平均 s/iter，step 2–10 | 48.975744 | 48.876289 |
| 各 rank 全程最大 allocated，MiB | 69929.703 | 69912.320 |
| 各 rank 全程最大 reserved，MiB | 74058 | 74058 |
| 完成步数 | 10/10 | 10/10 |
| launcher rc | 0 | 0 |

均值下降 **0.203071%**，属于很小的单次对照变化，不能证明稳定收益或统计显著性。最大逐步 loss 相对差 **0.000918083%**，grad norm 相对差 **0.0190056%**。成功的 A/B 均无 skip/NaN，各 8 rank 的全程 allocator 峰值齐全，测试改动已恢复。

## 首次失败必须保留

首次 B 在进入模型训练/FA 前，启动时间的 float64 `all_reduce(MIN)` 后 `item()` 等待，600 秒 watchdog 触发，完成 0 step。该次不计入性能均值，无完整显存峰值或可信 launcher 退出码；延迟退出后核对 12 个源码哈希并清理临时 helper、残留 rank。

用户报告可能同时启动了另一组训练，但未独立核实时间和设备占用交集，根因仍未确定。相同训练配置重试成功不抹除首次失败；额外 float32 最小诊断未对齐完整训练环境，不能替代训练验收，也没有将训练改为 float32。

## Trace 机制证据

| rank0 单采样 step | A | B |
|---|---:|---:|
| aten::equal 次数 | 2704 | 912 |
| equal host inclusive，s | 17.284722 | 17.118547 |
| musaStreamSynchronize 次数 | 6375 | 4583 |
| StreamSynchronize host 累计，s | 21.702431 | 21.637889 |
| musaMemcpyAsync 次数 | 7734 | 5942 |
| device busy 区间并集，s | 46.209452 | 46.192214 |
| device 首末活动跨度，s | 49.883282 | 49.824992 |
| 大于 100 µs 空泡合计，s | 1.400168 | 1.473405 |

equal、MemcpyAsync、StreamSynchronize 各减少 1792 次，证明目标 device 读取被消除。但等待累计时长没有同幅下降，device busy 几乎不变，空泡也未一致改善。host 父子 inclusive 时长不可相加，更不能用同步次数下降推导 iter 收益。

B 剩余 equal：FA 后端 `_has_varlen_mismatched_seqlens` backward 448 次、17.056032 秒；forward 448 次、0.062448 秒；另有 16 次、67.582 µs。它们不在本 PR 的 TE padded-metadata 比较范围内，不能声称“所有 FA equal 只剩 16 次”。

device-to-host 分析通过 correlation 找 runtime，再以同 pid/tid 的时间包含关系查 Python 栈。B 最大空泡 22.975 ms 的下一 kernel 在空泡开始之后才 host launch，提示局部 host 提交受限；相关栈不足以确定具体操作，不能强行归因。本分析仅 rank0 单步。

[A 审查 trace](A.review.trace.json.gz) · [B 审查 trace](B.review.trace.json.gz) · [逐步数值、全 rank 峰值、SHA256 与裁剪声明](evidence.json)

附件是明确裁剪后的 Perfetto JSON，不是完整原件：保留所有非 Python 事件、相关 Python frames、原始时间戳和 correlation/stream；移除无关 Python frames 及绝对安装路径等元数据。完整原件哈希、大小、事件计数在 evidence.json；原件另外留存。缺少的 frame 不表示没有调用。

trace 数值最大 loss 相对差 0.000656697%，grad norm 相对差 0.0110140%；两组各 6/6、rc=0、8 rank 峰值齐全且恢复通过。所有显存数值是完整运行 allocator 峰值，不是设备总显存采样值。

## 交付状态

完整四项优化栈的 20-step A/B 尚待完成，本 PR 保持 draft，不自动合并。历史 200-step B-only 结果不替代此次完整栈对照。
