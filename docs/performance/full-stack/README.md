# MiniCPM5 四项优化栈：20-step A/B

本页是四个独立 draft PR 的最终组合验证：

- [#8 THD RoPE metadata cache](https://github.com/wangmx17/megatron-lm-musa-patch/pull/8)
- [#9 TE padded-metadata no-sync](https://github.com/wangmx17/megatron-lm-musa-patch/pull/9)
- [#10 Flex MoE route conversion fusion](https://github.com/wangmx17/megatron-lm-musa-patch/pull/10)
- [#11 TE expert cross-parameter batched Muon NS](https://github.com/wangmx17/megatron-lm-musa-patch/pull/11)

A为现有原始优化栈，四项均不加载/关闭；B加载四个PR的精确候选文件并开启三个显式开关。runner在每个case前按SHA256核对staging，在launch前记录active source hash，退出后恢复初始源码和helper。详细commit/file hash见[source_provenance.json](source_provenance.json)。

## 20-step性能结果

真实indexed数据、随机初始化；8张MTT S5000，BF16、THD/span、Muon，TP2 PP1 CP4 EP8 DP1，GBS16、MBS1、seq65536。两组各20step，均值排除首步，使用step2–20算术平均；不是profiler运行计时。

| 完整非profiler运行 | A | B |
|---|---:|---:|
| step2–20平均s/iter | 48.983911 | 48.015321 |
| 全rank全程最大allocated，MiB | 71782.847 | 71814.535 |
| 全rank全程最大reserved，MiB | 74916 | 74850 |
| 完成步数 / launcher rc | 20/20 / 0 | 20/20 / 0 |
| FINAL_MEMORY rank数 | 8 | 8 |

B减少**0.968589s/iter，改善1.977362%**。最大逐step loss相对差**0.003922910%**，grad norm相对差**0.401228907%**；均低于本轮预设0.1%/1%门禁。grad norm不是全模型逐元素梯度等价性证明。

两组均无skip/NaN、无OOM，完整退出并通过逐case源码/helper恢复检查。B allocated峰值比A高31.688MiB，reserved低66MiB；allocator峰值不是设备总显存采样。长20-step峰值不能与单项10-step峰值直接比较。

## 单项结果与组合边界

| 单项，非profiler step2–10 | A s/iter | B s/iter | 改善 |
|---|---:|---:|---:|
| RoPE metadata cache | 48.969889 | 48.743956 | 0.461372% |
| TE padded metadata no-sync | 48.975744 | 48.876289 | 0.203071% |
| Flex route conversion fusion | 48.933211 | 48.841844 | 0.186717% |
| TE expert batched Muon NS | 48.955411 | 48.510700 | 0.908400% |

各单项及完整栈是不同时间的独立短A/B，不能将百分比简单相加、相减或据此证明协同/冲突。前三项单次变化都很小；Muon机制和端到端变化更清楚，但仍需更长重复实验才能判断统计稳定性。历史200-step只有B组，不替代这里的完整A/B。

## 完整栈 A/B trace

A/B另各独立运行6step，均6/6、rc=0、无skip/NaN；rank0第4step采集stack trace。Trace运行的iter不参与上述性能结论。最大逐step loss相对差0.000571471%，grad norm相对差0.006068452%。

| rank0单采样step | A | B |
|---|---:|---:|
| device首末活动跨度，s | 49.949323 | 48.137200 |
| device busy并集，s | 46.206607 | 46.409398 |
| >100µs空泡数量 | 8573 | 3717 |
| >100µs空泡合计，s | 1.488615 | 0.713679 |
| FA kernel次数 / 累计，s | 3584 / 19.162775 | 3584 / 19.127337 |

四项均有实际执行证据：RoPE的896次cache查询仅16次设备读取；TE后`aten::equal` 2704→912；路由融合forward/backward kernel各432次，`nonzero` 1312→16；Muon由1089次单参数NS变为136次batched NS加9次fallback，`mm` 16335→135并出现2040次`bmm`。

同时，`musaStreamSynchronize`调用数6375→2407，但host inclusive等待21.695458→30.731225s，说明等待向后续消费点迁移；inclusive时间互相嵌套，不能相加或直接当成iter收益。B最大单空泡40.716ms，另有11.142ms Muon临时stack/concat相关空泡，仍是后续线索。FA kernel本身不在四个PR范围内。

[A审查trace](A.review.trace.json.gz) · [B审查trace](B.review.trace.json.gz) · [机器可读trace分析](trace_analysis.json) · [完整运行证据](evidence.json) · [恢复检查](restoration.json) · [运行环境与计时口径](RUNTIME_CONTEXT.md) · [源码来源](source_provenance.json)

两份审查trace是明确裁剪后的Perfetto JSON，不是原始完整trace：保留所有非Python事件、优化相关Python frame、原时间戳和correlation/stream，移除张量内容、无关Python frame及内部绝对路径。原件和附件SHA256见`trace_analysis.json`。

四个PR保持draft，不自动合并；最终是否保留由审查者决定。
