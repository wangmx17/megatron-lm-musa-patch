# CP 反向 batched P2P：原理与验证

> 已完成首轮A/B、反序复测、最终模块10-step及最终文件的逐元素梯度检查。默认关闭，未验证前后向两项同时开启或长期训练。

## 修改解决什么问题

原 MUSA TE 的 CP ring 用 `flash_attn_p2p_communicate_sync` 先完成通信依赖，再提交当前 FA。不同 stream 本身不代表重叠：当前计算不需要下一轮收到的内容时，仍被过早等待挡住。

本项只修改 backward：将两处 KV/dKV 交换改为成组异步 P2P；在当前 FA 和 dq 操作提交后、累加收到的 dKV 前执行 request wait。没有删除消费者依赖。forward 不变。

MCCL 的 wait 在当前已核对实现中通过完成 event 建立当前流依赖，默认不是 CPU 全局同步。通信提交时仍要等待发送输入就绪。依赖的正确位置才是关键，不能简单删除 wait。

本项同时改变同步提交方式和通信成组方式，因此不是“只测 batch 开关”或“只测延后 wait”的因果拆分。

### 为什么这个等待位置有依赖依据

已核对原TE源码：2892行原本就分配了两个独立通信buffer，每个包含KV和dKV；2947行起轮流使用它们。本项没有新增这两个buffer。

1. 当前轮通信读取buffer[i%2]，写另一个buffer[(i+1)%2]。
2. 当前FA读取当前KV，写独立的临时dq_/dkv_（例如3066–3067行的empty_like），并不写通信正在接收的dKV区域。
3. 原3494行的wait位置恢复为有效代码：FA/dq先提交，再等待通信，最后才将本轮梯度累加到接收的dKV（3503行起）。下一轮才消费该KV。
4. MCCL提交时仍对当前计算流建立发送输入就绪依赖，因此复用另一个buffer前，之前读取该buffer的计算也受到流顺序保护。

这些是该实现的依赖依据，不是对所有MUSA版本无hang的证明；仍需结合短跑、梯度检查和失败路径记录判断。

## 范围与入口

- `musa_patch/cp_backward_batch_overlap.py`：精确源码保护、单方向函数变换、运行时配置保护。
- `musa_patch/__init__.py`：安装入口，默认关闭。
- 开启：`MUSA_CP_BACKWARD_BATCH_OVERLAP=1 NVTE_BATCH_MHA_P2P_COMM=1`。
- 目标分支已包含独立的 forward 方向适配；前后向同时开启尚未验证，因此安装器显式拒绝
  `MUSA_CP_FORWARD_BATCH_OVERLAP=1` 与本开关同时启用。
- 仅进程内替换 Python 静态函数，不写入安装的 TE 文件；新进程取消开关即可恢复。
- 已验证范围：MUSA、BF16、THD、CP4、Q shape=(16384,16,128)，零 dropout、padding_causal、无 attention bias、非 fused-attention/非 FP8。
- TE attention.py SHA256 必须为 `510afa9a3da138697c8c16538efabcb22b8ec5dacaadd6aef5bdd02ff9b510c1`。未知源码显式报错，不静默应用补丁。
- 非法配置会显式报错；这不是通用 CUDA/FP8/任意 CP size 的实现。

## 测量口径

worker31012 / megatron_test，单机8卡S5000；TP2/PP1/CP4/EP8/DP1，seq65536、GBS16、MBS1、BF16。固定原1000-step参数及PR8/10/11；不叠加PR9、PR13或另一方向的候选修改。

真实本地 indexed 数据；随机初始化，tokenizer路径不是加载Megatron checkpoint。seed42、warmup5、原Muon参数不变。每组从头运行10步，用step2–10算平均iter，排除第一步初始化开销。独立6步trace在第4步采样，不将profiling耗时混入性能均值，也不拿6步与10步不同数据索引做loss逐步对照。

远端默认分支后来合入#12。PR代码可以针对该分支审查，但这里的性能数据来自上述固定历史栈，不能写成新默认分支全部优化组合后的已验证收益。

## 首轮完整结果

### 反序复测及最终代码复验

| 组别 | step2–10平均iter | 对反序A的提升 | 完整步数 / rc |
|---|---:|---:|---:|
| 反序A | 48.219922 s | — | 10/10 / 0 |
| 反序B | 46.948833 s | 2.636024% | 10/10 / 0 |
| 最终保护模块B | 46.951222 s | 2.631070% | 10/10 / 0 |

最终保护模块相对反序A，逐step最大loss相对差0.000911%、grad norm相对差0.015204%；skip/NaN均为0，8rank FINAL_MEMORY齐全，最大allocated=69951.566MiB、最大reserved=74532MiB。最终文件64项逐元素复核rc=0、全有限，最大relative L2=3.925840429e-5，最大absolute difference=0.0009765625。

下面保留初次A/B，避免只挑选最好的单轮结果：

| 指标 | A 固定基线 | B 本项 |
|---|---:|---:|
| 完整步数 / rc | 10/10 / 0 | 10/10 / 0 |
| step2–10平均iter | 48.174533 s | 46.927922 s |
| iter减少比例 | — | 2.5877% |
| 最大rank allocated峰值 | 69943.141 MiB | 69933.842 MiB |
| 最大rank reserved峰值 | 74594 MiB | 74430 MiB |
| skip / NaN | 0 / 0 | 0 / 0 |

峰值来自8个rank退出时的allocator全程峰值，不是外部5秒采样。不出现OOM不代表没有增加内存，故仍报告绝对值。

独立相同Q/K/V及上游梯度诊断覆盖8rank、2种span布局（单65536和两个32768）、out/dQ/dK/dV共64项，rc=0，全有限；最大relative L2=4.020479537e-5，最大absolute difference=0.0009765625。grad norm相近不等价于逐元素梯度相同，以上单独测试补足该检查。短跑通过不证明长期训练完全等价。

## Trace证据及解释

| 全步rank0指标 | A | B |
|---|---:|---:|
| 本方向P2P kernel数 | 3584 | 1792 |
| 本方向P2P累计时间 | 1.470206 s | 0.964412 s |
| 本方向FA × P2P区间交集 | 0 s | 0.063099 s |
| 另一方向FA × P2P区间交集 | 0 s | 0 s |

归属按CP Python调用栈→runtime correlation→device kernel确认，不凭stream号猜测。重叠用时间区间并集交集计算，避免同类kernel重复计数。

反向只新增约0.063s交集，而iter下降约1.247s，不能把全部收益归因于overlap。成组P2P减少kernel数量及通信本身开销，也是观察到的变化。

## 失败路径也必须保留

`NVTE_BATCH_MHA_P2P_COMM=0`的普通异步版本不予推荐：前向未完成首步，出现DeepEP recv timeout；反向8/10后出现segfault/MUSA错误。不能用其部分step计算收益。错误发生位置不等价于已经证明底层根因。

成组版本短跑成功不能推导所有长跑都稳定；保持默认关闭，建议在实际部署配置继续长时间验证。

## 原始证据位置

实验根目录：`/mbzz_ssd/overlap_pr8_10_11_20260916`

- A：`results/A_base_perf10/results.json`，`results/A_base_trace/`
- B：`results/B_cp_backward_batch_perf10/results.json`，`results/B_cp_backward_batch_trace/`
- 逐元素检查：`gradient_suite_v2.json`
- 反序复测：`repeat_suite.json`（三组均完整退出0）
- 最终模块复验：`guarded_suite.json`（两组均10/10、退出0）
- 最终文件逐元素复核：`gradient_suite_final.json`（两方向各64项、退出0）

Trace采自最初的单方向候选，最终模块生成的计算函数SHA256与该候选逐字一致；新增的host配置保护另做完整10-step验证，不把历史trace冒称为最终包装模块的新trace。

测试结束后，原TE及原训练源码哈希检查通过，8卡显存为0且无GPU进程。测试源码和入口已移到实验根目录的`evidence/archived_sources/`，可恢复；日志与trace保留。该PR没有自动开启优化或覆盖用户正式训练源码。
