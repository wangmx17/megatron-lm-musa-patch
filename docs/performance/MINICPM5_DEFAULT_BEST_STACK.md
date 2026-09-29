# MiniCPM5 单机默认最佳性能栈

`examples/minicpm5/run_16a3b.sh` 默认启用已在单机 8 卡 MTT S5000 上保留的
MiniCPM5 16A3B 性能栈。直接使用默认脚本时不再需要额外的 `stack_state.env`。

## 默认配置

验证边界为 BF16、THD/span、TP2/PP1/CP4/EP8/DP1、MBS1、GBS16、
seq65536、Muon、真实 indexed 数据。默认开启：

- Transformer Engine CrossEntropy、bias-SwiGLU、gradient accumulation fusion；
- routed-expert grouped GEMM、MoE permute fusion、Router fusion、RoPE fusion；
- manual GC（100 step）、DeepEP compact permute、deferred expert counts；
- MCCL 16 channels、16 MiB buffer、DeepEP 56 SM；
- THD RoPE metadata cache（当前 patch 为常开实现）；
- Flex route conversion fusion；
- Muon TE expert batched Newton-Schulz，最大 batch 8；
- CP forward/backward batched P2P 和 consumer-side waits；
- MATE 0.2.7 routed-expert GroupedLinear fprop/dgrad；
- 关闭 activation recompute。

对应的新增默认值仍可在启动前通过同名环境变量覆盖。CP overlap 只支持 patch
内运行时 guard 所列的 MUSA、BF16、THD、CP4、固定 Q/K/V shape 和指定 TE
源码；改变序列长度、CP size、attention backend 或 TE 版本时，应显式关闭
`MUSA_CP_FORWARD_BATCH_OVERLAP` 和 `MUSA_CP_BACKWARD_BATCH_OVERLAP`，而不是绕过
guard。

本次保留栈使用的 Transformer Engine 不提供 native THD LSE fp32 capability，
因此 launcher 默认使用 `MUSA_TE_THD_LSE_FP32=disable` 的兼容路径。只有安装了
匹配 TE 并确认其 capability 后，才应切换为 `require`。

## 性能证据

2026-09-28 在 worker31009 的隔离源码副本上逐级累计测试。每级运行 6 step，
统计 step 2--6；所有正式通过阶段均为 rc=0、skip=0、NaN=0：

| 配置 | step 2--6 平均 |
|---|---:|
| 无性能优化，full/block rc14 | 83.0542 s/iter |
| 完整默认收益栈，关闭重计算 | 46.6785 s/iter |

短测 iteration time 减少 36.3757 秒，观察提升 43.797%。最终值与历史
PR #16 + #17 合并 20-step 的 46.7537 s/iter 基本一致。

这是同一固定硬件、数据和拓扑上的短测结果，不代表任意 Megatron、TE、MATE、
DeepEP、MCCL 版本或任意拓扑都能获得同等收益。小于约 1% 的单项收益仍需
40/100-step A/B 才能证明稳定性。

## 明确保留关闭的候选

以下项目没有进入默认最佳栈：DeepEP ACE、shared-expert overlap、MATE FA3、
RMSNorm 替换、FA auxiliary fusion、THD CP correction fusion、MoE token drop，
以及尚未完成相同口径端到端验证的其它 open PR。它们仍可单独实验，但不能和
本默认栈的 46.6785 s/iter 结果混为一谈。
