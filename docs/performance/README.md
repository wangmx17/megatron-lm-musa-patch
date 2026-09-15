# THD RoPE CPU metadata cache

## 改动

Packed THD RoPE 会从 `cu_seqlens` 计算每条 sequence 的长度。Transformer 各层
处理同一批 token，Q/K 数值会变化，但 token 排列和 sequence 边界不变，因此原
路径会在每层重复执行：

```python
(cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
```

当 `cu_seqlens` 位于 MUSA 时，`.tolist()` 会产生 device-to-host 读取和 CPU
等待。该优化在第一次访问时读取 CPU lengths，后续层直接复用同一个 Python list：

```python
full_lengths = thd_seqlens_cpu(cu_seqlens)
local_lengths = [length // cp_size for length in full_lengths]
```

缓存只替换 sequence lengths 的重复读取。后续仍使用 Megatron 原来的
`torch.split`、`_get_thd_freqs_on_this_cp_rank()`、逐 sequence `torch.rope` 和
`torch.cat`，不改变 packed THD、attention mask、RoPE 数值算法或 CP 通信。

## 缓存安全边界

缓存以 tensor identity、weakref 和 `_version` 共同校验：

- 同一个 tensor 且 version 不变时复用 CPU list；
- tensor 被原地或通过别名修改时重新读取；
- tensor 释放时通过 weakref 删除条目；
- 最多保留 64 项，超过容量后删除最久未使用的条目；
- inference tensor 无法安全获取 version 时不缓存；
- 调用方不得修改返回的 Python list。

CPU cache 不能修复非法 THD/CP 数据。非法 sequence 边界、长度或 CP 布局在原
Megatron 路径和 cached 路径中都可能报错；它的风险较低，是因为除 lengths 读取外
基本保留了原有计算路径。

## 单项验证

测试条件为 8 x MTT S5000、真实 indexed 数据、随机初始化、BF16、THD/span、
TP2 PP1 CP4 EP8 DP1、GBS16、MBS1、seq65536。A/B 各完成 10/10 step，排除首步：

| 指标 | 原路径 | CPU metadata cache |
|---|---:|---:|
| step 2-10 平均 | 48.969889 s/iter | 48.743956 s/iter |
| 全 rank 最大 allocated | 69927.471 MiB | 69923.084 MiB |
| 全 rank 最大 reserved | 74078 MiB | 74116 MiB |

观察到 iteration time 下降 0.461372%。这是一次顺序短 A/B 的小幅变化，未证明
统计显著。最大逐 step loss 相对差为 0.000686324%，gradient norm 相对差为
0.0200232%；gradient norm 对照不等于全模型逐元素梯度验证。两组均无 skip、
NaN、OOM 或 hang，launcher rc=0。

4 项 CPU 单测覆盖缓存命中、原地和别名修改失效、weakref 生命周期、有界容量及
inference tensor 回退。已有采样中每个 step 包含 16 个 microbatch：896 次各层
查询只有 16 次真实读取，其余 880 次命中缓存。CPU 同步通常发生在每个 microbatch
首次访问 `cu_seqlens` 时，不是整个训练只读取一次。

本 PR 不包含原始或裁剪后的 trace、机器可读 trace 分析、运行 manifest 或其他
性能附件，只保留本 README 中的必要验证摘要。最终是否采用由审查者决定。
