# THD RoPE metadata cache：单项验证

## 改动与边界

在四个 THD RoPE 长度读取入口复用 CPU sequence lengths，CP 的长度除法在 CPU 列表上执行。不改变 packed THD、attention mask、RoPE 数值算法或 CP 通信。

缓存验证 tensor identity、weakref 和 `_version`；原地或别名修改会失效。inference tensor 没有 version counter 时不缓存。最多 64 项，并在 tensor 释放时清理。调用方不得修改返回列表。

这比历史 200-step 使用的 identity-only 缓存增加了版本检查，因此下面使用的是本 PR 实现重新运行的证据，而非直接引用历史性能。

## 测试条件及计时

8 × MTT S5000，TP2 PP1 CP4 EP8 DP1，GBS16、MBS1、seq65536、BF16、THD/span、Muon。使用真实 indexed/tokenized 数据，随机初始化，不是 Megatron checkpoint 恢复。A 为原始优化栈，B 仅增加本 PR。

性能各运行 10 step，排除首步，取 step 2–10 算术平均；独立 trace 各运行 6 step，rank0 采集一个 step，with_stack=1、record_shapes=0、profile_memory=0。Profiler 运行的 iter 不纳入性能均值。

| 完整性能运行指标 | A | B |
|---|---:|---:|
| 平均 s/iter，step 2–10 | 48.969889 | 48.743956 |
| 全程各 rank 最大 allocated，MiB | 69927.471 | 69923.084 |
| 全程各 rank 最大 reserved，MiB | 74078 | 74116 |
| 完成 step | 10/10 | 10/10 |
| launcher rc | 0 | 0 |
| 全程峰值报告 rank 数 | 8 | 8 |

iter 均值下降 **0.461372%**。这是单次 A/B 的小幅均值差，**未证明超出运行波动**。逐步最大 loss 相对差 0.000686324%，最大 grad norm 相对差 0.0200232%；无 skip/NaN/OOM，无 hang。正常退出、无残留测试进程和源码恢复检查通过。allocator 全程峰值不同于驱动总显存占用。

训练 grad norm 对照不是全模型逐元素梯度校验。`python test/test_thd_metadata_cache.py` 的四项 CPU 测试验证命中、原地/别名修改、生命周期/LRU 和 inference fallback，均通过。

## Trace 证据

- [A：审查 trace](A.review.trace.json.gz)
- [B：审查 trace](B.review.trace.json.gz)
- [逐步数值、全 rank 峰值及文件哈希](evidence.json)

附件为**明确裁剪的 Perfetto JSON**，不是未经修改的完整 trace：保留全部非 Python 事件，仅保留优化相关 Python frames；保留原时间戳、correlation、pid/tid 和 stream；删去其他 args，并将绝对源码路径改为代码相对路径。完整原件已保留，SHA256 记录在 evidence.json。下载 gzip 文件后可在 Perfetto 中打开；必要时先解压。

| 单个采样 step | A | B |
|---|---:|---:|
| tolist 次数 | 1345 | 465 |
| tolist host inclusive，s | 3.374653 | 0.027383 |
| musaMemcpyAsync 次数 | 7734 | 6854 |
| musaMemcpyAsync host 累计，s | 9.464275 | 6.113583 |
| musaStreamSynchronize 次数 | 6375 | 5495 |
| musaStreamSynchronize host 累计，s | 21.693069 | 24.697737 |
| aten::equal 次数 | 2704 | 2704 |
| aten::equal host inclusive，s | 17.286133 | 20.306999 |
| device busy 区间并集，s | 46.232414 | 46.220304 |
| 大于 100 µs 的 device 空泡合计，s | 1.472281 | 1.227289 |

B 中 cache lookup 896 次、实际 `_read` 16 次：880 次重复读取被消除。correlation 与 host 时间包含关系可将相关活动定位到 `rotary_pos_embedding.py → apply_rotary_pos_emb_thd_torch_rope → tolist`。

与此同时，后续 equal 的次数不变、累计等待增大，device busy 几乎不变。这提示部分等待可能转移到后续同步点，不能把减少的 host inclusive 时间直接换算成 iter 收益；父子 inclusive 时间也不能相加。这里只观察了 rank0，不能推广为所有 rank 的时间线结论。

## 来源与未完成项

PR 分支起点为 `b1511ab68f76d384d2df24c400056df4d3adc4fb`。测试目录是无 `.git` 的运行源码副本，不能声称整个运行环境恰好等于该提交；测试保存了受测源码的初始、候选和恢复哈希。

安装版本元数据：torch/torch_musa 2.7.1.post1、TransformerEngine 2.0.0+1dfcd675、flash_attn 2.6.3、DeepEP 1.1.0+0bd46de、MATE 0.2.3，driver 3.3.8-server。包版本号本身不证明源码未修改。

**完整四项优化栈的 A/B 各 20-step 对照尚待补充**，不得以历史仅 B 组的 200-step 结果替代。本 PR 先以 draft 独立提交，不自动合并。
