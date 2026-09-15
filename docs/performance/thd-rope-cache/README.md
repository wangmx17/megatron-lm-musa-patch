# THD RoPE、Context Parallel 与 CPU metadata cache

## 问题背景

Packed THD 输入使用 `cu_seqlens` 表示每条 sequence 的边界。例如：

```text
cu_seqlens = [0, 5, 8, 12]
lengths     = [5, 3, 4]
```

Transformer 各层处理的是同一批 token。层与层之间 Q/K 的数值会变化，token
排列和 sequence 边界不会变化，因此各层反复读取的是同一个 `cu_seqlens`。

旧路径每次执行：

```python
(cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
```

当 `cu_seqlens` 位于 MUSA 时，`.tolist()` 会使 CPU 等待 device 结果。随后
Python 使用 lengths 调用 `torch.split`，逐 sequence 处理 RoPE，再用
`torch.cat` 拼回完整 THD tensor。

## CPU metadata cache

`musa_patch/thd_metadata_cache.py` 在第一次访问一个 `cu_seqlens` tensor 时读取
CPU lengths，后续层复用同一个 Python list。缓存同时检查 tensor identity、
weakref 和 version counter；原地或别名修改会使缓存失效。无法安全获取 version
的 inference tensor 不缓存。缓存最多保留 64 项，tensor 释放时对应条目自动清理。

该优化减少重复 device-to-host 同步，但第一次读取仍需 `.tolist()`，逐 sequence
的 Python `split`、循环和 `cat` 也仍然存在。

## MUSA device fast path

`musa_patch/thd_rope_device.py` 使用 Triton kernel 直接在 MUSA 上读取
`cu_seqlens`。kernel 为每个本地 token 查找所属 sequence，根据 CP rank 还原它在
完整 sequence 中的 RoPE position，并生成连续的 `mapped_freqs`。随后对完整本地
THD tensor 一次调用原生 `torch.rope`。

这条路径不读取 CPU lengths，也不执行逐 sequence 的 Python
`split`、循环和 `cat`。代价是增加一次 frequency-gather kernel，并分配形状为
`[num_local_tokens, rotary_dim]` 的临时 MUSA tensor。

## 为什么不能无条件全走 MUSA

当前 device fast path 是针对一组明确输入契约实现的专用路径，并不覆盖
`rotary_pos_embedding.py` 可以接收的全部输入。它要求：

- Triton 可用，且 `t`、`cu_seqlens`、`freqs` 位于同一 MUSA device；
- `t` 为三维非空 THD tensor；
- `cu_seqlens` 为一维、contiguous、`torch.int32`，并至少包含两个元素；
- `freqs` 形状为 `[S, 1, 1, R]`；
- `R` 等于完整 head dimension 且为偶数，即 full-RoPE；
- `freqs.requires_grad` 为 false；
- sequence lengths 满足 Megatron CP 可整除和对称双分块契约。

不能直接删除这些判断并强制调用 device 实现，原因包括：

1. 当前原生 MUSA `torch.rope` 路径不支持 partial-RoPE。如果
   `freqs.shape[-1] != t.shape[-1]`，强制进入会报错，且还缺少未旋转维度的拼接逻辑。
2. Triton gather kernel 没有实现对 `freqs` 的梯度传播，不能处理
   `freqs.requires_grad=True` 的输入。
3. CPU 调试、无 Triton 环境、不同 dtype/layout 和 unfused RoPE 仍需兼容实现。
4. device kernel 根据 Megatron 的 CP 布局公式计算位置；不满足该布局的输入不能
   当作同一种数据直接处理。
5. 环境变量开关提供运行问题发生时的快速回退能力，避免必须修改代码才能恢复训练。

因此 fallback 的作用不是让一次调用同时使用 CPU 和 MUSA，而是在 device 专用
实现不能保证正确时保留兼容路径。如果当前 MiniCPM5 输入始终满足上述条件，实际
训练会在函数开头进入 device fast path 并立即返回，CPU metadata cache 不会执行。

## 实际判断顺序

```text
config.apply_rope_fusion == true
        ↓
进入 THD torch.rope 路径
        ↓
MUSA_THD_ROPE_DEVICE_METADATA 默认开启
        ↓
device_thd_rope_supported(...) 检查 device、dtype、shape 和梯度条件
        ↓
├─ 全部满足：MUSA device fast path，直接返回
└─ 任一不满足：CPU metadata cached split fallback

config.apply_rope_fusion == false
        ↓
unfused THD 路径，使用 CPU metadata cached split fallback
```

当前 MiniCPM5 `run_16a3b.sh` 默认设置 `ENABLE_ROPE_FUSION=1`；
`MUSA_THD_ROPE_DEVICE_METADATA` 在代码中的默认值也是 `1`。这只表示软件开关已经
打开，最终是否命中仍由运行时 tensor 条件决定。

## 验证边界

CPU cache 单测覆盖缓存命中、原地和别名修改失效、weakref 生命周期、有界容量及
inference tensor 回退。MUSA device 单测覆盖 CP=1/2/4 的全部 rank、1/16 heads、
interleaved 开关和三条变长 packed sequence，共 28 组 BF16 前向和输入梯度对照，
最大绝对差为 0。

测试没有覆盖所有 dtype、partial-RoPE 的新实现、非法 sequence 边界或其他 CP
布局。整合 PR 不上传 trace、`.json.gz` 或 profiler 原始附件。
