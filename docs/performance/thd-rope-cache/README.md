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

## CPU metadata cache 保留了哪些原有行为

这里的“更稳妥”不是指 CPU 能修复非法 THD/CP 数据。非法 sequence 边界、长度或
CP 布局在 cached fallback 中同样可能导致 `torch.split` 报错或产生错误位置。
CPU cache 基本保留了 Megatron 原来的 RoPE 数据流；这减少了映射逻辑的改动，
但不足以证明它总体更安全，缓存自身的新增风险见后文：

```text
计算 sequence lengths
        ↓
torch.split
        ↓
_get_thd_freqs_on_this_cp_rank
        ↓
每条 sequence 执行 torch.rope
        ↓
torch.cat
```

CPU metadata cache 只改变第一步的重复读取方式：

```text
第一次读取 CPU lengths
        ↓
相同 cu_seqlens 的后续访问复用
        ↓
其余 split、CP frequency 选择、RoPE 和 cat 逻辑保持不变
```

device fast path 则重新实现了一部分数据流：

```text
解析 cu_seqlens
        ↓
判断每个 token 属于哪条 sequence
        ↓
还原 CP 全局 position
        ↓
生成 mapped_freqs
        ↓
对完整本地 THD tensor 执行 torch.rope
```

device fast path 的实现变化、支持限制和验证边界分别包括：

- 重新实现 CP token-position 映射；
- 只支持 full-RoPE；
- 没有覆盖 `freqs` 的梯度；
- 没有完整检查 `cu_seqlens` 内部数值和 CP 整除关系；
- 增加 Triton gather kernel 和临时 `mapped_freqs` tensor；
- 当前只有 28 组数值测试和 10-step 训练，没有长期训练验证；
- 相对 CPU cache 的一次短 A/B 观察收益约为 0.263%，尚未证明统计显著。

CPU cache 继续使用 Megatron 原来的
`_get_thd_freqs_on_this_cp_rank()`，不重新实现 CP position 映射，支持范围与原
路径基本一致。缓存还通过 `_version` 检测 tensor 的原地或别名修改，通过 weakref
在 tensor 释放后清理条目，最多保留 64 项；inference tensor 无法安全获取 version
时直接不缓存。

CPU cache 仍然存在 device-to-host 同步，但通常不是整个训练只同步一次，而是每个
microbatch 的 `cu_seqlens` 第一次访问时同步一次，随后该 microbatch 的所有
Transformer 层复用：

```text
每个 microbatch 第一次访问 cu_seqlens
        ↓
读取一次 CPU lengths
        ↓
这个 microbatch 的所有 Transformer 层复用
```

已有采样中，每个 step 包含 16 个 microbatch：896 次各层查询只有 16 次真实读取，
其余 880 次命中缓存。因此 CPU 同步已经被压缩到较低频率。

采用策略需要同时评估缓存失效机制和 device 边界保护，不能仅根据处理位置判断安全性。
当前整合分支的代码默认值仍是 `MUSA_THD_ROPE_DEVICE_METADATA=1`，本次只更新说明。

## 三套实现的具体风险对比（2026-09-15复核）

本节比较 patch 中的原始未缓存路径、CPU cached 路径和新增 MUSA kernel。
原路径与 cached 路径的 RoPE 张量计算本来就在 MUSA；区别在于 metadata 的读取、
缓存和分段方式。以下区分实际复现、源码推导和验证范围，不将异常输入推导冒充为
正常训练中已发生的故障。

### MUSA kernel 新增的边界风险

1. **token 总数大于 metadata 描述时可能越界读取。**
   CP=1、`cu_seqlens=[0,4]`、`t.shape[0]=5` 时，原路径和 cache 路径调用
   `torch.split(t,[4])` 会因总长度不匹配报错，该行为已用小测试复现。
   MUSA kernel 按5个token启动，`token=4` 的二分查找最终得到
   `lo=num_sequences=1`，随后读取 `cu_seqlens[lo+1]`，即不存在的第3个元素。
   这是源码推导的越界，未在设备上故意执行非法读取。
2. **token 总数小于 metadata 描述时可能静默接受错误输入。**
   CP=1、`cu_seqlens=[0,6]`、`t.shape[0]=5` 时，原路径和cache路径同样会在
   `torch.split` 报错；MUSA kernel 只处理现有5个token，可能返回结果而遗漏总数错误。
3. **frequency 直接读取缺少行下标保护。**
   `tl.load(freqs + position * stride + offsets, mask=offsets < rotary_dim)`
   只保护列下标，没有检查 `0 <= position < freqs.shape[0]`，异常position或过短的
   频率表可能导致越界读。原路径使用有边界的张量切片，但切片长度不足后原生
   `torch.rope` 的错误行为仍需单独核实，不能声称整个原路径都已有完整保护。
4. **不合法的奇数CP本地长度可能被掩盖。**
   序列长度6、CP=2时，本地长度为3；原CP helper按`3//2`选择前后两个块，只生成
   2个频率位置，无法与3个token完整对应；MUSA公式按3个token生成位置，可能返回
   完整输出。两者都依赖对称双分块契约，但拒绝或暴露非法布局的行为不同。

相关实现见 `musa_patch/thd_rope_device.py` 的 `gather_cp_freqs`：二分查找后的
`cu_seqlens` 读取和frequency读取都需要进一步的边界保护。仅填零并继续训练不能
代替错误报告，否则会将越界故障变成静默错误。

### CPU cache 自身新增的风险

以下两项已用 `thd_metadata_cache.py` 的实际helper在CPU小测试中复现：

```python
cu = torch.tensor([0, 4, 12], dtype=torch.int32)
cache.get(cu)       # [4, 8]
cu.data[1] = 6      # cu._version 没有改变
cache.get(cu)       # 仍为 [4, 8]
# 直接重新读取实际为 [6, 6]
```

这两组长度总和相同，后续 `torch.split` 可能通过，但分段边界已经错误。普通
`cu[1]=6` 会更新version；风险来自绕过该机制的写入，不能把version保护视为覆盖
一切底层数据修改。原始未缓存路径没有跨调用复用旧lengths的问题。

```python
lengths = cache.get(cu)
lengths[0] = 99
cache.get(cu)       # 返回已被污染的列表
```

返回的Python list不是只读对象。当前RoPE调用方没有修改它，所以这是接口使用约束，
不是已证实训练触发的问题；但缓存确实新增了这一风险。identity、weakref、容量上限
均无法代替内容更新检测。

### 不做cache优化，原路径也有的风险

原路径会每次重新读取lengths，仍不完整校验THD/CP数据契约：

| 问题 | 原始路径与CPU cached路径 |
|---|---|
| 分段长度总和与输入不匹配 | `torch.split`会拒绝，已复现 |
| 分段长度为负 | `torch.split`会拒绝，已复现 |
| 序列长度不能被CP整除 | 没有显式校验，整数除法会截断 |
| CP本地长度为奇数 | helper使用整数除法分半，没有完整契约校验 |
| token排列不符合CP对称分块 | 形状可能合法，但位置语义仍可能错误 |
| frequency内容错误 | 形状检查无法判断位置编码内容是否正确 |

例如`cu_seqlens=[0,15]`、CP=4、本地token数为3时，lengths为`[15//4]=[3]`，
实测`torch.split`接受该输入。这只证明split通过，不能证明后续RoPE或CP语义正确。

### 合法输入的映射等价性与证据边界

本次将kernel位置公式与原CP helper的对称切片公式做主机模拟对照：随机变长序列、
CP=1/2/3/4/8及全部rank，共1800组合法布局，位置索引全部一致。采用固定随机种子42，
每组1至12条序列，CP>1时长度取`2*CP`的正整数倍。
这是公式级主机模拟，不是1800组MUSA数值测试，不覆盖Triton编译、显存读写和长期训练。
本次未发现合法布局上的映射错误，也没有证明存在异步读写竞争。

partial-RoPE和`freqs.requires_grad=True`已由入口排除，属于受控支持限制；
临时`mapped_freqs`和Triton依赖属于显存与运行环境成本，均不能直接当作已发生的
数值错误。历史28组设备数值测试和10-step训练的覆盖边界仍然适用。

### 汇报摘要

相比CPU cache，当前MUSA kernel绕过了`torch.split`的长度检查，并以缺少完整边界
保护的指针读取metadata和frequency，异常输入时可能出现越界读或静默错误；
CPU cache本身也存在绕过version更新后复用旧长度、以及返回列表被修改的风险。

因此应补齐device边界保护及错误报告、扩充分段/CP映射测试，同时明确cache的更新
约束；不能按CPU或MUSA简单划分安全与不安全。

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
