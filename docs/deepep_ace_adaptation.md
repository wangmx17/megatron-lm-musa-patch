# DeepEP ACE 适配说明

## 1. 背景

该修改位于 `megatron-lm-musa-patch`，用于把 Megatron-LM 的 DeepEP fused all-to-all token dispatcher 接入 MUSA DeepEP 的 ACE（Async Copy Engine）注册内存窗口。

本次修改不是对 Megatron-LM 源码的直接修改。MUSA patch 在加载时替换 Megatron-LM 中的部分 DeepEP dispatcher 方法，并复用 TransformerEngine（TE）的 fused permute/unpermute 接口。

涉及文件：

```text
musa_patch/deepep_ace/__init__.py
musa_patch/deepep_ace/fused_a2a_ace.py
musa_patch/deepep_ace/token_dispatcher.py
```

## 2. 为什么需要修改

### 2.1 ACE Buffer 的 `token_num` 语义与旧配置不一致

DeepEP ACE 构造参数中的 `token_num` 表示单个 EP rank 在 dispatch 前的本地 token 容量，combine 可用容量为：

```text
combine_capacity = local_token_capacity * router_topk
```

旧启动配置曾把 EP 全局 token 数（例如 `65536`）作为 `DEEPEP_ACE_TOKEN_NUM`。该数值不是 ACE 原生接口要求的本地容量，会导致多 GiB 级别的过度分配；如果 ACE 元数据没有正确传入，还可能出现容量为 0 或实际 token 数超过容量的错误。

因此，新实现包装 `FusedDispatch.forward`，在第一次 dispatch 时读取真实的：

- 本地 token 行数；
- hidden size；
- router top-k。

随后使用这些真实维度创建 `use_ace=True` 的 DeepEP Buffer。动态模式默认开启，避免依赖容易配置错误的全局 token 数。

### 2.2 仅有相同 shape 不足以启用 ACE

ACE 要求 dispatch/combine 使用 DeepEP 注册窗口中的实际内存地址。普通 PyTorch tensor 即使 shape、dtype 完全相同，只要底层指针不属于 ACE 窗口，就不能作为正确的 ACE combine 输入。

因此，`token_dispatcher.py` 做了以下适配：

1. `fused_permute_with_probs` 的输出写入 ACE 预分配 hidden buffer；
2. `fused_unpermute` 的输出直接写入 ACE combine input buffer；
3. routing probability 的反向梯度通过 autograd hook 搬入 ACE probability buffer；
4. 检查返回 tensor 的 data pointer，确保 TE 没有忽略预分配输出；
5. 检查 shape、dtype 和容量，在错误数据进入 DeepEP kernel 前给出明确报错。

这些修改的目的不仅是减少一次 tensor 分配或拷贝，更重要的是保证 DeepEP kernel 收到的输入确实来自已注册的 ACE 内存窗口。

### 2.3 Buffer 必须按运行时几何信息重建

Megatron 的 fused all-to-all 使用进程级共享 Buffer。已有 Buffer 可能由非 ACE 路径创建，也可能与当前 EP group、hidden size、top-k 或 token capacity 不匹配。

新的 `get_buffer` 会检查：

- EP process group；
- NVL/RDMA buffer 容量；
- `use_ace` 状态；
- 本地 token capacity；
- hidden size；
- top-k；
- ACE buffer 数量。

只要现有 Buffer 无法满足当前配置，就按新的运行时参数重新创建，避免复用不兼容的全局缓存。

### 2.4 导入顺序必须固定

`musa_patch/deepep_ace/__init__.py` 按以下顺序加载：

```python
from . import fused_a2a_ace
from . import token_dispatcher
```

`fused_a2a_ace` 必须先替换 Megatron fused all-to-all 的 `get_buffer`，然后 `token_dispatcher` 才能导入并使用替换后的实现。反转顺序可能使 dispatcher 保留旧函数引用，导致 ACE Buffer 适配不生效。

## 3. 关键执行路径

```text
FusedDispatch.forward
  -> 捕获本地 token/hidden/top-k
  -> 创建或复用 ACE Buffer
  -> TE fused permute 写入 ACE 预分配区
  -> MoE expert 计算
  -> TE fused unpermute 写入 ACE combine 输入区
  -> DeepEP combine 校验并使用注册内存

Backward
  -> routing probability gradient hook
  -> 梯度写入 ACE probability buffer
  -> DeepEP backward combine 使用注册内存
```

首次走通相应路径时会输出一次性标记：

```text
[deepep_ace] Buffer ready ...
[deepep_ace] TE permute hidden-gradient target registered ...
[deepep_ace] TE unpermute wrote registered combine input ...
[deepep_ace] routing probability gradient moved to registered buffer ...
[deepep_ace] registered combine input verified ...
```

这些标记用于确认 ACE 不只是被环境变量打开，而是实际进入了注册内存路径。

## 4. 启用条件

MUSA patch 的 ACE 加载入口由以下变量控制：

```bash
export USE_DEEPEP_ACE=1
```

动态本地 token 容量默认开启，也可以显式设置：

```bash
export DEEPEP_ACE_DYNAMIC_TOKEN_NUM=1
```

训练侧还必须实际选择 DeepEP/flex token dispatcher，并满足以下依赖：

- DeepEP `1.1.0+8ac8f4d` 或具有相同 ACE Buffer API 的版本；
- 支持预分配输出参数的 TE fused permute/unpermute；
- routing probability 使用 FP32；
- MUSA patch 在 Megatron 创建和使用 DeepEP manager 前完成加载。

已验证配置使用一个 ACE buffer 和 buffer index 0。增加 buffer 数量或切换 buffer index 需要单独验证，不能直接沿用当前结论。

## 5. 验证结果

ACE 单变量真实数据 A/B 配置：

```text
TP=1, PP=2, CP=4, EP=4
DeepEP=1.1.0+8ac8f4d
FA-P2P=关闭
每组20 step，统计iter 2-20
```

结果：

| 配置 | iter 2-20 平均耗时 | 完成状态 |
| --- | ---: | --- |
| ACE 关闭 | 51.3108 s | 20/20，status 0 |
| ACE 开启 | 50.6397 s | 20/20，status 0 |

ACE 平均减少 `0.6711 s/iter`，耗时下降约 `1.31%`。两组均无 skipped iteration 和 NaN。

上述性能数据来自包含相同 ACE 核心路径的已验证候选版本。本次精简提交删除的仅是默认关闭的 allocator/cache 实验代码；精简提交已通过 Python 语法检查和配套 Megatron-LM 导入预检，但未重新执行完整 20-step 训练。

## 6. 本次明确不包含的内容

本次修改不包含：

```text
musa_patch/deepep_ace/memory.py
DEEPEP_ACE_EMPTY_CACHE_BEFORE_BACKWARD
DEEPEP_ACE_RELEASE_CACHE_BEFORE_DISPATCH
DEEPEP_ACE_RELEASE_CACHE_BEFORE_BWD_COMBINE
```

这些功能属于后续显存/allocator 缓解实验，不是 ACE 正确性适配的必要组成：

- `memory.py` 会全局替换 Megatron pipeline schedule 的 backward 入口；
- `torch.musa.empty_cache()` 位于训练热路径时可能引入同步、破坏缓存复用并降低性能；
- 现有 1.31% 收益测试没有启用这些策略。

如果后续确实遇到 ACE workspace OOM，应把它们作为独立修改重新评估，并至少完成 OOM 复现、开启/关闭 A/B、iter 性能、loss/grad、skip/NaN 和退出状态检查。

## 7. 当前边界与注意事项

- 该实现针对当前 MiniCPM5 + Megatron + TE + DeepEP 组合完成验证；升级其中任一组件后应重新检查 API 和内存指针约束。
- `DEEPEP_ACE_TOKEN_NUM` 在关闭动态模式时表示本地 dispatch capacity，不是 EP 全局 token 数。
- capacity 或 data pointer 校验失败时不应绕过检查继续训练，应先确认实际导入路径、DeepEP 版本和运行时几何信息。
- FA-P2P 等 attention/CP 通信优化不属于 ACE 修改，其额外收益不能计入 ACE 的 1.31%。
