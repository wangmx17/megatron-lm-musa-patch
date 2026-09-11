# Flex/DeepEP shared expert 显式 TP/SP 状态机

## 一句话结论

这个补丁允许 Flex/DeepEP 使用 Megatron 官方的
`--moe-shared-expert-overlap` 标志，但不会直接调用已经关闭内部 TP/SP 通信的
shared-expert `forward()`。它在异步 dispatch 提交后完整执行外部
`AllGather -> FC1 -> activation -> FC2 -> ReduceScatter -> output` 状态机，并将
shared-expert GEMM 留在当前计算流，避免在目标 MUSA 栈上引入第二条并发 TE GEMM 流。

两轮反序 10-step A/B 的 step 2--10 合并均值由 `48.004611` 降至
`47.726244 s/iter`，改善 `0.278367 s/iter`，约 `0.5799%`。数值、退出、
NaN/skip/hang 门禁通过；B 的全 rank 峰值 allocated 增加约 `1.70 GiB`。

## 初学者需要先理解的三个概念

### 1. routed expert 和 shared expert

MoE 层里，router 只把每个 token 发给少数 routed expert。shared expert 则处理
所有 token，最后两条分支相加。Flex/DeepEP 的 dispatch/combine 负责 routed token
跨 EP rank 的通信；它不是 shared expert 的 TP/SP 通信。

### 2. TP 和 SP 为什么需要通信

TP（tensor parallel）把一层的权重拆到多个 rank。SP（sequence parallel）又把
输入 token 沿 sequence 维拆开。以 TP2 + SP 为例，每个 rank 开始时只拿到一半
sequence；shared FC1 计算前必须 AllGather 得到完整输入，FC2 后再 ReduceScatter，
让输出重新回到每个 rank 的 sequence 分片。

因此 TP/SP collective 是正确性的一部分，不是可以直接删除的额外开销。

### 3. 为什么打开官方 overlap 标志后，TE Linear 内部 TP/SP 会被关闭

普通路径把通信藏在 TE Linear 里面：

```text
TE FC1 内部 AllGather -> FC1 -> activation -> FC2 -> TE FC2 内部 ReduceScatter
```

通信藏在 Linear 内部时，外层 dispatcher 无法把它移动到更早或更合适的位置。
所以 Megatron 的 official overlap 模式在 `SharedExpertMLP.__init__` 中主动把两个
TE Linear 的 `parallel_mode` 设为 `None`，同时关闭 TE 的 UB overlap 标志。它的
含义不是“以后不做 TP/SP 通信”，而是“Linear 不再自动做，由外层状态机显式做”。

正确的新路径必须完整调用：

```text
pre_forward_comm(input)        # SP AllGather，或非 SP 下的 TP copy
linear_fc1_forward_and_act()   # FC1 + activation
linear_fc2_forward()           # FC2
post_forward_comm()            # SP ReduceScatter，或非 SP 下的 TP reduce
get_output()                   # gate、流依赖和缓存清理
```

如果打开标志后仍直接写 `shared_experts(input)`，TE Linear 内部通信已经被关闭，
外部状态机又没有执行，程序可能仍能跑完，但每个 rank 只基于局部分片计算，结果不再
等价。先前错误原型正是这种情况：10/10、rc=0，却出现 grad norm 约 2%--3% 偏移，
同时 TP/SP stream 18 整段消失。能跑完不代表语义正确。

## 这个补丁具体怎么调度

仅当以下两个 official 条件同时成立时接管：

```text
moe_shared_expert_overlap == True
moe_token_dispatcher_type == "flex"
```

执行顺序为：

```text
router / dispatch preprocess
        |
        v
提交 Flex/DeepEP asynchronous dispatch
        |
        v
在当前计算流完整执行 shared expert 显式 TP/SP 状态机
        |
        v
等待/消费 dispatch 输出并计算 routed experts
        |
        v
DeepEP combine + shared output
```

实现分为四处：

- `TransformerConfig.__post_init__`：只放开 Flex + shared-overlap 这一项组合，仍保留
  shared-expert recompute 与 EP-overlap 的官方互斥检查；
- `SharedExpertMLP.__init__`：让官方初始化继续关闭 TE Linear 内部 TP/SP，但把
  shared stream 绑定到当前计算流；
- `MoELayer.forward`：dispatch 提交后一次性执行五态显式状态机，不跨 routed expert
  长时间保存 FC1 activation；
- `MoELayer.backward_dw`：Flex 显式路径同时提交 routed 和 shared expert 的延迟
  weight-gradient 计算。

为什么不用第二条计算流：目标 MUSA 栈曾出现并发 TE GEMM 的 hang/OOM 风险。当前
实现只让 DeepEP 使用自己的通信流，不主动制造两条 TE GEMM 计算流。它优先保证语义
与稳定性，然后验证调度带来的小幅收益。

该 Megatron 版本的 `pre_forward_comm` 只有一个 `input` 参数，并在开头让 shared
stream 等待 current stream。补丁没有改这个公共方法签名，而是在 Flex 目标路径复用
其主体；因为 shared stream 已绑定为 current stream，所以省去冗余 self-wait。非 Flex
路径仍调用原方法。

## 如何启用

MiniCPM5 示例脚本使用一个总开关：

```bash
ENABLE_FLEX_SHARED_EXPERT_EAGER=1 bash run_16a3b.sh
```

脚本会同时导出 `MUSA_FLEX_SHARED_EXPERT_EAGER=1` 并追加 official
`--moe-shared-expert-overlap`。只设置补丁环境变量但不打开 official flag 不会接管；
非 Flex dispatcher 也完全走原实现。

## 两轮反序 10-step A/B

共同条件：真实 indexed 数据、8 张 MTT S5000、BF16、THD/span、Muon，
TP2/PP1/CP4/EP8/DP1、GBS16、MBS1、seq65536。运行时共同叠加 PR #8--#11；本 PR
仍从默认分支独立提交，因此这里报告的是“在该共同栈之上的增量收益”。性能均值排除
step 1，使用 step 2--10。

| 顺序 | A：关闭本项 | B：打开本项 | B-A | 相对改善 |
|---|---:|---:|---:|---:|
| A -> B | `47.995222 s` | `47.782900 s` | `-0.212322 s` | `0.4424%` |
| B -> A | `48.014000 s` | `47.669589 s` | `-0.344411 s` | `0.7173%` |
| 合并 | `48.004611 s` | `47.726244 s` | `-0.278367 s` | `0.5799%` |

两次顺序相反但收益方向一致。短测仍不等于长期统计显著性，因此 PR 保持 draft。

## 数值、显存与退出门禁

| 门禁 | A -> B | B -> A |
|---|---:|---:|
| 最大逐 step loss 相对差 | `1.2129e-5` | `1.1888e-5` |
| 最大逐 step grad norm 相对差 | `1.3826e-4` | `1.0013e-4` |
| A/B 完成 | `10/10, rc=0` | `10/10, rc=0` |
| skip / NaN / hang | `0 / 0 / 无` | `0 / 0 / 无` |

全程峰值取八个 rank 中的最大值：

| 顺序 | 组别 | max allocated | max reserved |
|---|---|---:|---:|
| A -> B | A | `69951.764 MiB` | `74430 MiB` |
| A -> B | B | `71688.539 MiB` | `74928 MiB` |
| B -> A | A | `69941.102 MiB` | `74592 MiB` |
| B -> A | B | `71687.161 MiB` | `74930 MiB` |

B allocated 增加约 `1.70 GiB`。80 GiB 卡上未 OOM，但长跑、profiler、checkpoint
保存或其他 workspace 叠加时仍需关注；当前没有 200-step 长稳结论。

## A/B Trace：证明了什么，没有证明什么

| rank0 单采样 step | A | B | B-A |
|---|---:|---:|---:|
| device span | `48.092580 s` | `47.778596 s` | `-0.313984 s` |
| device busy union | `46.385290 s` | `46.051411 s` | `-0.333879 s` |
| 大于 100 us 空泡 | `0.696247 s` | `0.704146 s` | `+0.007899 s` |
| stream 0 busy | `32.984835 s` | `32.685092 s` | `-0.299743 s` |
| DeepEP stream 5 busy | `7.008463 s` | `6.049945 s` | `-0.958518 s` |
| cached_notify_combine | `3.603463 s` | `2.574461 s` | `-1.029003 s` |
| TP/SP stream 18 busy | `5.264307 s` | `5.830193 s` | `+0.565886 s` |
| DeepEP x GEMM | `0` | `0` | 没有形成设备并发 |

stream 18 的 AllGather 从 `2752 / 3.166814 s` 变为 `2320 / 2.701724 s`，
减少 `432` 次，恰好等于 `27 MoE layers x 16 microbatches`；ReduceScatter 数量
保持 `1824` 次，但累计时长从 `2.092927` 增至 `3.123185 s`。这说明必要的 TP/SP
通信仍存在，显式状态机避免了一次每层/每 microbatch 的额外 AllGather，并改变了
资源竞争和关键路径。

这两份 Trace **没有**证明 shared GEMM 与 DeepEP kernel 真正重叠：交集仍为 0。
因此 PR 名称使用“eager explicit TP/SP state machine”，不把约 0.58% 收益包装成
不存在的 device compute/communication overlap。

[A Perfetto 审查 Trace](A.review.trace.json.gz) ·
[B Perfetto 审查 Trace](B.review.trace.json.gz) ·
[机器可读证据](evidence.json)

公开 Trace 是裁剪副本，不是原始文件：保留全部非 Python 事件、相关 Python frame、
原始时间戳、correlation 和 stream；移除无关 Python frame、tensor 元数据与绝对
安装路径。原始文件 SHA256、大小和裁剪文件 SHA256 记录在 `evidence.json`。

## PR 包装层运行时验收

性能 A/B 使用直接源文件候选。为确认 PR 中的环境变量注册和 monkey patch 包装层也能
在默认分支配套的 Megatron 版本运行，另做了 1-step 真实数据 smoke：最终 `1/1`、
`rc=0`、loss `12.23689`、grad norm `52.501`、skip/NaN `0/0`。

提交前的前两次包装 smoke 均在 iteration 0 失败，分别暴露出 `pre_forward_comm`
签名差异以及该版本没有 `wait_current_stream` 方法。最终实现不再假定这两个新接口，
第三次通过后才提交。失败运行不计入正确性或性能证据。

## 当前限制

- 只验证 Flex/DeepEP、TP2 + SP、BF16 的 MiniCPM5 shape；
- 不与 `overlap_moe_expert_parallel_comm` 或 shared-expert selective recompute 共用；
- 没有第二条 shared GEMM stream，也没有 device 上的 DeepEP x GEMM 重叠；
- 当前证据是两轮反序 10-step，不是 200-step 长稳测试；
- 约 `1.70 GiB` allocated 增量需要在合并前由维护者评估。
