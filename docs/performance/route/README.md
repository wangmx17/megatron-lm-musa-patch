# Flex MoE 路由转换融合：独立验证

## 改动原因与实现

DeepEP dispatch 返回每个 token 的局部 expert 索引及概率，后续 expert permute 需要 dense multihot routing map 与按局部 expert 排列的概率。这一步是路由表示转换，不是传输 token 的通信本身。

`musa_patch/moe_route_conversion.py` 用一个 Triton forward kernel 完成输出初始化、索引 scatter、概率写入和 backward position map 保存；backward kernel 根据 position map 将概率梯度还原到输入 top-k 位置，padding 对应梯度为零。非连续上游梯度先 contiguous。没有梯度的整数索引不参与求导。

通过 `MUSA_FUSED_ROUTE_CONVERSION=1` 显式开启，默认关闭。只替换 `_DeepepManager._indices_to_multihot`，不改变 DeepEP dispatch/combine 协议、ACE、通信 buffer 或其他 dispatcher。

## 支持边界

快速路径要求 MUSA 上同 device、连续二维、同 shape 的 int64 indices 与 float32 probabilities，正的 token/top-k/expert 大小及可用 Triton；不符合这些布局/类型条件时保留原路径。

值语义遵循 DeepEP 输出契约：每行有效局部 expert 索引唯一，或为 -1 padding。不是任意 scatter API，不支持重复有效 expert 或任意负索引；为避免新增 device 同步，不逐值读取 GPU 索引做运行时验证。启用位置因此限定为实际 DeepEP manager。

## 正确性检查

MUSA 单项测试覆盖 `(tokens, local_experts, topk)` 为 `(1,5,3)`、`(37,20,16)`、`(64,32,8)`，包含全 padding 行、混合 padding、非连续上游梯度；routing map、概率及 probability gradient 均与参考实现逐元素完全一致。

训练比较另行检查逐步 loss 和 grad norm；grad norm 不等于全模型逐元素梯度校验。

## 10-step A/B 性能

真实 indexed 数据、随机初始化，8 张 MTT S5000，BF16、THD/span、Muon；TP2 PP1 CP4 EP8 DP1、GBS16、MBS1、seq65536。仅开启本项，未叠加其它三个待提交优化。均值取非 profiler 运行 step 2–10；各完整 10/10，rc=0，无 skip/NaN，源码/helper 恢复通过。

| 完整非 profiler 运行 | A | B |
|---|---:|---:|
| step 2–10 平均 s/iter | 48.933211 | 48.841844 |
| 全 rank 全程最大 allocated，MiB | 69920.208 | 69957.241 |
| 全 rank 全程最大 reserved，MiB | 74036 | 74132 |
| 峰值记录 rank 数 | 8 | 8 |

均值下降 0.186717%，是很小的单次短测变化，不证明稳定收益或统计显著性。最大逐步 loss 相对差 0.001882157%，grad norm 相对差 0.012015861%。allocator 峰值不等同于设备总显存采样值。

独立 A/B trace 各6步、rank0第4步单步 stack 采集，均完成6/6、rc=0、8 rank 峰值齐全且恢复通过。最大 loss 相对差0.000827210%，grad norm 相对差0.004045471%；profiler iter 不用于性能收益。

## Trace 机制证据

A原 `_indices_to_multihot`432次、host inclusive 1.225548秒；B `convert`432次、0.102673秒。B forward融合kernel432次、device累计0.020190秒；backward融合kernel432次、0.016613秒，证明实际走了融合路径。

| rank0单采样step | A | B |
|---|---:|---:|
| aten::nonzero次数 | 1312 | 16 |
| aten::index次数 | 1296 | 0 |
| aten::_index_put_impl_次数 | 1344 | 48 |
| musaMemcpyAsync次数 | 7734 | 5142 |
| MemcpyAsync host累计，s | 9.386832 | 4.089292 |
| musaStreamSynchronize次数 | 6375 | 5079 |
| StreamSynchronize host累计，s | 21.655947 | 26.956039 |
| aten::equal次数 | 2704 | 2704 |
| equal host inclusive，s | 17.249988 | 22.571542 |
| device busy并集，s | 46.209784 | 46.550303 |
| device首末跨度，s | 50.038607 | 49.765633 |
| 大于100µs空泡合计，s | 1.535035 | 1.205290 |

推断部分等待转移到了后续同步点：MemcpyAsync host时间减少约5.30秒，equal等待却增加约5.32秒，B FA backward varlen检查的448次equal合计22.304963秒。不能把移除的host inclusive时长当作直接iter收益，也不能将父子嵌套统计相加。空泡减少但device busy增加，单次profile不证明GPU计算量显著下降。

device-to-host分析通过correlation找到runtime，再按相同pid/tid的时间包含关系找相关Python栈。A/B最大空泡约22.419/21.860ms，下一kernel在空泡开始后才host launch，提示局部host提交受限；保留栈不足以确定具体操作，不进一步强行归因。

[A审查trace](A.review.trace.json.gz) · [B审查trace](B.review.trace.json.gz) · [逐步数值、全rank峰值、哈希及裁剪声明](evidence.json)

公开附件是裁剪后的Perfetto JSON，不是完整原件：保留全部非Python事件、相关Python frames、原始时间戳及correlation/stream，移除其它Python frames和绝对安装路径等元数据。原件SHA256、大小、事件计数随证据提供，原件另外留存。缺失frame不等于没有调用。

完整四项20-step A/B仍待运行，保持draft，不自动合并。
