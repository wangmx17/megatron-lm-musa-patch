# 本轮 PR 验证的公共运行口径

本文件描述 2026-09-10 实测栈，不声称仅用任意上游仓库或任意镜像即可复现。真实训练的依赖源码、驱动和参数必须一起对齐；算子最小测试不能代替训练。

## 数据与参数

- 8 张 MTT S5000，BF16；TP2 PP1 CP4 EP8 DP1，expert TP1；GBS16、MBS1、seq65536，THD/span attention，Muon。
- 真实 `minicpm5_real_part00071_text_document` indexed 数据；`ALLOW_RANDOM_INIT=1`、`RESUME=0`，未从 Megatron checkpoint 恢复。HF/tokenizer 目录不是已加载训练权重的证据。
- 单项 A/B 各10步，完整栈 A/B 各20步；seed42、LR warmup5。均值排除首步，不拿 profiler 运行的 iter 计时做性能结论。
- 每个 A/B 另有独立6步 trace，rank0 第4步单步采集：freq4、warmup1、active1、repeat1、stack1、shapes0、profile_memory0、modules0。

## 保持一致的原始优化栈

关闭重计算，启用 CE_TE、manual GC、RoPE fusion、router fusion、gradient accumulation fusion 与 grouped GEMM。Muon 原有 `MUON_BATCH_NS=1`、`MUON_BATCH_NS_MAX_B=8` 保持不变；新 PR 扩展的是 TE 独立 Parameter 的跨参数 batching，不是首次设置这两个既有变量。

ACE、shared-expert overlap、EP-overlap、Muon DP1 low-memory、Muon fused pointwise 关闭；这些关闭状态是本轮控制配置，不列为新增性能优化项。

MCCL channels 固定16，buffer16 MiB，DeepEP SMS配置56。DeepEP 实际通过 flex dispatcher 使用；不能单凭遗留 `ENABLE_DEEPEP=0` 推断 DeepEP 没生效，也不能把这些既有通信配置归为新 PR 的独立收益。

## 新增单项开关

| 对照项 | A | B |
|---|---|---|
| RoPE metadata cache | 原始 metadata 读取源码 | 仅加载提交的 cache/helper 源码；无单独运行时开关 |
| TE padded metadata | MUSA_TE_PADDED_METADATA_NO_SYNC=0 | =1 |
| Flex route conversion | MUSA_FUSED_ROUTE_CONVERSION=0 | =1 |
| TE expert Muon NS | MUON_TE_EXPERT_BATCH_NS=0 | =1 |
| 完整栈 | 上述四项均不加载/关闭 | 上述四项共同加载/开启 |

单项 B 不叠加其它三项。每个 case 后恢复源码和临时 helper；启用开关不是持久安装或用户已决定留用的表示。

## 采集与限制

测试直接调用既有训练脚本，入口 wrapper 用 runpy 执行原 pretrain，并在正常返回后记录每 rank 的全程 allocator allocated/reserved 峰值。与历史200步的外层 SSH launcher 不同；已核对实际训练参数，不能宣称启动链路逐字节相同。

数值门禁为逐步 loss 最大相对差不超过0.1%、grad norm 不超过1%；文档同时报告实际差值，不只报告通过。grad norm 不等于全模型逐元素梯度校验；路由和 Muon 另有算子/更新级对照。

完整结束需步数连续齐全、退出码0、无 skip/NaN、8 rank 全程峰值及源码恢复/no ranks 检查。首步600秒/后续300秒进度 watchdog 用于发现停滞；一次成功不抹除先前失败。单次短 A/B 的小幅差异不证明统计显著性。原始 trace 保留，公开附件明确标注裁剪范围与 SHA256。
