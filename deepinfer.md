# DeepInfer 学习笔记：有并发空位，为什么请求还是算不完？

我在写一个简化调度器时遇到一个容易想错的问题：系统允许两条请求同时在场，为什么第二条请求这轮可以一个 token 都不处理？原因是「允许多少条请求在场」和「这一轮一共能处理多少 token」是两个独立的限制。这个小实验成了我理解 DeepInfer 推理流程的入口。

## 1. DeepInfer 是什么

DeepInfer 是 DeepSeek 的高性能推理框架。它关注的是模型训练完成后，怎么组织请求、安排计算并生成输出。我阅读的开源版本提供离线推理核心，支持列表里列出了 DeepSeek-V4.1-Flash，以及 NVIDIA SM100 和华为昇腾 950。仓库同时说明，完整的推理服务还需要额外的组件，例如跨副本负载均衡。能运行离线示例，不等于已经搭好一套在线服务。[该版本 README](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md)

## 2. 一条请求在源码里怎么走

我从 `examples/offline_inference.py` 开始读。示例调用 `LLM.generate()`，接口把输入和采样配置交给引擎。引擎按轮推进请求，最后按请求收集结果。把优化细节先放一边，我把主线分成四个职责。

| 环节 | 它做什么 | 源码入口 |
| --- | --- | --- |
| 请求入口 | 接收提示词，将文字转换成模型能处理的 token，把请求交给引擎。 | [LLM.generate](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/entrypoints/llm.py#L90) |
| 调度 | 决定这一轮安排哪些请求，并为执行准备调度结果。 | [Scheduler](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp) |
| 模型执行 | Worker 接收这一步的任务，调 ModelRunner 准备输入并运行模型前向计算。 | [Worker](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/worker.py#L88) · [ModelRunner](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/model_runner.py#L308) |
| 采样与结果收集 | 根据模型输出和采样参数选择 token，再整理并返回相应请求的结果。 | [Sampler](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/sampler.py#L59) |

这张表描述的是职责，不是现场执行的逐行顺序。比如在源码里，Worker 可以在处理当前计算步骤时整理上一轮结果。把职责拆开看，比较容易分清**调度决定谁去算，模型产生候选分数，采样决定最后选谁**。[离线推理示例](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/offline_inference.py) · [Worker 的执行与后处理](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/worker.py#L88)

## 3. 调度实验：两个名额不等于每条都能算

我用 `scheduler_demo.py` 写了一个只处理整数的模拟。A、B、C 分别有 3,000、3,000、2,000 个待处理的输入 token。每轮最多允许 **2 条请求在场**，这两条请求合计最多处理 **2,048 个 token**。脚本每轮从队首开始分预算，用不完的请求留到下一轮。

| 轮次 | A 处理量 | B 处理量 | C 处理量 | 本轮合计 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2,048 | 0 | — | 2,048 |
| 2 | 952 | 1,096 | — | 2,048 |
| 3 | — | 1,904 | 144 | 2,048 |
| 4 | — | — | 1,856 | 1,856 |

单位为 token。「—」表示请求不在本轮，**0 表示请求已经在场，但没分到预算**。第一轮 B 就是这种情况。第二轮 A 做完后腾出名额，所以第三轮 C 能补进来，但 B 先用掉了 1,904 个额度，C 这轮只处理了 144 个。

此前我也用 **4,096 token/轮**跑过同一组输入。这时 A、B、C 两轮都能处理完，和表里的四轮并不冲突，因为改的是每轮 token 预算，没改两条请求的名额。

这只是用来区分两类限制的 Python 模拟，**不是 DeepInfer 的调度算法复现，也不是 GPU 性能测试**。真实实现还有设备执行、缓存与请求状态等约束，不能拿模拟的轮数换算真实耗时。

## 4. 采样实验：温度为什么会让结果更集中

我用 `sampling_demo.py` 做了另一个小实验。为「巴黎、伦敦、东京、其他」四个候选手动设定分数 3、2、1、0，固定 `top_p = 0.9`。脚本先用温度缩放分数，再把分数转成概率。然后从概率最高的候选开始保留，直到累计概率达到 0.9，最后把保留项重新归一化。每个温度用固定随机种子抽样 1,000 次。

| 温度 | 巴黎 | 伦敦 | 东京 | 其他 |
| ---: | ---: | ---: | ---: | ---: |
| 0.2 | 1,000 | 0 | 0 | 0 |
| 1.0 | 653 | 251 | 96 | 0 |
| 2.0 | 438 | 285 | 167 | 110 |

这里最值得注意的是，**温度 0.2 时巴黎被抽中 1,000 次，不是因为低温天然等于确定性输出**。而是这组分数经过低温缩放后，最高概率的候选单独就达到了 top-p 阈值，其他候选被筛掉了。换一组分数，同样的温度和 top-p 未必只留下一个候选。

这些是人工分数下的抽样次数，不是真实模型的预测，也不是正确率。这个实验帮我把两件事分开了：**模型给候选打分，采样规则决定怎样从这些候选里选出输出**。

## 5. 异步调度和我的模拟有什么区别

我的模拟每轮在 Python 里顺序完成预算分配与更新，没有设备计算。DeepInfer 的 README 里说的**异步调度**，则是让 CPU 侧的批次调度和计算设备执行有机会重叠。直观上，设备处理一批工作时，CPU 可以推进下一批的部分准备，而不是每次都等设备空下来才开始安排。[异步调度说明](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md) · [Worker 的执行步骤](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/worker.py#L88)

所以，**一轮放了多少请求**和**CPU 与设备是否能重叠工作**是两个不同的问题。调度玩具只演示了前者的两类约束，不能用来证明异步执行带来的性能收益。

## 6. 学到了什么，下一步验证什么

读完源码和跑完这两个脚本后，我能从请求、调度、模型执行和采样四个环节解释一条生成请求。最有用的不是记住某个参数的默认值，而是知道该问什么：请求没被推进，是没排进这一轮，还是这一轮的计算额度用完了？

我目前做的是**源码阅读与本地模拟**，没有把它们当作 DeepInfer 本体的运行结果。下一步如果有兼容设备，再按对应版本跑最小离线示例。先确认同一组输入能稳定跑通，再记录模型版本、硬件、输入输出长度和整批吞吐量。有了实测以后，才值得讨论优化是否真的带来收益。
