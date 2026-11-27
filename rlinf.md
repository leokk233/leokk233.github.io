# RLinf 学习笔记：策略更新了，为什么机器人还可能照旧行动？

读 RLinf 的训练入口时，我最想弄清楚的是，训练器已经更新了模型，为什么还要再做一次权重同步？因为**负责更新策略的 Actor 和负责试动作的 Rollout 是两个组件**。Actor 手里的策略变了，不代表 Rollout 手里的那份也变了。把这两份模型分开看，机器人强化学习的训练循环就容易读懂了。

前两篇看的 nano-vLLM 和 DeepInfer，都在研究怎样让模型拿着现有权重高效地给出输出。RLinf 的 Rollout 也要做推理，但它给出的动作不是最终交付物。动作会送回环境，环境产生新的经验，训练器再用这些经验改策略。这篇只看我跟过的 **ManiSkill3 + OpenVLA-OFT + PPO** 仿真例子。[RLinf 架构概览](https://rlinf.readthedocs.io/en/latest/rst_source/concepts/overview.html) · [VLA 快速入门](https://rlinf.readthedocs.io/en/latest/rst_source/start/vla.html)

## 1. 一次抓放，从哪里开始？

假设任务是把桌上的物体放进盘子。仿真器先给出当前观测，策略结合观测和任务指令提出动作。仿真器执行动作，告诉系统物体变成了什么状态、这一步得了多少奖励、这次尝试是否结束。这个过程重复多次，才攒出可供训练的一批交互经验。

| 组件 | 在这个例子里做什么 | 交出的东西 |
| --- | --- | --- |
| Env（环境） | 用 ManiSkill3 模拟场景和机器人，执行动作并判断结果。 | 新观测、奖励、是否结束等信息。 |
| Rollout（采样） | 用当前载入的 OpenVLA-OFT 策略，根据观测和指令生成动作，与 Env 反复交互。 | 发给 Env 的动作，以及交给 Actor 的交互轨迹。 |
| Actor（训练） | 用收集到的轨迹估计优势，再用 PPO 更新策略。 | 更新后的策略参数，后续同步给 Rollout。 |

最容易混淆的是 **Actor 不等于仿真里实际执行动作的机械臂**。在这套代码里，Rollout 负责用策略提出动作，Env 负责把动作执行出来，Actor 则负责把已经发生的尝试变成一次参数更新。[ManiSkill 示例](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/maniskill.html) · [架构概览](https://rlinf.readthedocs.io/en/latest/rst_source/concepts/overview.html)

## 2. PPO 学的不是「成功就原样重做」

假设机械臂伸手、夹住物体，最后移动时把它掉在了盘子外。一次失败并不能说明整条轨迹上的每个决策都不好。PPO 训练会用采样经验估计**优势**，也就是相对当时预期，这个动作后续带来的结果更好还是更差。优化时会倾向提高正优势动作的概率，降低负优势动作的概率。

但一批经验毕竟来自更新前的策略。PPO 的裁剪目标会限制优化器从过度改变这些已采样动作的概率中获得的收益。这个裁剪是作用在**优化目标里的新旧概率比**，不是裁剪机械臂动作的幅度，也不保证每次更新后成功率都会上升。[PPO 算法说明](https://rlinf.readthedocs.io/en/latest/rst_source/reference/algorithms/ppo.html)

## 3. 代码里，经验和权重走的是两条路

| 我找的入口 | 它回答的问题 |
| --- | --- |
| [`run_embodiment.sh`](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/run_embodiment.sh) 和[快速入门的 YAML 配置](https://rlinf.readthedocs.io/en/latest/rst_source/start/vla.html) | 这次跑哪个训练配方？我读的是 `maniskill_ppo_openvlaoft_quickstart.yaml`，里面要核对模型路径、仿真任务、训练参数和组件分配。 |
| [`train_embodied_agent.py`](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/train_embodied_agent.py) | 配置怎么变成 Env、Rollout 和 Actor 工作组，再交给 Runner？ |
| [`embodied_runner.py`](https://github.com/RLinf/RLinf/blob/main/rlinf/runners/embodied_runner.py) | 一批数据什么时候回到 Actor？新策略又什么时候交给 Rollout？ |

在我对照的同步训练主线里，Runner 先在符合配置的步数上同步策略权重，再启动 Env 与 Rollout 的交互。Actor 收到轨迹，计算优势和回报，然后执行训练。**这一轮更新后的策略要等到后续同步点，才会被 Rollout 用来采样。** 所以「Actor 更新完成」和「Rollout 开始用新策略」是两个不同的事件。读代码时用这两个方向找调用，比从头逐行看更容易跟上。[Runner 源码](https://github.com/RLinf/RLinf/blob/main/rlinf/runners/embodied_runner.py)

## 4. 跑通一轮，和证明机器人学得更好，是两件事

我目前走过的是启动脚本、配置和训练数据流，还没有个人完整训练结果，也没有做真机实验。下一步我会先选一个抓放任务，从官方支持的起始模型出发，用日志确认四件事：动作确实执行了、环境确实返回了经验、Actor 确实更新了、Rollout 确实收到了同步后的权重。完成这一步，只能说训练链路跑通。

要回答策略有没有变好，我会让**起始策略和训练后的策略跑同一组未参与训练的仿真初始场景**，固定评估条件，记下成功次数、试验总数、完成步数和典型失败。`env/success_once` 关心的是每个回合有没有至少成功一次，`env/return` 则是回合累积奖励。做正式对照时，我会看评估记录里的 `eval/success_once`，而不是只看训练日志。如果任务给了过程奖励，机械臂可能因为更常抓住物体而得分更高，却仍然一次也没有放进盘子。总奖励上涨，不等于任务成功率上涨。[RLinf 指标说明](https://rlinf.readthedocs.io/en/latest/rst_source/reference/metrics.html)

这篇读完，我最想记住的不是启动命令，而是训练里的两个问题：**新策略有没有被用起来？用起来之后，它有没有完成更多任务？** 前一个查权重和日志，后一个做固定条件下的对照评估。两者都回答了，才知道下一步该查训练系统，还是继续改策略。
