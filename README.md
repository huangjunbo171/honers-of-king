# 王者荣耀 1v1 智能体

腾讯开悟「智能决策 1v1」V63.0.9 任务一的 PPO 实现。双方英雄默认为鲁班七号（ID 112）。智能体把每帧编码为 350 维观测，通过 LSTM 维护对局状态，输出六个动作分支和一个局面价值估计。

| 从哪里读 | 对应问题 |
| --- | --- |
| [特征编码](agent_ppo/feature/feature_process/) | 双方英雄各占 67 维，小兵预留 144 维，四个建筑各占 18 维。任务一的小兵槽填零。 |
| [逐帧奖励](agent_ppo/feature/reward_process.py) · [权重配置](agent_ppo/conf/conf.py) | 前进项取当帧位置值，塔血量比、击杀和死亡取双方增量之差。权重依次为 0.01、5.0、0.6 和 −1.0。 |
| [智能体入口](agent_ppo/agent.py) · [网络与损失](agent_ppo/model/model.py) | `predict` 从合法动作中采样，`exploit` 选择概率最大的合法动作。目标分支按已选动作键使用对应的目标掩码。 |
| [课程资料索引](docs/README.md) | 查阅任务一手册、奖励算例、任务二扩展和平台操作说明。 |

## 离线验证

在安装了 NumPy 和 PyTorch 的环境中，从仓库根目录运行以下命令。测试用合成帧检查特征布局、换边、奖励符号、LSTM 状态传递和 PPO 裁剪损失。

```sh
python -m unittest discover -s tests -v
```

真实对局由腾讯开悟提供 Gamecore 和平台 SDK。平台从 [PPO 工作流](agent_ppo/workflow/train_workflow.py) 读取[对局配置](agent_ppo/conf/train_env_conf.toml)。
