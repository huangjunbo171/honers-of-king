# PPO V2 对战 V1

腾讯开悟「智能决策 1v1」评估，平台版本 V63.0.9。任务号 879957，提交时间为 2026-11-19 14:47:20（北京时间）。双方均使用鲁班七号，算法为 PPO，训练模式为分布式。

| 阵营 | 模型 | 训练步数 | 训练时长 | 获胜局数 |
| --- | --- | ---: | ---: | ---: |
| A | ppo_V2_2240 | 2240 | 2 小时 23 秒 | 6 |
| B | ppo_V1_184 | 184 | 10 分 41 秒 | 0 |

共 6 局，超时 0 局、异常 0 局。**V2 以 6:0 获胜**。该结果只涵盖此项评估的 6 局，截图未提供逐局换边信息。

## 模型与源码

| 导出包 | SHA-256 |
| --- | --- |
| [ppo_V1_184.zip](https://github.com/huangjunbo171/honers-of-king/releases/download/ppo-v1-v2/ppo_V1_184.zip) | e65f1aaf6a36042785cd2c90b4d30c7c07c990e0af421ec85eec18df66544c10 |
| [ppo_V2_2240.zip](https://github.com/huangjunbo171/honers-of-king/releases/download/ppo-v1-v2/ppo_V2_2240.zip) | 8b060b31ec32df2deea859615b2d81a4a2e338f25bd8bb02fc8647a358475e38 |

ZIP 为开悟原始导出包，分别包含 model.ckpt-184.pkl 和 model.ckpt-2240.pkl，以及随模型导出的代码和配置。[GitHub Release](https://github.com/huangjunbo171/honers-of-king/releases/tag/ppo-v1-v2)

两个包的智能体、模型和工作流源码一致。conf/configure_app.toml 的导出配置只在评估模型编号上不同，分别为 184 和 2240。仓库使用通用运行配置，各模型的导出配置保存在对应 ZIP 中。
