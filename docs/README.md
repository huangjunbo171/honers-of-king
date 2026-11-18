# 腾讯开悟王者荣耀 1v1 实验

课程「AI 博弈方法实践」分配的实验是「智能决策1v1」。实验页显示版本 `V63.0.9`、小组实验、分布式训练，以及 PPO / DIY 两种算法入口。实验目标是控制己方英雄，利用走位、补兵和对抗积累优势，率先摧毁敌方水晶。记录胜负时，以该实验页和运行时的结算结果为准。[课程实验页](https://tencentarena.com/p/edu/course/7129/exp/76485)

## Agent 阅读顺序

1. [平台操作与评估](platform-and-evaluation.md)说明如何识别任务包、短测、提交训练和评估模型。
2. [4.2 构建智能体](task1-baseline.md)按文件列出归一化、英雄、机关、四项奖励以及验证用例。
3. [4.3 策略优化](task2-optimization.md)列出九项奖励、小兵特征、全局特征和调参方式。

在真实任务包中，先以 `agent_ppo/`、`TODO`、`set_cur_calc_frame_vec`、`get_reward` 和 `REWARD_WEIGHT_DICT` 为入口定位实现。用代码包中的字段、枚举和模型尺寸确定接入方式。文档中的 Python 代码块是独立参考实现，可用合成帧核对公式和分支。平台包含 Gamecore、训练工作流、PPO 模型以及评估环境。

## 资料索引

| 课程资料 | 对实验的用途 |
| --- | --- |
| [第 3 章平台实验指导（学生版）](https://tencentarena.com/p/edu/course-content/7129/7288/209105?fileId=954901&projectId=36817) | 正文第 3–7、51–59 页提供 IDE 调试、训练提交、模型管理和评估流程。操作示例使用峡谷漫步。1v1 的环境、时长和指标由本课实验设置提供。 |
| [第 4 章讲义](https://tencentarena.com/p/edu/course-content/7129/7288/209106?fileId=954915&projectId=36818) | 第 7–13 页介绍目标、观测与动作，第 14–21 页介绍 PPO 和网络架构，第 40、52 页给出调参和基线对手路线。讲义区分终局水晶目标与用于快速验证的首塔目标。 |
| [第 4 章教案](https://tencentarena.com/p/edu/course-content/7129/7288/209106?fileId=954916&projectId=36818) · [配套课件](https://tencentarena.com/p/edu/course-content/7129/7288/209106?fileId=954914&projectId=36818) | 用于定位项目规则、350 维观测示意、六分支动作以及训练曲线。讲义按课件页码展开相关内容。 |
| [第 2 课实验指导（学生版）](https://tencentarena.com/p/edu/course-content/7129/7288/209107?fileId=954918&projectId=36818) · [教师版](https://tencentarena.com/p/edu/course-content/7129/7288/209107?fileId=954919&projectId=36818) | Task1 的目录、TODO、数据字段和四项奖励。教师版正文第 18–22 页给出奖励算法与权重示例。 |
| [第 3 课实验指导（学生版）](https://tencentarena.com/p/edu/course-content/7129/7288/209108?fileId=954926&projectId=36818) · [教师版](https://tencentarena.com/p/edu/course-content/7129/7288/209108?fileId=954927&projectId=36818) | Task2 的九项奖励、小兵编码和边界处理。教师版正文第 14–21 页提供奖励尺度、全局特征和 PPO 调参示例。 |

页码沿用文档正文。课程资料和实验页通过腾讯开悟账号访问。[课程目录](https://tencentarena.com/p/edu/course-content/7129/7288)
