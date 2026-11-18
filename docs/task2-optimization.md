# 4.3 策略优化（Task2）

在 Task1 可训练且可评估的基础上，依次处理五项新增奖励、小兵特征、可选全局特征。先核对当前任务包的 TODO、帧字段和输入配置。课程任务包的奖励项和模型输入宽度由包内实现确定。[第 3 课学生版](https://tencentarena.com/p/edu/course-content/7129/7288/209108?fileId=954926&projectId=36818) · [第 3 课教师版](https://tencentarena.com/p/edu/course-content/7129/7288/209108?fileId=954927&projectId=36818)

## 奖励从四项扩展至九项

沿用 [Task1](task1-baseline.md) 的 `forward`、`tower_hp_point`、`kill`、`death`。在 `feature/reward_process.py` 按「提取当前帧原始值 → 计算当帧奖励 → 按权重求和」处理新增项。

| 项 | 原始值 | 合成方式与边界 |
| --- | --- | --- |
| `money` | `hero["money_cnt"]` | 双方金币增量之差。先核对字段表达累计收入还是当前余额。 |
| `hp_point` | `(hp / max_hp) ** 0.25` | 分别计算双方增量，再取差。某一方上一帧的 HP 比为 0 时，该方本帧增量取 0。活到死这一帧仍有掉血增量。四次方根提高低血量区域的单位 HP 变化幅度。 |
| `ep_rate` | 存活且 `max_ep > 0` 时为 `ep / max_ep`，其余为 0 | 只取本方增量，上一帧原始值为 0 时，本帧增量取 0。 |
| `exp` | `calculate_exp_sum(hero)` 返回累计经验 | 通过此前各级经验需求之和加当前等级内经验，衔接升级时的经验进度重置。满级以后将对应阵营经验增量置零，再算双方差。 |
| `last_hit` | 遍历 `frame_data["frame_action"]["dead_action"]`，匹配 `death["sub_type"] == 11` | 用 `killer["runtime_id"]` 对照双方英雄。同一帧本方补刀 +1，对方补刀 -1。直接使用该帧的差值。 |

教师版对满级经验提供两种口径。文字按 15 级处理，参考代码把当前累计经验 `15000` 作为演示阈值。以任务包的等级和累计经验配置判定满级，并用一帧升级样本确认满级门槛当帧的经验处理。计算 HP 比时可给零分母和非正 HP 一个 `0.0` 返回值。[教师版正文第 2–7 页](https://tencentarena.com/p/edu/course-content/7129/7288/209108?fileId=954927&projectId=36818)

```python
def shaped_hp(hp, max_hp):
    return (hp / max_hp) ** 0.25 if hp > 0 and max_hp > 0 else 0.0


def energy_ratio(hp, ep, max_ep):
    return ep / max_ep if hp > 0 and max_ep > 0 else 0.0


def delta_from_positive(current, previous):
    return current - previous if previous > 0 else 0.0


def last_hit_delta(frame_action, own_runtime_id, enemy_runtime_id):
    sides = {own_runtime_id: 1, enemy_runtime_id: -1}
    return sum(
        sides.get(event["killer"]["runtime_id"], 0)
        for event in frame_action.get("dead_action", ())
        if event["death"]["sub_type"] == 11
    )


def advanced_terms(
    own_now, own_previous, enemy_now, enemy_previous, own_capped, enemy_capped,
    frame_action, own_runtime_id, enemy_runtime_id,
):
    own_exp = 0.0 if own_capped else own_now["exp"] - own_previous["exp"]
    enemy_exp = 0.0 if enemy_capped else enemy_now["exp"] - enemy_previous["exp"]
    return {
        "money": (own_now["money"] - own_previous["money"])
        - (enemy_now["money"] - enemy_previous["money"]),
        "hp_point": delta_from_positive(own_now["hp_point"], own_previous["hp_point"])
        - delta_from_positive(enemy_now["hp_point"], enemy_previous["hp_point"]),
        "ep_rate": delta_from_positive(own_now["ep_rate"], own_previous["ep_rate"]),
        "exp": own_exp - enemy_exp,
        "last_hit": last_hit_delta(frame_action, own_runtime_id, enemy_runtime_id),
    }
```

示例中的 `own_now`、`enemy_now` 和上一帧输入都包含原始奖励值。`own_capped`、`enemy_capped` 由实际经验规则提供。奖励代码接入时核对死亡事件字段和缺省行为。

## 小兵特征

在 `agent_ppo/feature/feature_process/soldier_process.py` 实现 `generate_soldier_info_dict` 及 `get_hp`、`get_max_hp`、`get_hp_rate`、`get_atk`、`get_kill_income`、`get_soldier_type`、`get_location_x`、`get_location_z`。

- 从 `frame_state["npc_states"]` 筛选 `sub_type == 11` 且 `hp > 0` 的实体。按 `camp` 分组，再按 `runtime_id` 排序。讲义和手册的部分例子使用符号名 `ACTOR_SUB_SOLDIER`，教师版小兵参考实现用整数 `11`。核对包内枚举。
- 使用 `hp`、`max_hp`、`kill_income`。物理攻击示例给出 `soldier["values"]["phy_atk"]` 与 `soldier["phy_atk"]` 两种访问路径，按实际帧格式选择。X/Z 沿用 Task1 的阵营镜像和哨兵处理。
- 在任务包的每个小兵分组里，按既有特征顺序抽取最多 `unit_buff_num = 4` 个槽位。调用 `no_soldier_feature` 为剩余槽位填零。手册示例每个小兵 18 维，视距常量为 15000。沿用包内视野判断、归一化配置和模型尺寸。[教师版正文第 7–14 页](https://tencentarena.com/p/edu/course-content/7129/7288/209108?fileId=954927&projectId=36818)

```python
MELEE_IDS = {6023, 6024, 6088, 6089, 6320, 6321}
RANGED_IDS = {6002, 6005, 6014, 6015, 6372, 6375, 6085, 6086}


def soldier_type(config_id):
    if config_id in MELEE_IDS:
        return 1
    if config_id in RANGED_IDS:
        return 2
    return 3


def living_soldiers(npc_states, main_camp):
    soldiers = [
        npc for npc in npc_states
        if npc["sub_type"] == 11 and npc["hp"] > 0
    ]
    soldiers.sort(key=lambda npc: npc["runtime_id"])
    own = [npc for npc in soldiers if npc["camp"] == main_camp]
    enemy = [npc for npc in soldiers if npc["camp"] != main_camp]
    return own, enemy
```

## 全局特征扩展

在任务包提供的观测字段上构造以下示例。新增维度时，同步核对特征拼接顺序、模型输入层和 checkpoint 兼容性。

| 特征 | 手册中的计算示例 |
| --- | --- |
| 时间 | `min(frameNo / 9000.0, 1.0)`。按帧号 `< 3000`、`< 6000`、其余三段编码。9000 来自示例假设的 15 帧/秒和 10 分钟上限，按实验设置校准。 |
| 经济差 | 把 `己方经济 - 敌方经济` 从 `[-5000, 5000]` 映射到 `[0, 1]`，超限截断，0.5 表示均势。全局示例用 `money` 键，奖励章节用 `money_cnt` 键，接入时核对观测字段。 |
| 等级差 | 同法把 `己方等级 - 敌方等级` 从 `[-5, 5]` 映射到 `[0, 1]`。 |
| 双方存活小兵 | 分别统计观测中存活的小兵数，用 `min(count / 10.0, 1.0)` 压缩。统计口径随游戏观测的视野覆盖变化。 |

[教师版正文第 15–18 页](https://tencentarena.com/p/edu/course-content/7129/7288/209108?fileId=954927&projectId=36818)

## 权重与训练曲线

先量化各项的典型单帧变化，再比较「变化量 × 权重」和每局累计贡献。例如，金币增加 100 乘 0.006 得 0.6，敌塔血量比下降 0.1 乘 5.0 得 0.5。讲义建议优先维持推塔目标，再调战斗、经济和前进项。每次对比只改一个因素，固定训练预算及评估口径。用固定对手胜率和回放核实高奖励对应的行为。

| 现象 | 优先检查 |
| --- | --- |
| 奖励上升，固定对手胜率持平 | 看前进、经济和推塔的分项贡献，并从回放检查是否持续对关键目标推进。 |
| 策略熵迅速下跌，动作重复 | 查看熵系数、动作掩码和奖励是否鼓励单一动作。 |
| 价值损失持续波动 | 检查奖励尺度、个别异常帧、学习率和回报分布。 |
| 指标突然恶化 | 对照同一配置较早的 checkpoint，检查学习率和 PPO 裁剪范围。 |

教师版建议从以下范围开始搜索。学习率 `1e-4–3e-4`、折扣因子 `0.995–0.999`、PPO clip `0.1–0.2`。熵系数示例从 `0.001` 到 `0.01`。这些是课程调参示意值，实际实验由任务包默认值、每决策步采样间隔和固定口径的实测结果确定。[教师版正文第 14–15、19–21 页](https://tencentarena.com/p/edu/course-content/7129/7288/209108?fileId=954927&projectId=36818)

## 最小验证样本

- 己方 HP 从正数降为 0 时记掉血，连续死亡帧及从 0 复活的首帧使该方 HP 增量为 0。
- 最大 EP 为 0、死亡以及上一帧 EP 比为 0 时覆盖对应分支。敌方 EP 变化与己方 EP 奖励独立。
- 模拟一次升级和一次满级后的经验变化，确认累计经验连续、满级之后的对应阵营增量为 0。
- 同帧包含双方各一次小兵补刀时净值为 0。只有己方补刀时为 +1。死亡事件列表为空时为 0。
- 分别给出零个、少于四个、四个和超过四个小兵，检查排序、每组输出宽度、填零和可见性标志。检查远近兵配置 ID、坐标镜像及 `100000` 哨兵。
