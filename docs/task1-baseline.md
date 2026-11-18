# 4.2 构建智能体（Task1）

当前代码入口是 [agent_ppo](../agent_ppo/)。手册主要讲解特征和奖励。V63.0.9 代码包还要求补齐敌方英雄编码、LSTM 和 PPO 损失，本仓库的[模型实现](../agent_ppo/model/model.py)包含这些部分。课程运行页显示 `V63.0.9`，教师版的四项基础奖励说明同时指向 `63.0.9` 和 `63.0.10`。[学生版 §2.1、§3.1–3.3](https://tencentarena.com/p/edu/course-content/7129/7288/209107?fileId=954918&projectId=36818) · [教师版 §3.2](https://tencentarena.com/p/edu/course-content/7129/7288/209107?fileId=954919&projectId=36818)

## 任务一实现位置

以下路径以 `agent_ppo/` 为起点。

| 文件 | 课程任务与数据 |
| --- | --- |
| `feature/feature_process/feature_normalizer.py` | 1.1 `one_hot(value, value_list, compare_method)` 按 `value_list` 的顺序编码，比较方法为 `eq`。1.2 `min_max(value, min_value, max_value)` 将范围映射到 `[0, 1]`，先检查下界，再检查上界。 |
| `feature/feature_process/hero_process.py` | 2.1 遍历 `frame_state["hero_states"]`，按 `camp` 分类，使用 `config_id` 作为键并排序。2.2–2.3 对照 `get_from_actor_state` 和 `get_from_actor_state_values` 的调用读取 `hero[feature_name]`。2.4–2.6 分别提取血量比、`float(level)`、`float(hp > 0)`。2.7–2.8 提取 `location["x"]` 和 `location["z"]` 并转换阵营坐标。 |
| `feature/feature_process/organ_process.py` | 3.1 遍历 `npc_states`，按阵营分别保存 `tower`（`sub_type == 21`）和 `crystal`（`sub_type == 24`）。3.2–3.7 返回当前 HP、最大 HP、血量比、存活标志、己方阵营标志、`sub_type`。3.8–3.9 转换机关的 X/Z 坐标。 |
| `feature/feature_process/soldier_process.py` | 基础实验的小兵输入使用全零占位。Task2 提供小兵特征 TODO。 |
| `feature/reward_process.py` | 在 `set_cur_calc_frame_vec` 中填写四项当前帧原始值。在 `get_reward` 中计算当前帧奖励。 |
| `conf/conf.py` | 调整 `GameConfig.REWARD_WEIGHT_DICT`。 |
| `conf/train_env_conf.toml`、`agent.py`、`feature/definition.py` | 分别用于核对训练环境配置、智能体入口和数据定义。 |

### 特征计算示例

阵营转换由观察者所属的阵营决定。开启 `transform_camp2_to_camp1` 时，英雄和建筑的有效 X/Z 坐标取反。保留无效坐标哨兵 `100000`。血量比例在 `max_hp <= 0` 时取 `0.0`。字段读到后按任务包指定顺序追加到特征向量。[教师版 §3.1，第 4–17 页](https://tencentarena.com/p/edu/course-content/7129/7288/209107?fileId=954919&projectId=36818)

```python
def one_hot(value, candidates, compare_method="eq"):
    if compare_method != "eq":
        raise ValueError(f"Unsupported compare method {compare_method!r}")
    return [float(value == candidate) for candidate in candidates]


def min_max(value, lower, upper):
    if value <= lower:
        return 0.0
    if value >= upper:
        return 1.0
    return (value - lower) / (upper - lower)


def hp_rate(entity):
    return entity["hp"] / entity["max_hp"] if entity["max_hp"] > 0 else 0.0


def camp_coordinate(value, transform_camp2_to_camp1):
    if transform_camp2_to_camp1 and value != 100000:
        return -value
    return value


def group_heroes(hero_states, main_camp):
    own, enemy = {}, {}
    for hero in hero_states:
        side = own if hero["camp"] == main_camp else enemy
        side[hero["config_id"]] = hero
    return dict(sorted(own.items())), dict(sorted(enemy.items()))


def group_organs(npc_states, main_camp):
    own, enemy = {}, {}
    names = {21: "tower", 24: "crystal"}
    for npc in npc_states:
        name = names.get(npc["sub_type"])
        if name is not None:
            side = own if npc["camp"] == main_camp else enemy
            side[name] = npc
    return own, enemy
```

## 四项奖励

设 `delta_own = own_now - own_previous`，`delta_enemy = enemy_now - enemy_previous`。零和增量为 `delta_own - delta_enemy`。最终标量是各项奖励乘对应权重后的和。帧间状态的初始化和更新沿用任务包的生命周期。

| 项 | 当前帧原始值 | 合成方式 |
| --- | --- | --- |
| `forward` | `calculate_forward(main_hero, main_tower, enemy_tower)` | 当前帧绝对值。保留任务包的空间与边界逻辑。 |
| `tower_hp_point` | 本阵营防御塔 `hp / max_hp`，防御塔缺失或 `max_hp <= 0` 时为 0 | 零和增量。敌塔掉血时该项为正。 |
| `kill` | 英雄累计 `kill_cnt` | 零和增量。我方增加 1 次击杀时，该项为 +1。 |
| `death` | 英雄累计 `dead_cnt` | 零和增量。我方增加 1 次死亡时，该项为 +1，乘负权重后形成惩罚。 |

```python
def zero_sum_delta(own_now, own_previous, enemy_now, enemy_previous):
    return (own_now - own_previous) - (enemy_now - enemy_previous)


def task1_terms(own_now, own_previous, enemy_now, enemy_previous):
    terms = {"forward": own_now["forward"]}
    for name in ("tower_hp_point", "kill", "death"):
        terms[name] = zero_sum_delta(
            own_now[name], own_previous[name], enemy_now[name], enemy_previous[name]
        )
    return terms


def weighted_reward(terms, weights):
    return sum(value * weights[name] for name, value in terms.items())
```

| 配置 | `forward` | `tower_hp_point` | `kill` | `death` |
| --- | ---: | ---: | ---: | ---: |
| 学生版初值 | 0.0 | 1.0 | 0.0 | 0.0 |
| 教师版原值 | 0.01 | 5.0 | -0.6 | -1.0 |
| 仓库实现 | 0.01 | 5.0 | +0.6 | -1.0 |

教师版文字把击杀描述为正奖励。按同一手册的 `kill_cnt` 和零和增量公式代入，我方击杀对应的 `kill` 加权项为 `1 × -0.6 = -0.6`。同一帧若对手死亡计数也加 1，`death` 项为 `-1 × -1.0 = +1.0`，两项合计 `+0.4`。仓库按手册的文字目标把击杀权重设为 +0.6。一次己方击杀伴随敌方死亡时，两项合计 +1.6。[学生版 §3.2.2](https://tencentarena.com/p/edu/course-content/7129/7288/209107?fileId=954918&projectId=36818) · [教师版 §3.2.1–3.2.2](https://tencentarena.com/p/edu/course-content/7129/7288/209107?fileId=954919&projectId=36818)

## 启动训练前核对

- `one_hot(2, [1, 2, 3])` 得到 `[0.0, 1.0, 0.0]`。区间 `[0, 10000]` 中，5000 映射为 0.5，两端分别为 0 和 1。
- 两个阵营使用相同的特征顺序和长度。坐标 500 在开启转换时变为 -500，哨兵 100000 保持原值。
- 敌塔血量比从 1.0 变成 0.8，己塔保持不变时，推塔项约为 +0.2，乘 5.0 后约为 +1.0。
- 我方击杀且对手死亡时，检查 `kill == +1`、`death == -1`。静止帧的计数增量为 0，`forward` 保留该帧值。新一局的首帧按任务包定义重置上一帧状态。
