"""用合成帧验证 Task1 的特征、奖励、LSTM 和 PPO 公式。运行测试只需 NumPy 和 PyTorch。"""

import math
import unittest

import torch

from agent_ppo.conf.conf import Config
from agent_ppo.feature.feature_process import FeatureProcess
from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
from agent_ppo.feature.reward_process import GameRewardManager
from agent_ppo.model.model import Model


def make_hero(camp, x):
    fields = (
        "hp_recover ep max_ep ep_recover phy_atk mgc_atk phy_def mgc_def kill_cnt dead_cnt "
        "money_cnt mov_spd attack_range atk_spd phy_armor_hurt mgc_armor_hurt crit_rate crit_effe "
        "phy_vamp mgc_vamp cd_reduce ctrl_reduce exp money revive_time kill_income"
    )
    hero = dict.fromkeys(fields.split(), 0)
    hero.update(
        camp=camp, config_id=112, runtime_id=camp * 101, level=3, hp=5000, max_hp=5000,
        location={"x": x, "z": x}, skill_state={"slot_states": []},
    )
    return hero


def make_organ(camp, sub_type, x):
    return dict(camp=camp, sub_type=sub_type, hp=100, max_hp=100, phy_atk=300,
                attack_range=6000, kill_income=100, location={"x": x, "z": x})


def make_frame():
    return {
        "frame_no": 0,
        "hero_states": [make_hero(1, -12000), make_hero(2, 12000)],
        "npc_states": [make_organ(2, 21, 12000), make_organ(1, 24, -24000),
                       make_organ(1, 21, -12000), make_organ(2, 24, 24000)],
    }


class FeatureRewardTests(unittest.TestCase):
    def test_normalization(self):
        normalizer = FeatureNormalizer()
        self.assertEqual(normalizer.one_hot(2, [3, 1, 2], "eq"), [0, 0, 1])
        self.assertEqual([normalizer.min_max(v, 0, 10) for v in (-1, 5, 11)], [0, 0.5, 1])

    def test_layout_camp_swap_and_missing_units(self):
        frame = make_frame()
        first = FeatureProcess(1)
        features = first.process_feature({"frame_state": frame})
        self.assertEqual(len(features), 350)
        self.assertEqual(features, FeatureProcess(2).process_feature({"frame_state": frame}))
        self.assertEqual(features[134:278], [0] * 144)
        self.assertEqual(features[278:284], [1, 1, 0, 0, 0, 1])  # 己方水晶
        self.assertEqual(features[296:302], [1, 1, 1, 0, 0, 0])  # 己方塔
        self.assertEqual(features[:5], [1, 0, 0, 1, 0])
        self.assertEqual(features[17], 1)
        self.assertAlmostEqual(features[31], 0.4)

        frame["hero_states"].pop(0)
        frame["npc_states"].pop(1)
        sparse = first.process_feature({"frame_state": frame})
        self.assertEqual(len(sparse), 350)
        self.assertEqual(sparse[:67], [0] * 67)
        self.assertEqual(sparse[278:296], [0] * 18)
        self.assertEqual(sparse[296:302], features[296:302])
        self.assertEqual(sparse[296 + 10], 0)  # 己方英雄距离槽
        self.assertGreater(sparse[296 + 11], 0)

        hero = frame["hero_states"][0]
        hero.update(hp=0, max_hp=0, location={"x": 100000, "z": 0})
        sparse = FeatureProcess(2).process_feature({"frame_state": frame})
        self.assertEqual((sparse[0], sparse[17], sparse[31]), (0, 0, 1))

    def test_reward_deltas_and_tower_disappearance(self):
        frame = make_frame()
        own, enemy = GameRewardManager(101), GameRewardManager(202)
        for manager in (own, enemy):
            self.assertEqual(manager.result(frame)["reward_sum"], 0)
        frame["frame_no"] += 1
        frame["npc_states"][0]["hp"] = 80
        frame["hero_states"][0]["kill_cnt"] += 1
        frame["hero_states"][1]["dead_cnt"] += 1
        reward = own.result(frame)
        self.assertAlmostEqual(reward["tower_hp_point"], 0.2)
        self.assertEqual((reward["kill"], reward["death"]), (1, -1))
        self.assertAlmostEqual(reward["reward_sum"], 2.6)
        self.assertAlmostEqual(enemy.result(frame)["reward_sum"], -2.6)
        frame["frame_no"] += 1
        self.assertEqual(own.result(frame)["reward_sum"], 0)
        frame["frame_no"] += 1
        frame["npc_states"].pop(0)
        self.assertAlmostEqual(own.result(frame)["reward_sum"], 4)
        frame["frame_no"] += 1
        self.assertEqual(own.result(frame)["reward_sum"], 0)

    def test_forward_is_per_frame_and_health_gated(self):
        frame = make_frame()
        own = GameRewardManager(101)
        hero = frame["hero_states"][0]
        hero["location"] = {"x": -24000, "z": -24000}
        for frame_no in (0, 1):
            frame["frame_no"] = frame_no
            reward = own.result(frame)
            self.assertAlmostEqual(reward["forward"], -0.5)
            self.assertAlmostEqual(reward["reward_sum"], -0.005)
        hero["hp"] = 2000
        frame["frame_no"] = 2
        self.assertEqual(own.result(frame)["forward"], 0)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.model = Model()

    @torch.no_grad()
    def test_recurrent_batch_matches_sixteen_steps(self):
        model = self.model
        steps, batch = Config.LSTM_TIME_STEPS, 2
        features = torch.randn(batch, steps, 350)
        h0, c0 = torch.randn(batch, 512), torch.randn(batch, 512)
        model.set_train_mode()
        heads = model([features.flatten(0, 1), h0, c0])
        self.assertEqual([t.shape[1] for t in heads], [12, 16, 16, 16, 16, 9, 1])
        expected = torch.cat(heads, dim=1).reshape(batch, steps, 86)
        expected_h, expected_c = model.lstm_hidden_output, model.lstm_cell_output

        model.set_eval_mode()
        h, c = h0, c0
        for i in range(steps):
            logits, value, cell, hidden = model([features[:, i], h, c], inference=True)
            torch.testing.assert_close(torch.cat([logits, value], dim=1), expected[:, i])
            h, c = hidden.squeeze(0), cell.squeeze(0)
        torch.testing.assert_close(hidden, expected_h)
        torch.testing.assert_close(cell, expected_c)
        self.assertFalse(torch.equal(h, h0))

    def test_clipped_loss_and_legal_action_gradients(self):
        sizes = Config.LABEL_SIZE_LIST
        # 首分支两项合法，前两帧所选动作的新旧概率比均为 2，优势分别为 +1 和 -1。
        # 第三帧死亡，第四帧无合法动作。其余分支全部使用占位标签。
        n = 4
        seri = torch.zeros(n, 350 + sum(sizes))
        seri[:3, 350:352] = 1
        labels = [torch.tensor([0, 0, -1, -1])] + [torch.full((n,), -1)] * 5
        old = [torch.zeros(n, width) for width in sizes]
        old[0][:, :2] = torch.tensor([0.25, 0.75])
        weights = [torch.ones(n)] + [torch.zeros(n)] * 5
        reward = torch.tensor([1., 1., 0., 0.])
        advantage = torch.tensor([1., -1., 99., 99.])
        flags = torch.tensor([1., 1., 0., 1.])
        data = [seri, reward, advantage, *labels, *old, *weights, flags, None, None]
        logits = [torch.zeros(n, width) for width in sizes]
        logits[0][:, 2:] = 100  # 非法动作的高分应被掩蔽
        heads = [t.requires_grad_() for t in logits]
        value = torch.zeros(n, 1, requires_grad=True)
        loss, (_, (value_cost, policy_cost, entropy_cost)) = self.model.compute_loss(data, heads + [value])
        self.assertAlmostEqual(value_cost.item(), 0.25)
        self.assertAlmostEqual(policy_cost.item(), 0.4)
        self.assertAlmostEqual(entropy_cost.item(), -math.log(2))
        self.assertAlmostEqual(loss.item(), 0.65 - Config.BETA_START * math.log(2))
        policy_grad, value_grad = torch.autograd.grad(loss, [heads[0], value])
        self.assertGreater(policy_grad[1, 0].item(), 0)
        self.assertEqual(policy_grad[:, 2:].abs().sum().item(), 0)
        self.assertEqual(policy_grad[2:].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(value_grad).all())


if __name__ == "__main__":
    unittest.main()
