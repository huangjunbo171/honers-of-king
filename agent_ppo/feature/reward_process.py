#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

GameRewardManager 由 Agent.reset 按局创建。result 将逐帧英雄计数和塔血量转换成双方增量差，
叠加当帧位置项，返回各项和按 GameConfig 加权的 reward_sum。RewardStruct 保存相邻帧原始值。
"""


import math
from agent_ppo.conf.conf import GameConfig


# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.enemy_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1

    def result(self, frame_data):
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)
        self.m_last_frame_no = frame_data["frame_no"]

        frame_no = frame_data["frame_no"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):

        hero = next((unit for unit in frame_data["hero_states"] if unit["camp"] == camp), None)
        towers = {unit["camp"]: unit for unit in frame_data["npc_states"] if unit["sub_type"] == 21}
        tower = towers.get(camp)
        enemy_tower = next((unit for side, unit in towers.items() if side != camp), None)

        for name, item in cul_calc_frame_map.items():
            item.last_frame_value = item.cur_frame_value
            if name == "forward":
                item.cur_frame_value = self.calculate_forward(hero, tower, enemy_tower)
            elif name == "tower_hp_point":
                item.cur_frame_value = tower["hp"] / tower["max_hp"] if tower and tower["max_hp"] > 0 else 0.0
            elif hero is not None and name == "kill":
                item.cur_frame_value = hero["kill_cnt"]
            elif hero is not None and name == "death":
                item.cur_frame_value = hero["dead_cnt"]

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        if not main_hero or not main_tower or not enemy_tower or main_hero["max_hp"] <= 0:
            return 0.0
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (
            main_hero["location"]["x"],
            main_hero["location"]["z"],
        )
        forward_value = 0
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        if main_hero["hp"] / main_hero["max_hp"] > 0.99 and dist_hero2emy > dist_main2emy > 0:
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
        return forward_value

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = self.main_hero_camp, self.enemy_hero_camp

        for hero in frame_data["hero_states"]:
            if hero["runtime_id"] == self.main_hero_player_id:
                main_camp = hero["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["camp"]
                self.enemy_hero_camp = enemy_camp
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum = 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            own = self.m_main_calc_frame_map[reward_name]
            enemy = self.m_enemy_calc_frame_map[reward_name]
            if reward_name == "forward":
                reward_struct.value = own.cur_frame_value
            elif self.m_last_frame_no == -1:
                reward_struct.value = 0.0
            else:
                own_delta = own.cur_frame_value - own.last_frame_value
                enemy_delta = enemy.cur_frame_value - enemy.last_frame_value
                reward_struct.value = own_delta - enemy_delta

            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
        reward_dict["reward_sum"] = reward_sum
