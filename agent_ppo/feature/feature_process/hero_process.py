#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

HeroProcess 按观察方划分双方英雄，并按 hero_feature_config.ini 的字段顺序编码。
process_vec_hero 为每个阵营输出 67 维状态，供 FeatureProcess 拼接。坐标采用统一阵营朝向。
"""

from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
import configparser
import os
import math


class HeroProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.transform_camp2_to_camp1 = camp in (2, "PLAYERCAMP_2")
        self.get_hero_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.hero_feature_config)
        self.view_dist = 15000
        self.one_unit_feature_num = 67
        self.unit_buff_num = 1

    def get_hero_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "hero_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.hero_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.hero_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_hero(self, frame_state):

        self.generate_hero_info_list(frame_state)

        # Generate hero features for our camp
        # 生成我方阵营的英雄特征
        main_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.main_camp_hero_dict, "main_camp")

        # Generate hero features for enemy camp
        # 生成敌方阵营的英雄特征
        enemy_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.enemy_camp_hero_dict, "enemy_camp")

        return main_camp_hero_vector_feature, enemy_camp_hero_vector_feature

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in sorted(frame_state["hero_states"], key=lambda item: item["config_id"]):
            side = self.main_camp_hero_dict if hero["camp"] == self.main_camp else self.enemy_camp_hero_dict
            side[hero["config_id"]] = hero

    def generate_one_type_hero_feature(self, one_type_hero_info, camp):
        vector_feature = []
        num_heros_considered = 0
        for hero in one_type_hero_info.values():
            if num_heros_considered >= self.unit_buff_num:
                break

            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](hero, value, feature_name)
                # Normalize the specific features
                # 对具体特征进行正则化
                if feature_name not in self.map_feature_to_norm:
                    assert False
                for k in value:
                    value_vec = []
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_heros_considered += 1

        if num_heros_considered < self.unit_buff_num:
            self.no_hero_feature(vector_feature, num_heros_considered)
        return vector_feature

    def no_hero_feature(self, vector_feature, num_heros_considered):
        for _ in range((self.unit_buff_num - num_heros_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def get_from_hero(self, hero, vector_feature, feature_name):
        """
        直接从hero对象获取特征值（如killCnt, deadCnt等）
        此函数已实现，可作为参考
        """
        value = hero[feature_name]
        vector_feature.append(value)

    def get_from_actor_state(self, hero, vector_feature, feature_name):
        vector_feature.append(hero[feature_name])

    def get_from_actor_state_values(self, hero, vector_feature, feature_name):
        vector_feature.append(hero[feature_name])

    def get_hp_rate(self, hero, vector_feature, feature_name):
        current = hero["hp"]
        maximum = hero["max_hp"]
        vector_feature.append(current / maximum if maximum > 0 else 0.0)

    def get_ep_rate(self, hero, vector_feature, feature_name):
        value = 0
        if hero["max_ep"] > 0:
            value = hero["ep"] / hero["max_ep"]
        vector_feature.append(value)

    def get_summon_skill_cd(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_5":
                value = skill_state["cooldown"]
        vector_feature.append(value)

    def get_common_attack_is_useable(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_0":
                value = 1 if skill_state["usable"] else 0
        vector_feature.append(value)

    def get_skill_1_useable(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_1":
                value = 1 if skill_state["usable"] else 0
        vector_feature.append(value)

    def get_skill_2_useable(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_2":
                value = 1 if skill_state["usable"] else 0
        vector_feature.append(value)

    def get_skill_3_useable(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_3":
                value = 1 if skill_state["usable"] else 0
        vector_feature.append(value)

    def get_skill_4_useable(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_4":
                value = 1 if skill_state["usable"] else 0
        vector_feature.append(value)

    def get_hero_skill_1_cd(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_1":
                value = skill_state["cooldown"]
        vector_feature.append(value)

    def get_hero_skill_2_cd(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_2":
                value = skill_state["cooldown"]
        vector_feature.append(value)

    def get_hero_skill_3_cd(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_3":
                value = skill_state["cooldown"]
        vector_feature.append(value)

    def get_hero_skill_4_cd(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_4":
                value = skill_state["cooldown"]
        vector_feature.append(value)

    def get_summon_skill_type(self, hero, vector_feature, feature_name):
        summon_skill_type_map = {
            "80102": 1,
            "80109": 2,
            "80104": 3,
            "80116": 3,
            "80108": 4,
            "80110": 5,
            "80105": 6,
            "80103": 7,
            "80107": 8,
            "80121": 9,
            "80115": 10,
        }
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_5":
                skill_id = skill_state.get("config_id", skill_state.get("configId"))
                value = summon_skill_type_map.get(str(skill_id), 0)
        vector_feature.append(value)

    def get_dist_from_enemy_heros(self, main_hero, vector_feature, feature_name):
        opponents = self.enemy_camp_hero_dict if main_hero["camp"] == self.main_camp else self.main_camp_hero_dict
        opponent = next(iter(opponents.values()), None)
        distance = self.cal_dist(main_hero["location"], opponent["location"]) * 100 if opponent else 0.0
        vector_feature.append(distance)

    def cal_dist(self, pos1, pos2):
        dist = math.sqrt((pos1["x"] / 100.0 - pos2["x"] / 100.0) ** 2 + (pos1["z"] / 100.0 - pos2["z"] / 100.0) ** 2)
        return dist

    def get_hero_level(self, hero, vector_feature, feature_name):
        vector_feature.append(float(hero["level"]))

    def is_alive(self, hero, vector_feature, feature_name):
        vector_feature.append(float(hero["hp"] > 0))

    def get_location_x(self, hero, vector_feature, feature_name):
        value = hero["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = -value
        vector_feature.append(value)

    def get_location_z(self, hero, vector_feature, feature_name):
        value = hero["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = -value
        vector_feature.append(value)
