#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

OrganProcess 从 npc_states 识别水晶和防御塔，按观察方、对方和建筑类型排列槽位。
process_vec_organ 按 organ_feature_config.ini 编码血量、位置和双方英雄距离，每个槽位 18 维。
"""

from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
import configparser
import os
import math


class OrganProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp

        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.main_camp_organ_dict = {}
        self.enemy_camp_organ_dict = {}

        self.transform_camp2_to_camp1 = camp in (2, "PLAYERCAMP_2")
        self.get_organ_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.organ_feature_config)
        self.view_dist = 15000
        self.one_unit_feature_num = 18
        self.unit_buff_num = 2

    def get_organ_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "organ_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.organ_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.organ_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_organ(self, frame_state):
        self.generate_organ_info_dict(frame_state)
        self.generate_hero_info_list(frame_state)

        local_vector_feature = []

        # Generate features for our team's towers
        # 生成我方阵营的防御塔特征
        main_camp_organ_vector_feature = self.generate_one_type_organ_feature(self.main_camp_organ_dict, "main_camp")
        local_vector_feature.extend(main_camp_organ_vector_feature)

        # Generate features for enemy team's towers
        # 生成敌方阵营的防御塔特征
        enemy_camp_organ_vector_feature = self.generate_one_type_organ_feature(self.enemy_camp_organ_dict, "enemy_camp")
        local_vector_feature.extend(enemy_camp_organ_vector_feature)

        vector_feature = local_vector_feature
        return vector_feature

    def generate_hero_info_list(self, frame_state):
        self.main_hero_info = None
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["config_id"]] = hero

    def generate_organ_info_dict(self, frame_state):
        self.main_camp_organ_dict.clear()
        self.enemy_camp_organ_dict.clear()
        names = {21: "tower", 24: "crystal"}
        for organ in frame_state["npc_states"]:
            name = names.get(organ["sub_type"])
            if name is not None:
                side = self.main_camp_organ_dict if organ["camp"] == self.main_camp else self.enemy_camp_organ_dict
                side[name] = organ

    def generate_one_type_organ_feature(self, one_type_organ_info, camp):
        vector_feature = []
        for name in ("crystal", "tower"):
            organ = one_type_organ_info.get(name)
            if organ is None:
                vector_feature.extend([0.0] * self.one_unit_feature_num)
                continue
            for feature_name, feature_func in self.feature_func_map.items():
                values = []
                feature_func(organ, values)
                normalize, *params = self.map_feature_to_norm[feature_name]
                for value in values:
                    normalized = normalize(value, *params)
                    if isinstance(normalized, list):
                        vector_feature.extend(normalized)
                    else:
                        vector_feature.append(normalized)
        return vector_feature

    def get_hp(self, organ, vector_feature):
        vector_feature.append(organ["hp"])

    def get_max_hp(self, organ, vector_feature):
        vector_feature.append(organ["max_hp"])

    def get_hp_rate(self, organ, vector_feature):
        current = organ["hp"]
        maximum = organ["max_hp"]
        vector_feature.append(current / maximum if maximum > 0 else 0.0)

    def get_atk(self, organ, vector_feature):
        value = organ["phy_atk"]
        vector_feature.append(value)

    def get_atk_range(self, organ, vector_feature):
        value = organ["attack_range"]
        vector_feature.append(value)

    def get_kill_income(self, organ, vector_feature):
        value = organ["kill_income"]
        vector_feature.append(value)

    def judge_in_main_hero_sight(self, organ, vector_feature):
        value = 0.0
        if self.main_hero_info and self.judge_in_view(self.main_hero_info["location"], organ["location"]):
            value = 1.0
        vector_feature.append(value)

    def judge_in_view(self, main_hero_location, obj_location):
        if (
            (main_hero_location["x"] - obj_location["x"] >= 0 - self.view_dist)
            and (main_hero_location["x"] - obj_location["x"] <= self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] >= 0 - self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] <= self.view_dist)
        ):
            return True
        return False

    def cal_dist(self, pos1, pos2):
        dist = math.sqrt((pos1["x"] / 100.0 - pos2["x"] / 100.0) ** 2 + (pos1["z"] / 100.0 - pos2["z"] / 100.0) ** 2)
        return dist

    def is_alive(self, organ, vector_feature):
        vector_feature.append(float(organ["hp"] > 0))

    def belong_to_main_camp(self, organ, vector_feature):
        vector_feature.append(float(organ["camp"] == self.main_camp))

    def get_organ_type(self, organ, vector_feature):
        vector_feature.append(organ["sub_type"])

    def generate_distance_with_each_hero_feature(self, organ, vector_feature):
        for heroes in (self.main_camp_hero_dict, self.enemy_camp_hero_dict):
            hero = next(iter(heroes.values()), None)
            distance = self.cal_dist(hero["location"], organ["location"]) * 100 if hero else 0.0
            vector_feature.append(distance)

    def get_normal_organ_location_x(self, organ, vector_feature):
        value = organ["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = -value
        vector_feature.append(value)

    def get_normal_organ_location_z(self, organ, vector_feature):
        value = organ["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = -value
        vector_feature.append(value)

    def relative_location_x(self, organ, vector_feature):
        if self.main_hero_info is None:
            vector_feature.append(0.0)
            return
        organ_location_x = organ["location"]["x"]
        location_x = self.main_hero_info["location"]["x"]
        x_diff = organ_location_x - location_x
        if self.transform_camp2_to_camp1 and organ_location_x != 100000:
            x_diff = -x_diff
        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def relative_location_z(self, organ, vector_feature):
        if self.main_hero_info is None:
            vector_feature.append(0.0)
            return
        organ_location_z = organ["location"]["z"]
        location_z = self.main_hero_info["location"]["z"]
        z_diff = organ_location_z - location_z
        if self.transform_camp2_to_camp1 and organ_location_z != 100000:
            z_diff = -z_diff
        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)
