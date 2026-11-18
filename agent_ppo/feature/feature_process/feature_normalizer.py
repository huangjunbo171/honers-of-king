#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

英雄和机关编码器通过 parse_config 读取 INI 中的函数和参数。
one_hot 按候选值顺序输出指示量，min_max 将连续属性截断并映射到单位区间。
"""


class FeatureNormalizer:
    def one_hot(self, value, value_list, compare_method):
        if compare_method != "eq":
            raise ValueError("Unsupported compare method: " + compare_method)
        return [int(value == candidate) for candidate in value_list]

    def min_max(self, value, min_value, max_value):
        if value <= min_value:
            return 0.0
        if value >= max_value:
            return 1.0
        return (value - min_value) / (max_value - min_value)

    def parse_config(self, config_list):
        config_dict = {}
        for config_str in config_list:
            parts = config_str.split(":")
            feature_name = parts[0]
            method = parts[1]
            if method == "one_hot":
                values = list(map(int, parts[2:-1]))
                compare_method = parts[-1]
                config_dict[feature_name] = (self.one_hot, values, compare_method)
            elif method == "min_max":
                min_value = int(parts[2])
                max_value = int(parts[3])
                config_dict[feature_name] = (self.min_max, min_value, max_value)
            else:
                raise ValueError("Unsupported method: " + method)
        return config_dict
