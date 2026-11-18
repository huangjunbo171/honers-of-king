#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

# 任务一简配：占位实现
# 任务一不涉及小兵特征（小兵特征是任务二的考察内容）
# 为保持模型输入维度与满配一致（避免任务一运行报错），本类返回全零小兵特征向量
# 向量长度 = one_unit_feature_num(18) × unit_buff_num(4) × 2（双方） = 144


class SoldierProcess:
    # 每个小兵展开后的特征维度（与 DimConfig.DIM_OF_SOLDIER_* 对齐）
    one_unit_feature_num = 18
    # 每侧最多统计的小兵数量
    unit_buff_num = 4

    def __init__(self, camp):
        self.main_camp = camp

    def process_vec_soldier(self, frame_state):
        # 返回双方小兵的全零特征向量，长度固定为 144
        total_len = self.one_unit_feature_num * self.unit_buff_num * 2
        return [0] * total_len
