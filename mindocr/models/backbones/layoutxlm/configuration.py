# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# This code is adapted from https://github.com/PaddlePaddle/PaddleNLP
# with modifications to run on MindSpore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass


@dataclass
class LayoutXLMPretrainedConfig:
    def __init__(self, use_visual_backbone=True, use_float16=False):
        pretrained_config = {
            "use_visual_backbone": use_visual_backbone,
            "use_float16": use_float16,
            "attention_probs_dropout_prob": 0.1,
            "use_visual_backbone": use_visual_backbone,
            "use_float16": use_float16,
            "bos_token_id": 0,
            "coordinate_size": 128,
            "eos_token_id": 2,
            "fast_qkv": False,
            "gradient_checkpointing": False,
            "has_relative_attention_bias": False,
            "has_spatial_attention_bias": False,
            "has_visual_segment_embedding": True,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "image_feature_pool_shape": [7, 7, 256],
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-05,
            "max_2d_position_embeddings": 1024,
            "max_position_embeddings": 514,
            "max_rel_2d_pos": 256,
            "max_rel_pos": 128,
            "model_type": "layoutxlm",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "output_past": True,
            "pad_token_id": 1,
            "shape_size": 128,
            "rel_2d_pos_bins": 64,
            "rel_pos_bins": 32,
            "type_vocab_size": 1,
            "vocab_size": 250002,
        }

        if use_visual_backbone is False:
            pretrained_config["attention_probs_dropout_prob"] = 0
            pretrained_config["hidden_dropout_prob"] = 0

        for key, value in pretrained_config.items():
            setattr(self, key, value)
