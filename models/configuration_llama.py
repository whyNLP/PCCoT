# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" LLaMA model configuration"""
from transformers.models.llama.configuration_llama import LlamaConfig

class PCCoTLlamaConfig(LlamaConfig):
    model_type = "pccot-llama"

    def __init__(
        self,
        loss_alpha=1.0,
        loss_beta=1.0,
        loss_gamma=1.0,
        use_layerwise_std=True,
        use_projection=False,
        num_iterations=6,
        **kwargs
    ):
        """
        Args:
        
        Loss Arguments:
            loss_alpha (`float`, *optional*, defaults to 1.0):
                The weight to use for the CoT loss.
            loss_beta (`float`, *optional*, defaults to 1.0):
                The weight to use for the CCoT loss.
            loss_gamma (`float`, *optional*, defaults to 1.0):
                The weight to use for the KD loss.
            use_layerwise_std (`bool`, *optional*, defaults to True):
                Whether to use layerwise standard deviation for the KD loss. Experiments
                show that this does not affect the performance of the model much.
        
        Projection Arguments:
            use_projection (`bool`, *optional*, defaults to False):
                Whether to use the projection head for the ccot process.
        
        CoT Arguments:
            num_iterations (`int`, *optional*, defaults to 6):
                The number of iterations to use in the PCCoT model.
        """
        super().__init__(**kwargs)
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.loss_gamma = loss_gamma
        self.use_layerwise_std = use_layerwise_std
        self.use_projection = use_projection
        self.num_iterations = num_iterations
