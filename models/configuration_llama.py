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

class PCoTLlamaConfig(LlamaConfig):
    model_type = "pcot-llama"

    def __init__(
        self,
        loss_alpha=1.0,
        loss_beta=1.0,
        loss_gamma=1.0,
        use_peft=True,
        lora_r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        use_projection=False,
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

        PEFT Arguments:
            use_peft (`bool`, *optional*, defaults to True):
                Whether to use the PEFT strategy.
            lora_r (`int`, *optional*, defaults to 128):
                The rank of the LoRA matrix.
            lora_alpha (`int`, *optional*, defaults to 32):
                The weight to use for the LoRA loss.
            lora_dropout (`float`, *optional*, defaults to 0.1):
                The dropout to use in the LoRA matrix.
        
        Projection Arguments:
            use_projection (`bool`, *optional*, defaults to False):
                Whether to use the projection head for the ccot process.
        """
        super().__init__(**kwargs)
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.loss_gamma = loss_gamma
        self.use_peft = use_peft
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_projection = use_projection
