import math
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel
from .configuration_llama import PCCoTLlamaConfig
from .generate import PCCoTGenerationMixin


class PCCoTLlamaForCausalLM(LlamaForCausalLM, PCCoTGenerationMixin):
    
    config_class = PCCoTLlamaConfig

    def __init__(self, config: PCCoTLlamaConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.use_projection:
            hidden_dim = config.hidden_size
            self.prj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.prj = lambda x: x

        self._log_cache: Dict[str, Any] = {}

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        key_indices = None,
        cot_input_ids = None,
        cot_labels = None,
        cot_attention_mask = None,
        cot_kd_indices = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ## Part 1. teacher CoT

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=cot_input_ids,
            attention_mask=cot_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        cot_loss = None
        if cot_labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = cot_labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            cot_loss = loss_fct(shift_logits, shift_labels)

        ## Part 2. student CoT
            
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        question_boundary, latent_boundary, ccot_kd_index = key_indices
        ccot_outputs = self.model(
            input_ids=input_ids[:, :latent_boundary],
            attention_mask=attention_mask[:, :latent_boundary],
            past_key_values=DynamicCache(),
        )
        last_hidden_state = ccot_outputs[0][:, question_boundary-1:latent_boundary-1]
        latent_input_embeds = self.prj(last_hidden_state).to(dtype=last_hidden_state.dtype)
        question_past_key_values = DynamicCache()
        question_past_key_values.key_cache = ccot_outputs.past_key_values.key_cache[:]
        question_past_key_values.value_cache = ccot_outputs.past_key_values.value_cache[:]
        question_past_key_values.crop(question_boundary)

        # iteratively predict the latent tokens
        for _ in range(self.config.num_iterations):
            # manually duplicate the past_key_values
            ccot_past_key_values = DynamicCache()
            ccot_past_key_values.key_cache = question_past_key_values.key_cache[:]
            ccot_past_key_values.value_cache = question_past_key_values.value_cache[:]
            # update the ccot_outputs
            ccot_outputs = self.model(inputs_embeds=latent_input_embeds, past_key_values=ccot_past_key_values)
            # get the last hidden state
            last_hidden_state = ccot_outputs[0]
            # project the hidden state
            projected_hidden_state = self.prj(last_hidden_state).to(dtype=last_hidden_state.dtype)
            # get the new input_ids
            latent_input_embeds = torch.cat([latent_input_embeds[:, :1], projected_hidden_state[:, :-1]], dim=1)
        
        # predict the answer
        answer_outputs = self.model(
            input_ids=input_ids[:, latent_boundary:],
            attention_mask=attention_mask,
            past_key_values=ccot_outputs.past_key_values,
            output_hidden_states=True,
        )

        # get the logits
        answer_hidden_states = answer_outputs[0]
        answer_logits = self.lm_head(answer_hidden_states)
        answer_logits = answer_logits.float()

        # calculate the loss
        ccot_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = answer_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ccot_loss = loss_fct(shift_logits, shift_labels)

        ## Part 3. knowledge distillation
        teacher_hidden_states = torch.stack(outputs.hidden_states, dim=1).detach()[:, 1:] # (batch_size, num_layers, seq_len, hidden_size)
        teacher_hidden_states = teacher_hidden_states.gather(2, cot_kd_indices[:, None, None, None].expand(-1, self.config.num_hidden_layers, -1, self.config.hidden_size))
        student_hidden_states = torch.stack(answer_outputs.hidden_states, dim=1)[:, 1:, ccot_kd_index:ccot_kd_index+1]

        if self.config.use_layerwise_std:
            # transpose to (num_layers, batch_size, hidden_size)
            teacher_hidden_states = teacher_hidden_states.transpose(0, 1).squeeze(2)
            student_hidden_states = student_hidden_states.transpose(0, 1).squeeze(2)

            # calculate the loss
            kd_loss = F.smooth_l1_loss(student_hidden_states, teacher_hidden_states, reduction="none").reshape(self.config.num_hidden_layers, -1) # (num_layers, batch_size * hidden_size)
            kd_loss /= teacher_hidden_states.reshape(self.config.num_hidden_layers, -1).std(dim=-1, keepdim=True)
            kd_loss = kd_loss.mean()
        else:
            # calculate the loss
            kd_loss = F.smooth_l1_loss(student_hidden_states, teacher_hidden_states)
            kd_loss /= teacher_hidden_states.std()

        loss = cot_loss * self.config.loss_alpha + ccot_loss * self.config.loss_beta + kd_loss * self.config.loss_gamma

        # log cache
        self._log_cache["cot_loss"] = cot_loss.item()
        self._log_cache["ccot_loss"] = ccot_loss.item()
        self._log_cache["kd_loss"] = kd_loss.item()
        self._log_cache["teacher_std"] = teacher_hidden_states.std().item()

        logits_tuple = (answer_logits, logits)

        if not return_dict:
            output = (logits_tuple,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits_tuple,
            past_key_values=answer_outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
