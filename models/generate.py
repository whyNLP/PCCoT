import torch
from torch import nn
from transformers.generation.utils import GenerationMixin, LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from transformers.generation.configuration_utils import GenerationMode


class PCCoTGenerationMixin(GenerationMixin):
        
    def _sample_step(
        self,
        input_ids: torch.LongTensor,
        next_token_logits: torch.Tensor,
        unfinished_sequences: torch.Tensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        **model_kwargs,
    ) -> torch.LongTensor:
        # init values
        pad_token_id = generation_config._pad_token_tensor
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        return next_tokens[:, None]


    @torch.no_grad()
    def generate(self, collated, generation_config=None, **kwargs):
        """
        Generate answers using the PCCoT model.
        """
        # constants
        inputs_tensor = collated["input_ids"]
        input_ids_length = inputs_tensor.shape[1]

        # Prepare generation config
        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, **kwargs
        )

        # 6. Prepare `max_length` depending on other stopping criteria.
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name="input_ids",
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # setup the padding token
        if generation_config and generation_config.pad_token_id is None:
            generation_config.pad_token_id = generation_config.eos_token_id

        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=self.device)

        # prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=StoppingCriteriaList(), **kwargs
        )

        # validate generation mode
        generation_mode = generation_config.get_generation_mode()
        if generation_mode not in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            raise ValueError(
                f"Unsupported generation mode: {generation_mode}. "
                "Only SAMPLE and GREEDY_SEARCH are supported for PCCoT generation."
            )

        # Move to the same device
        for k, v in collated.items():
            if isinstance(v, torch.Tensor):
                collated[k] = v.to(self.device)

        unfinished_sequences = torch.ones(
            collated["input_ids"].shape[0], dtype=torch.long, device=self.device
        )

        # Parallel Continuous CoT
        outputs = self.forward(**collated, return_dict=True)
        attention_mask = collated["attention_mask"]

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[0].clone()[:, -1, :].float()
        next_token_logits = next_token_logits.to(self.device)

        # pre-process distribution
        next_tokens = self._sample_step(
            input_ids=collated["input_ids"],
            next_token_logits=next_token_logits,
            unfinished_sequences=unfinished_sequences,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
        )

        input_ids = torch.cat([collated["input_ids"], next_tokens], dim=1)

        # keep track of which sequences are already finished
        new_len = 1
        this_peer_finished = False

        while not this_peer_finished and new_len + input_ids_length < generation_config.max_length:

            # prepare the inputs for the next token
            attention_mask = torch.cat([attention_mask, attention_mask[:, -1:]], dim=1)
            position_ids = attention_mask.cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)[:, -1:]

            # use the original model to get the next token
            outputs = super(self.__class__, self).forward(
                input_ids=next_tokens,
                past_key_values=outputs.past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1, :].float()
            next_token_logits = next_token_logits.to(self.device)

            # pre-process distribution
            next_tokens = self._sample_step(
                input_ids=input_ids,
                next_token_logits=next_token_logits,
                unfinished_sequences=unfinished_sequences,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
            )

            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            unfinished_sequences = unfinished_sequences & ~prepared_stopping_criteria(input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0
            new_len += 1

        return input_ids
