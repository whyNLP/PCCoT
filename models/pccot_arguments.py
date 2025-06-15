from dataclasses import dataclass, field
from pathlib import Path
import json


PCCOT_ARGS_NAME = "pccot_args.json"

@dataclass
class PCCoTArguments:
    """
    Arguments pertaining to the discourse transformer model.
    """
    # Tokenizer arguments
    bot_token_id: int = field(
        default=None,
        metadata={"help": "The pccot-special id of the beginning of sentence token. Please do not set this argument unless you understand the consequences."}
    )
    eot_token_id: int = field(
        default=None,
        metadata={"help": "The pccot-special id of the end of sentence token. Please do not set this argument unless you understand the consequences."}
    )
    latent_token_id: int = field(
        default=None,
        metadata={"help": "The pccot-special id of the sink token. Please do not set this argument unless you understand the consequences."}
    )
    label_pad_token_id: int = field(
        default=-100,
        metadata={"help": "The id of the padding token for labels. Labels with this id will not be taken into account in the loss computation. Please do not set this argument unless you understand the consequences."}
    )

    # CoT specific arguments
    answer_prompt: str = field(
        default="The answer is:",
        metadata={"help": "The prompt to identify the answer in the input. The last token of the prompt will be used to do knowledge distillation."}
    )
    num_latent_tokens: int = field(
        default=6,
        metadata={"help": "The number of latent tokens to use in the PCCoT model."}
    )

    # PEFT arguments
    use_peft: bool = field(
        default=True,
        metadata={"help": "Whether to use the PEFT strategy."}
    )
    lora_r: int = field(
        default=128,
        metadata={"help": "The rank of the LoRA matrix."}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "The weight to use for the LoRA loss."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout to use in the LoRA matrix."}
    )
    lora_target_modules: str = field(
        default="q_proj-k_proj-v_proj-o_proj-down_proj-up_proj-gate_proj",
        metadata={"help": "hyphen separated list of target modules to apply LoRA layers to"},
    )
    lora_modules_to_save: str = field(
        default="",
        metadata={"help": "hyphen separated list of modules apart from adapter layers to be set as trainable and saved in the final checkpoint."},
    )

    def save(self, output_dir: Path):
        """ Save the arguments to a json file """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / PCCOT_ARGS_NAME, "w") as f:
            json.dump(vars(self), f, indent=4)
