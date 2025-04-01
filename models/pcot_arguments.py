from dataclasses import dataclass, field
from pathlib import Path
import json

@dataclass
class PCoTArguments:
    """
    Arguments pertaining to the discourse transformer model.
    """
    # Tokenizer arguments
    bot_token_id: int = field(
        default=None,
        metadata={"help": "The pcot-special id of the beginning of sentence token. Please do not set this argument unless you understand the consequences."}
    )
    eot_token_id: int = field(
        default=None,
        metadata={"help": "The pcot-special id of the end of sentence token. Please do not set this argument unless you understand the consequences."}
    )
    latent_token_id: int = field(
        default=None,
        metadata={"help": "The pcot-special id of the sink token. Please do not set this argument unless you understand the consequences."}
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
        metadata={"help": "The number of latent tokens to use in the PCoT model."}
    )
    num_iterations: int = field(
        default=6,
        metadata={"help": "The number of iterations to use in the PCoT model."}
    )

    
    def save(self, output_dir: Path):
        """ Save the arguments to a json file """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "pcot_args.json", "w") as f:
            json.dump(vars(self), f, indent=4)
