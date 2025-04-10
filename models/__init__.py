from .configuration_llama import PCoTLlamaConfig
from .modeling_llama import PCoTLlamaForCausalLM
from .configuration_gpt2 import PCoTGPT2Config
from .modeling_gpt2 import PCoTGPT2LMHeadModel
from .pcot_arguments import PCoTArguments
from .data_processor import COTDataProcessor
from .wandb_callback import CustomWandbCallback
from transformers.integrations import INTEGRATION_TO_CALLBACK

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForTokenClassification

AutoConfig.register("pcot-llama", PCoTLlamaConfig)
AutoModelForCausalLM.register(PCoTLlamaConfig, PCoTLlamaForCausalLM)

AutoConfig.register("pcot-gpt2", PCoTGPT2Config)
AutoModelForCausalLM.register(PCoTGPT2Config, PCoTGPT2LMHeadModel)

# register to transformer callback
INTEGRATION_TO_CALLBACK["wandb"] = CustomWandbCallback
