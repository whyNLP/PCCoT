from .configuration_llama import PCCoTLlamaConfig
from .modeling_llama import PCCoTLlamaForCausalLM
from .configuration_gpt2 import PCCoTGPT2Config
from .modeling_gpt2 import PCCoTGPT2LMHeadModel
from .pccot_arguments import PCCoTArguments, PCCOT_ARGS_NAME
from .data_processor import COTDataProcessor
from .wandb_callback import CustomWandbCallback
from transformers.integrations import INTEGRATION_TO_CALLBACK

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForTokenClassification

AutoConfig.register("pccot-llama", PCCoTLlamaConfig)
AutoModelForCausalLM.register(PCCoTLlamaConfig, PCCoTLlamaForCausalLM)

AutoConfig.register("pccot-gpt2", PCCoTGPT2Config)
AutoModelForCausalLM.register(PCCoTGPT2Config, PCCoTGPT2LMHeadModel)

# register to transformer callback
INTEGRATION_TO_CALLBACK["wandb"] = CustomWandbCallback
