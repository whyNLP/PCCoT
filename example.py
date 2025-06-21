import models
from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
from transformers.utils.hub import cached_file
from peft import AutoPeftModel

# Example model name
model_name_or_path = "whyNLP/pccot-gpt2"
# Example question
question = "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?"


# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
config = AutoConfig.from_pretrained(model_name_or_path)
model = AutoPeftModel.from_pretrained(model_name_or_path)

# we have to override the model config after loading the model, peft does not provide interface to
# load base model with custom config with AutoPeftModel.
model.get_base_model().config = config

# Load the PCCoT arguments
pccot_args_file = cached_file(model_name_or_path, models.PCCOT_ARGS_NAME)
parser = HfArgumentParser(models.PCCoTArguments)
(pccot_args, ) = parser.parse_json_file(json_file=pccot_args_file)

# Load the data processor
data_processor = models.COTDataProcessor(
    tokenizer=tokenizer,
    pccot_args=pccot_args,
)
collated = data_processor.process(question)

# generation
decoded_tokens = model.generate(
    collated=collated,
    max_new_tokens=10,
    do_sample=False,
)

decoded_tokens = decoded_tokens[:, collated["input_ids"].shape[1]:]  # remove the input_ids part
answers = tokenizer.batch_decode(decoded_tokens, skip_special_tokens=True)

# Print the answer
print("Question:", question)
print("Answer:", answers[0])
