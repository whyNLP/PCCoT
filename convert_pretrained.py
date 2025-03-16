from transformers import AutoModelForCausalLM, AutoTokenizer
import models

def main():
    model = models.PCoTLlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T")
    model.save_pretrained("outputs/TinyLlama-2.5T-pcot")

if __name__ == "__main__":
    main()
