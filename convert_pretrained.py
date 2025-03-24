from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import models

def main():
    model = models.PCoTLlamaForCausalLM.from_pretrained("JackFram/llama-160m")

    # check the embeddings to avoid zero lines
    embedding = model.model.get_input_embeddings()
    for i in range(embedding.weight.size(0)):
        line = embedding.weight[i]
        if torch.allclose(line, torch.zeros_like(line)):
            print("Found zero line", i)
            embedding.weight.data[i] = torch.randn_like(line) * 1e-5

    model.save_pretrained("outputs/JackFram--llama-160m-pcot")

if __name__ == "__main__":
    main()
