import torch
from transformers import LlamaTokenizer
from transformers import BertTokenizer
from model import Edward
from config import modelConfig

tokenizer = LlamaTokenizer.from_pretrained("./my_tokenizer")

model_config = modelConfig(vocab_size=tokenizer.vocab_size)
model = Edward(model_config)

try:
    checkpoint = torch.load("checkpoint.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.load_state_dict(torch.load("weights.pth"))
    print("✅ model loaded sucessfully")
except:
    raise ValueError(f"❗weights not found.")

def generate_sequence(query, max_tokens_to_generate, temperature=0.8, top_k=40, repetition_penalty=1.2):
    #query = f"<bos><human>{query} <robot>"
    query = f"<bos>{query}"
    
    input_ids = tokenizer(query, return_tensors="pt", add_special_tokens=False)["input_ids"]

    if input_ids.shape[1] > model_config.max_token_len:
        return "Input is too long for the model."

    generated_ids = input_ids.clone()
    past_tokens = set()

    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens_to_generate):
            outputs = model(generated_ids)
            logits = outputs[:, -1, :]  # shape: (1, vocab_size)

            # Apply repetition penalty
            for token_id in set(generated_ids[0].tolist()):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

            # Apply temperature scaling
            logits = logits / temperature

            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
            filtered_logits = torch.full_like(logits, float('-inf'))
            filtered_logits.scatter_(-1, top_k_indices, top_k_logits)

            probabilities = torch.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(probabilities, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Remove the prompt part
    prompt_len = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    response = generated_text[prompt_len:].strip()

    return response.replace("<nl>", "\n")




while True:
    query = input("-> ")
    if query.lower() == "exit":
        print("-> model stopped. ")
        break
    print("-> " + generate_sequence(query, max_tokens_to_generate=100))
