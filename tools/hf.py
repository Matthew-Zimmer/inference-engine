import sys, time, torch
from transformers import AutoTokenizer, AutoModel

warmup = 10

def mean_pooling(model_output, n):
    token_embeddings = model_output[0]
    return torch.sum(token_embeddings, 1) / n

sentences = [sys.argv[1]]

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, safe_serialization=True)
model.eval()

for _ in range(warmup):
    encoded_input = tokenizer(sentences, return_tensors='pt')

tokenizer_start_time = time.time()
encoded_input = tokenizer(sentences, return_tensors='pt')
tokenizer_end_time = time.time()

tokens = [str(int(x)) for y in encoded_input["input_ids"] for x in y]
print(",".join(tokens))
print(f"{int((tokenizer_end_time - tokenizer_start_time)*1e6)}")

for _ in range(warmup):
    with torch.no_grad():
        model_output = model(**encoded_input)

model_start_time = time.time()
with torch.no_grad():
    model_output = model(**encoded_input)


embeddings = mean_pooling(model_output, len(tokens))
model_end_time = time.time()
floats = [str(float(x)) for y in embeddings for x in y]
print(",".join(floats))

print(f"{int((model_end_time - model_start_time)*1e6)}")

