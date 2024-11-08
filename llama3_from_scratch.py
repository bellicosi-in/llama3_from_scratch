#llama3 from scratch

from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import matplotlib.pyplot as plt
import os


tokenizer_path = "Llama3.1-8B/tokenizer.model"
special_tokens = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|reserved_special_token_2|>",
    "<|reserved_special_token_3|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_4|>",
    "<|eot_id|>",#end of turn
] + [f"<|reserved_special_token_{i}|>" for i in range(5,256 -5)]

mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
tokenizer = tiktoken.Encoding(name = Path(tokenizer_path).name,
                             pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                             mergeable_ranks = mergeable_ranks,
                             special_tokens = {token: len(mergeable_ranks) + i for i,token in enumerate(special_tokens)},
                             )
print(tokenizer.decode(tokenizer.encode("hello_world!")))

model = torch.load("Llama3.1-8B/consolidated.00.pth",weights_only = True, map_location = torch.device('cpu'))


# print(json.dumps(list(model.keys())[:20],indent = 4))

# loading the config
model_llama = "Llama3.1-8B"
params_json = "params.json"
with open(os.path.join(model_llama,params_json),'r') as f:
    config = json.load(f)
# print(config)

# inferring details about the model :
dim = config["dim"]                             #4096
n_layers = config["n_layers"]                   #32
n_heads = config["n_heads"]                     #32
n_kv_heads = config["n_kv_heads"]               #8
vocab_size = config["vocab_size"]               #128256
multiple_of = config["multiple_of"]             #1024
ffn_dim_multiplier = config["ffn_dim_multiplier"] #1.3
norm_eps = config["norm_eps"]                   #1e-05
rope_theta = config["rope_theta"]               #500000.0

# converting text to tokens
prompt = "this tokyo night theme seems decent. but the neovim one seems better"
tokens = [128000] + tokenizer.encode(prompt)
# print(tokens)
tokens = torch.tensor(tokens)
prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
# print(prompt_split_as_tokens)


# converting tokens into embeddings
embedding_layer = torch.nn.Embedding(vocab_size,dim)    #vocab size = 128256, dim = 4096 = 128256 * 4096
embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)
# print(token_embeddings_unnormalized.shape)

# normalizing the embedding
def rms_norm(tensor, norm_weights):
    return(tensor*torch.sqrt(tensor.pow(2).mean(-1,keepdim = True) + norm_eps)) * norm_weights

# # BUILDING THE FIRST LAYER OF THE TRANSFORMER
# normaliztion
token_embeddings = rms_norm(token_embeddings_unnormalized, model["layers.0.attention_norm.weight"])
print(token_embeddings.shape)

# attention
# for layer 0: attention.wq.weight = 4096 x 4096, attention.wk.weight.shape = 1024 x 4096. attention.wv.weight.shape = 1024 x 4096, attention.wo.weight.shape = 4096 x 4096


# unwrapping the queries from multiple attention headds, the resulting shape is [32x128x4096]
# 32 is the number of attention heads in llama3, 128 is the size of he query vector, 4096 is the tok embeds.
q_layer0 = model["layers.0.attention.wq.weight"]
print(q_layer0.shape, n_heads)
head_dim = q_layer0.shape[0]//n_heads
q_layer0 = q_layer0.view(n_heads, head_dim, dim) #32 x 128 x 4096
# print(q_layer0.shape)

q_layer0_head0 = q_layer0[0]

# multiply the query weights with tok embds to receive a query for the token
q_per_token = torch.matmul(token_embeddings, q_layer0_head0.T)
print(q_per_token.shape)