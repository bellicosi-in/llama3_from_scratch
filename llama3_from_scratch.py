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
# print(tokenizer.decode(tokenizer.encode("hello_world!")))

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
# print(token_embeddings.shape) #[18,4096]

# attention
# for layer 0: attention.wq.weight = 4096 x 4096, attention.wk.weight.shape = 1024 x 4096. attention.wv.weight.shape = 1024 x 4096, attention.wo.weight.shape = 4096 x 4096


# unwrapping the queries from multiple attention headds, the resulting shape is [32x128x4096]
# 32 is the number of attention heads in llama3, 128 is the size of he query vector, 4096 is the tok embeds.
q_layer0 = model["layers.0.attention.wq.weight"] #[4096,4096]
# print(q_layer0.shape, n_heads)
head_dim = q_layer0.shape[0]//n_heads #[128]
q_layer0 = q_layer0.view(n_heads, head_dim, dim) #32 x 128 x 4096
# print(q_layer0.shape)

# q_layer0_head0 = q_layer0[0]

# # multiply the query weights with tok embds to receive a query for the token
# q_per_token = torch.matmul(token_embeddings, q_layer0_head0.T) #[18,128]
# # print(q_per_token.shape)

# # rotary positional embeddings
# # converting each vector into complex pairs
# q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2) #[18,64,2]

# using dot product of complex numbers ot rotate a vector
zero_to_one_split_into_64_parts = torch.tensor(range(64))/64
# freqs to rotate a vector
freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
freqs_for_each_token = torch.outer(torch.arange(18), freqs)
# generating complex exponential values based onb the phase shifts
freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token) #[18,64]
# print(freqs_cis)

# q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs) #[18,64]
# # element wise multiplication in complex space.applying the rotation
# q_per_token_as_complex_numbers_rotated = q_per_token_as_complex_numbers * freqs_cis #[18,64]
# q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers_rotated) #[18,64,2]
# q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape) #[18,128]


# # keys
k_layer0 = model["layers.0.attention.wk.weight"]
k_layer0 = k_layer0.view(n_kv_heads, k_layer0.shape[0]//n_kv_heads, dim)
# k_layer0_head0 = k_layer0[0]
# k_per_token = torch.matmul(token_embeddings, k_layer0_head0.T)
# k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0],-1,2)
# k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
# k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
# k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
# # print(k_per_token_rotated.shape)


# # multiplying the query and key values
# qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(head_dim)**0.5

# heatmap
def display_qk_heatmap(qk_per_token):
    _, ax = plt.subplots()
    im = ax.imshow(qk_per_token.to(float).detach(), cmap = "viridis")
    ax.set_xticks(range(len(prompt_split_as_tokens)))
    ax.set_yticks(range(len(prompt_split_as_tokens)))
    ax.set_xticklabels(prompt_split_as_tokens)
    ax.set_yticklabels(prompt_split_as_tokens)
    ax.figure.colorbar(im, ax=ax)


# display_qk_heatmap(qk_per_token)


# # masking the qk values
# mask = torch.full((len(tokens), len(tokens)), float("-inf"), device = tokens.device)
# mask = torch.triu(mask, diagonal = 1)
# # print(mask)
# qk_per_token_after_masking = qk_per_token + mask
# display_qk_heatmap(qk_per_token_after_masking)
# qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)

# # values
v_layer0 = model["layers.0.attention.wv.weight"]
v_layer0 = v_layer0.view(n_kv_heads, v_layer0.shape[0]//n_kv_heads, dim)
# v_layer0_head0 = v_layer0[0]

# v_per_token = torch.matmul(token_embeddings, v_layer0_head0.T)
# qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
# print(qkv_attention.shape)

# CONVERTING THE ABOVE INTO A FUNCTION FOR EVERY HEAD OF THE FIRST LAYER

qkv_attention_store = []

for head in range(n_heads):
    q_layer0_head = q_layer0[head]
    k_layer0_head = k_layer0[head//4] # key weights are shared across 4 heads
    v_layer0_head = v_layer0[head//4] # value weights are shared across 4 heads
    q_per_token = torch.matmul(token_embeddings, q_layer0_head.T)
    k_per_token = torch.matmul(token_embeddings, k_layer0_head.T)
    v_per_token = torch.matmul(token_embeddings, v_layer0_head.T)

    q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
    q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
    q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
    q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)

    k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
    k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
    k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
    k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)

    qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
    mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
    mask = torch.triu(mask, diagonal=1)
    qk_per_token_after_masking = qk_per_token + mask
    qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
    qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
    qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
    qkv_attention_store.append(qkv_attention)

# print(len(qkv_attention_store))


stacked_qkv_attention = torch.cat(qkv_attention_store, dim = -1)
# print(stacked_qkv_attention.shape)
w_layer0 = model["layers.0.attention.wo.weight"]

embedding_delta = torch.matmul(stacked_qkv_attention, w_layer0.T)
embedding_after_edit = token_embeddings_unnormalized + embedding_delta
# print(embedding_after_edit.shape)

embedding_after_edit_normalized = rms_norm(embedding_after_edit, model["layers.0.ffn_norm.weight"])

w1 = model["layers.0.feed_forward.w1.weight"]
w2 = model["layers.0.feed_forward.w2.weight"]
w3 = model["layers.0.feed_forward.w3.weight"]

output_after_feedforward = torch.matmul(torch.Functional.F.silu(torch.matmul(embedding_after_edit_normalized,w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T),w2.T)
layer_0_embedding = embedding_after_edit + output_after_feedforward
 