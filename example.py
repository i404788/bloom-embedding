import torch
from bloom_embedding import BloomEmbed

vocab = 2 ** 12
print(f'{vocab=}')
bloom = BloomEmbed(128, vocab, 128, 4)
print(f'{bloom.key_usage_rate()=}')
v = bloom(torch.arange(32).unsqueeze(0))
print(v.shape, v)
