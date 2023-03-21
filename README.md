# Bloom Embedding - Pytorch
Implemented based on <https://explosion.ai/blog/bloom-embeddings>, claims to reduce parameter usage of LLM by concatenating vectors based on a hash and LUT (similar to how bloom filters work which gives the name).

It also 'solves' OOV problems as any sufficiently sized bloom embedding will allow for *more* tokens to be dynamically added; 
recommend trying setting `vocab=12` and inputting 13,14,etc it will keep working although you do need to be careful for overlapping hashes which aren't garanteed be unique with this usage

Note that on initialization the module validates your configuration exaustively for overlaps (to prevent duplicate tokens mapping to the same exact vectors), to disable set `validate=False`

## Install
```
pip install bloom-embedding
```

## Example

```
from bloom_embedding import BloomEmbed

vocab=4096
bloom = BloomEmbed(128, vocab, 128, 4)
print(f'{bloom.key_usage_rate()=}') # bloom.key_usage_rate()=128.0

bloom(torch.arange(32).unsqueeze(0))
# torch.Size([1, 32, 128], dtype=torch.float) 
```