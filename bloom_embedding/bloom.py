from collections import defaultdict
from functools import reduce
import torch
import torch.jit
import numpy as np
from typing import List


def mueller_hash(k):
    k = ((k >> 16) ^ k) * 0x45D9F3B
    k = ((k >> 16) ^ k) * 0x45D9F3B
    k = (k >> 16) ^ k
    return k


def salted_mueller_hash(k, salt: int):
    salt = ((salt >> 16) ^ salt) * 0x45D9F3B
    salt = ((salt >> 16) ^ salt) * 0x45D9F3B
    k ^= (salt >> 16) ^ salt

    return mueller_hash(k)


@torch.jit.script
def embed_tensor(x: torch.IntTensor, lut: torch.Tensor, digests: int):
    lut_size, key_dim = lut.shape
    batch_size = x.shape[0]
    output: List[torch.Tensor] = []
    for n in range(digests):
        idxs = salted_mueller_hash(x, n) % lut_size
        key_embeds = torch.index_select(lut, 0, idxs.view(-1)).view(batch_size, -1, key_dim)
        output.append(key_embeds)

    return torch.concat(output, dim=-1)


class BloomEmbed(torch.nn.Module):
    def __init__(self, lut_size: int, vocab_size: int, embed_dim: int, digests=2, validate=True):
        super().__init__()
        assert embed_dim % digests == 0, "Embed_dim should be dividable by digests"
        self.key_dim = embed_dim//digests
        self.lut = torch.nn.parameter.Parameter(torch.normal(0., 1/np.sqrt(embed_dim), (lut_size, self.key_dim)))
        self.vocab_size = vocab_size
        self.lut_size = lut_size
        self.digests = digests
        if validate:
            self.validate_overlap(self.vocab_size)

    def key_usage_rate(self):
        usages = defaultdict(int)
        for n in range(self.vocab_size):
            for k in range(self.digests):
                idx = salted_mueller_hash(n, k) % self.lut_size
                usages[idx] += 1

        return np.mean(list(usages.values()))

    def validate_overlap(self, up_to: int):
        checked = set()
        for n in range(up_to):
            idxs = tuple(salted_mueller_hash(n, k) % self.lut_size for k in range(self.digests))
             # XOR Hash, to make it a set comparison; comment to also use ordering
            idxs = reduce(lambda c,p: p^mueller_hash(c), idxs, 0)
            assert idxs not in checked, f"Overlap found at token_id={n}"
            checked.add(idxs)

    def forward(self, x: torch.IntTensor) -> torch.Tensor:
        return embed_tensor(x, self.lut, self.digests)


if __name__ == '__main__':
    vocab = 2 ** 12
    print(f'{vocab=}')
    bloom = BloomEmbed(128, vocab, 128, 4)
    print(f'{bloom.key_usage_rate()=}')
    v = bloom(torch.arange(32).unsqueeze(0))
    print(v.shape, v)