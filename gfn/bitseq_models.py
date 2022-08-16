"""
Copied from https://github.com/Tikquuss/GflowNets_Tutorial/blob/main/4_Sequence_Generation.ipynb
"""

import torch
import torch.nn as nn


def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(
        *(
            sum(
                (
                    [nn.Linear(i, o)] + ([act] if n < len(l) - 2 else [])
                    for n, (i, o) in enumerate(zip(l, l[1:]))
                ),
                [],
            )
            + tail
        )
    )


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def get_padding_masks(slen, lengths):
    """
    Generate hidden states mask
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]
    # sanity check
    assert mask.size() == (bs, slen)
    return mask


class TransformerModel(nn.Module):

    # params : n_words, eos_index, pad_index, emb_dim

    def __init__(self, params, transformer_layers):
        """
        Transformer model
        """
        super().__init__()
        # embeddings : one hot is better in this case
        self.embeddings = Embedding(
            params.n_words, params.emb_dim, padding_idx=params.pad_index
        )
        # This can be replace by transformer model from torch.nn, huggingface transoformer ...
        self.transformer = transformer_layers

    def forward(self, x, lengths):
        """
        Inputs:
            `x` LongTensor(bs, slen), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
        """
        # padding_mask = x != self.pad_index
        # lengths = padding_mask.long().sum(dim=1).to(x.device)

        # check inputs
        bs, slen = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen

        # generate masks
        mask = get_padding_masks(slen, lengths)

        # embeddings
        tensor = self.embeddings(x)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # transformer layers
        tensor = self.transformer(tensor)

        return tensor
