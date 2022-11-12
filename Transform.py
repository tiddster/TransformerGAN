import math

import torch
import torch.nn as nn
import dataset.Amazonpre as A

from Config import config


class PreTransform(nn.Module):
    def __init__(self):
        super(PreTransform, self).__init__()
        self.posEmb = PositionalEncoding(config.emb_dim)
        self.Embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.transformer = nn.Transformer()

    def forward(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])

        srcPaddingMask = PreTransform.getPaddingMask(src)
        tgtPaddingMask = PreTransform.getPaddingMask(tgt)

        src = self.posEmb(self.Embedding(src)).transpose(0,1)
        tgt = self.posEmb(self.Embedding(tgt)).transpose(0,1)

        output = self.transformer(src, tgt, tgt_mask=tgt_mask,
                                  src_key_padding_mask=srcPaddingMask, tgt_key_padding_mask=tgtPaddingMask)
        output.transpose(0,1)
        print(output.shape)

    @staticmethod
    def getPaddingMask(tokens):
        """
        用于padding_mask
        """
        paddingMask = torch.zeros(tokens.size())
        paddingMask[tokens == 0] = -torch.inf
        return paddingMask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


p = PreTransform()

data_iter = A.get_iter(False)
for (tokens, rating) in data_iter:
    p(tokens, tokens)
