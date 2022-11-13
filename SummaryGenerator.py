import math
from random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset.AmazonPre as A

from Config import config

device = config.device

"""
预训练的Transformer
"""


class GneratorModel(nn.Module):
    def __init__(self):
        super(GneratorModel, self).__init__()
        self.Embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.posEmb = PositionalEncoding(config.emb_dim)
        self.transformer = nn.Transformer(config.emb_dim, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(config.emb_dim * (config.max_sumLen + 2), config.vocab_size),
            nn.Dropout(),
            nn.Linear(config.vocab_size, config.vocab_size)
        )

    def forward(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).bool().to(device)

        srcPaddingMask = GneratorModel.getPaddingMask(src).to(device)
        tgtPaddingMask = GneratorModel.getPaddingMask(tgt).to(device)

        src = self.posEmb(self.Embedding(src))
        tgt = self.posEmb(self.Embedding(tgt))

        output = self.transformer(src, tgt, tgt_mask=tgt_mask,
                                  src_key_padding_mask=srcPaddingMask, tgt_key_padding_mask=tgtPaddingMask)
        output.transpose(0, 1)
        output = output.reshape(-1, config.emb_dim)
        output = self.fc(output)
        softmax_output = F.softmax(output, dim=2)
        return softmax_output

    @staticmethod
    def getPaddingMask(tokens):
        """
        用于padding_mask
        """
        paddingMask = torch.zeros(tokens.size())
        paddingMask[tokens == 2] = -torch.inf
        return paddingMask.to(device)

    def generate_summary(self, test_src, test_tgt):
        """
        每次只能预测一个句子
        :param test_src:
        :param test_tgt:
        :return:
        """
        for i in range(config.max_sumLen):
            # 进行transformer计算
            out = model(test_src, test_tgt)

            # 找出最大值的index
            predWord = torch.argmax(out, dim=1)
            # 和之前的预测结果拼接到一起
            test_tgt = torch.concat([test_tgt, predWord.unsqueeze(0)], dim=1)

            # 如果为<eos>，说明预测结束，跳出循环
            if predWord == 1:
                break
        print(test_tgt)


"""
PositionEncoding
"""


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout()

        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x.to(device)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == '__main__':
    data_iter = A.get_gan_iter()

    model = GneratorModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        train_acc, train_sum, batch_num, loss_sum = 0, 0, 0, 0.0
        for inputs, sumX, sumY in data_iter:
            inputs, sumX, sumY = inputs.to(device), sumX.to(device), sumY.to(device)
            outputs = model(inputs, sumX)

            loss = criterion(outputs.contiguous().reshape(-1, outputs.size(-1)),
                             sumY.contiguous().view(-1)) / config.valid_rev_token_num
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            batch_num += 1

        if epoch % 5 == 0:
            test_sum = 0
            seed = [randint(0, batch_num - 1) for _ in range(batch_num // 10 * 3)]
            print(seed)
            with torch.no_grad():
                for idx, (inputs, sumX, sumY) in enumerate(data_iter):
                    if idx in seed:
                        inputs, sumX, sumY = inputs.to(device), sumX.to(device), sumY.to(device)
                        outputs = model(inputs, sumX)

                        loss = criterion(outputs.contiguous().reshape(-1, outputs.size(-1)),
                                         sumY.contiguous().view(-1)) / config.valid_rev_token_num

                        test_sum += loss.sum().item()

            print(loss_sum, test_sum)

    testSrcList = ["the wine is very bad don t buy it"]
    testTgtList = [[0]]

    tokenSrcList = A.getTokens(testSrcList)
    A.addPadding(tokenSrcList, config.max_seqLen)
    testSrcList = torch.tensor(tokenSrcList).to(device)
    testTgtList = torch.tensor(testTgtList).to(device)

    model.generate_summary(testSrcList, testTgtList)
