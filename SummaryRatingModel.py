import time
from random import randint

import torch.nn as nn
import torch
import torch.nn.functional as F

import dataset.AmazonPre as A

from Config import config

device = config.device


class SRModel(nn.Module):
    def __init__(self):
        super(SRModel, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.LSTM = nn.LSTM(config.emb_dim, config.emb_dim, batch_first=True)
        self.MTH = nn.MultiheadAttention(config.emb_dim, num_heads=4, batch_first=True)

        self.MTHList = nn.ModuleList([nn.MultiheadAttention(config.emb_dim, num_heads=4, batch_first=True) for _ in range(4)])
        self.LinearList = nn.ModuleList([nn.Linear(config.emb_dim, config.emb_dim) for _ in range(4)])
        self.layerNorm = nn.LayerNorm(config.emb_dim)

        self.fc = nn.Linear(config.emb_dim * (config.max_sumLen + 2), 3)

    def forward(self, input):
        paddingMask = SRModel.getPaddingMask(input)

        emb_output = self.embedding(input)
        # lstm_output, (_, _) = self.LSTM(emb_output)

        for mthModel, ffModel in zip(self.MTHList, self.LinearList):
            emb_output, _ = mthModel(emb_output, emb_output, emb_output, key_padding_mask=paddingMask)
            res = emb_output
            emb_output = ffModel(emb_output)
            emb_output = self.layerNorm(res + emb_output)

        mth_output = emb_output.reshape(input.shape[0], -1)
        lin_output = self.fc(mth_output)

        return F.softmax(lin_output, dim=1)

    @staticmethod
    def getPaddingMask(tokens):
        """
        用于padding_mask
        """
        paddingMask = torch.zeros(tokens.size())
        paddingMask[tokens == 2] = 1
        return paddingMask.bool().to(device)


# def train_SRM():
#     srm = SRModel().to(device)
#     data_iter = A.get_summary_iter()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(srm.parameters(), lr=0.001)
#
#     for epoch in range(10):
#         train_acc, train_sum, batch_num, loss_sum = 0, 0, 0, 0.0
#         for inputs, ratings in data_iter:
#             inputs, ratings = inputs.to(device), ratings.to(device)
#             outputs = srm(inputs)
#
#             loss = criterion(outputs, ratings) / config.valid_token_num
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             loss_sum += loss.item()
#             train_sum += ratings.shape[0]
#             train_acc += (outputs.argmax(dim=1) == ratings).sum().item()
#
#             batch_num += 1
#
#         if epoch % 5 == 0:
#             test_acc, test_sum = 0, 0
#             seed = [randint(0, batch_num - 1) for _ in range(batch_num//10*3)]
#             with torch.no_grad():
#                 for idx, (inputs, ratings) in enumerate(data_iter):
#                     if idx in seed:
#                         inputs, ratings = inputs.to(device), ratings.to(device)
#                         outputs = srm(inputs)
#
#                         test_sum += ratings.shape[0]
#                         test_acc += (outputs.argmax(dim=1) == ratings).sum().item()
#
#             print(loss_sum / batch_num, train_acc / train_sum, test_acc / test_sum)

if __name__ == '__main__':
    srm = SRModel().to(device)
    data_iter, test_iter = A.get_summary_iter()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(srm.parameters(), lr=0.00003)

    for epoch in range(8):
        train_acc, train_sum, batch_num, loss_sum, start_time = 0, 0, 0, 0.0, time.time()
        for inputs, ratings in data_iter:
            inputs, ratings = inputs.to(device), ratings.to(device)
            outputs = srm(inputs)

            loss = criterion(outputs, ratings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            train_sum += ratings.shape[0]
            train_acc += (outputs.argmax(dim=1) == ratings).sum().item()

            batch_num += 1

        if epoch % 1 == 0:
            test_acc, test_sum = 0, 0
            with torch.no_grad():
                for inputs, ratings in test_iter:
                    inputs, ratings = inputs.to(device), ratings.to(device)
                    outputs = srm(inputs)

                    test_sum += ratings.shape[0]
                    test_acc += (outputs.argmax(dim=1) == ratings).sum().item()
            end_time = time.time()
            print(loss_sum / batch_num, train_acc / train_sum, test_acc / test_sum, end_time-start_time)

    textList = ["unexpectedly incredible",  "fantastic wine", "not bad", "favorite wine"]

    tokenList = A.getTokens(textList)
    A.addPadding(tokenList, config.max_sumLen)
    tokenList = torch.tensor(tokenList).to(device)

    pred = srm(tokenList)
    print(pred.argmax(dim=1))

