from torch.utils.data import Dataset, DataLoader
import torch
import string

import json
import pandas as pd

from Config import config

amazon_path = "P:\Dataset\\reviews\Amazon\\reviews_Wine\\reviews_Wine.json"
output_path = "P:\TransformerGAN\\generator_output\\output.txt"

max_seqLen = config.max_seqLen
max_sumLen = config.max_sumLen


def formulate(textList):
    for i in range(len(textList)):
        textList[i] = textList[i].lower()
        translation = textList[i].maketrans(string.punctuation, " " * len(string.punctuation))
        textList[i] = textList[i].translate(translation)
    return textList


def get_data(path):
    with open(path, 'r') as data_file:
        data = []
        for line in data_file:
            line_data = json.loads(line)
            data.append({'reviewText': line_data['reviewText'], 'overall': line_data['overall'],
                         'summary': line_data['summary'], 'allText': line_data['reviewText'] + line_data['summary']})
        df = pd.DataFrame(data)
        df['reviewText'] = formulate(list(df['reviewText']))
        df['summary'] = formulate(list(df['summary']))
        df['allText'] = formulate(list(df['allText']))
    return df


def get_dicts():
    other = ['<bos>', '<eos>', '<pad>']
    id2word = []
    for text in amazon_df['allText']:
        for word in text.split():
            id2word.append(word)

    id2word = other + list(set(id2word))
    word2id = {word: i for i, word in enumerate(id2word)}

    return id2word, word2id


amazon_df = get_data(amazon_path)
id2word, word2id = get_dicts()
config.vocab_size = len(id2word)


def getTokens(textList):
    tokenList = []
    for text in textList:
        tokens = []
        for word in text.split():
            if word in id2word:
                token = word2id[word]
                tokens.append(token)
        tokenList.append(tokens)
    return tokenList


def addPadding(tokenList, maxLen):
    for i in range(len(tokenList)):
        if len(tokenList[i]) < maxLen:
            tokenList[i] = [0] + tokenList[i] + [1] + [2] * (maxLen - len(tokenList[i]))
        else:
            tokenList[i] = [0] + tokenList[i][:maxLen] + [1]


def get_gan_iter():
    print("111")
    revText = amazon_df['reviewText']
    sumText = amazon_df['summary']

    print("222")
    revTokenList = getTokens(revText)
    addPadding(revTokenList, max_seqLen)

    for list in revTokenList:
        for t in list:
            if t != 2:
                config.valid_rev_token_num += 1

    print("333")
    sumTokenList = getTokens(sumText)
    addPadding(sumTokenList, max_sumLen)

    sumTokenList = torch.tensor(sumTokenList)

    gan_dataset = GanDataset(revTokenList, sumTokenList, sumTokenList)
    return DataLoader(gan_dataset, batch_size=4, shuffle=True, num_workers=1)


def get_summary_iter():
    rating = amazon_df['overall'] - 1
    newRating = []
    for r in rating:
        if r == 0 or r == 1:
            newRating.append(0)
        elif r == 2:
            newRating.append(1)
        else:
            newRating.append(2)
    summary = amazon_df['summary']

    # 计算有效token量，防止计算出来的loss偏大
    for list in summary:
        for t in list:
            if t != 2:
                config.valid_sum_token_num += 1

    tokenList = getTokens(summary)
    addPadding(tokenList, max_sumLen)
    sumDataset = SummaryDataset(tokenList, newRating)
    return DataLoader(sumDataset, batch_size=64, shuffle=True, num_workers=1)


class SummaryDataset(Dataset):
    def __init__(self, sumTokens, ratings):
        self.sumTokens = torch.tensor(sumTokens).long()
        self.ratings = torch.tensor(ratings).long()

    def __getitem__(self, index):
        return self.sumTokens[index], self.ratings[index]

    def __len__(self):
        return self.ratings.shape[0]


# # 鉴别器的数据集
# class DisDataset(Dataset):
#     def __init__(self, tokens, labels):
#         self.tokens = torch.tensor(tokens).long()
#         self.labels = torch.tensor(labels).long()
#
#     def __getitem__(self, index):
#         return self.tokens[index], self.mask[index], self.labels[index]
#
#     def __len__(self):
#         return self.labels.shape[0]


# 生成器的数据集
class GanDataset(Dataset):
    def __init__(self, tokens, sumTokensX, sumTokensY):
        self.tokens = torch.tensor(tokens).long()
        self.sumTokensX = torch.tensor(sumTokensX).long()
        self.sumTokensY = torch.tensor(sumTokensY).long()

    def __getitem__(self, index):
        return self.tokens[index], self.sumTokensX[index], self.sumTokensY[index]

    def __len__(self):
        return self.tokens.shape[0]
