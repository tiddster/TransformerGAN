from torch.utils.data import Dataset, DataLoader
import torch

import json
import pandas as pd

amazon_path = "P:\Dataset\\reviews\Amazon\\reviews_Wine\\reviews_Wine.json"
output_path = "P:\TransformerGAN\\generator_output\\output.txt"

max_seqLen = 50
max_sumLen = 10

word2id, id2word, id2rating = {}, [], []


def get_data(path):
    with open(path, 'r') as data_file:
        data = []
        for line in data_file:
            line_data = json.loads(line)
            data.append({'reviewText': line_data['reviewText'], 'overall': line_data['overall'],
                         'summary': line_data['summary'], 'allText': line_data['reviewText'] + line_data['summary']})
        df = pd.DataFrame(data)
    return df


def get_vocab(df):
    for text in df['allText']:
        for word in text:
            id2word.append(word)
    word2id = {word: i for i, word in enumerate(id2word)}

    for overall in df['overall']:
        id2rating.append(int(overall))


amazon_df = get_data(amazon_path)
get_vocab(amazon_df)


def getTokens(textList):
    tokenList = ['<pad>']
    for text in textList:
        tokens = []
        for t in text:
            tokens.append(word2id[t])
        tokenList.append(tokens)
    return tokenList


def getPaddingAndMasks(tokenList, maxLen):
    maskList = []
    for i in range(len(tokenList)):
        if len(tokenList[i]) < maxLen:
            tokenList[i] += [0] * (maxLen - len(tokenList[i]))
            mask = [0] * len(tokenList[i]) + [1] * (maxLen - len(tokenList[i]))
        else:
            tokenList[i] = tokenList[i][:maxLen]
            mask = [1] * maxLen
        maskList.append(mask)
    return maskList


def get_iter(is_discrimination=False):
    pos_text = amazon_df['reviewText']

    if is_discrimination:
        neg_df = get_data(output_path)
        neg_labels = [0 for _ in range(len(list(neg_df['overall'])))]
        pos_labels = [1 for _ in range(len(list(amazon_df['overall'])))]
        textList = pos_text + neg_df['reviewText']

        tokenList = getTokens(textList)
        maskList = getPaddingAndMasks(tokenList)
        labels = pos_labels + neg_labels

        dis_dataset = DisDataset(tokenList, maskList, labels)
        return DataLoader(dis_dataset, batch_size=64, shuffle=True)

    else:
        pos_summary = amazon_df['summary']
        pos_rating = amazon_df['overall']

        tokenList = getTokens(pos_text)
        tokenMaskList = getPaddingAndMasks(tokenList, max_seqLen)
        summaryTokenList = getTokens(pos_summary)
        sumMaskList = getPaddingAndMasks(summaryTokenList, max_sumLen)

        gan_dataset = GanDataset(tokenList, tokenMaskList , summaryTokenList, summaryTokenList, pos_rating)
        return DataLoader(gan_dataset, batch_size=64, shuffle=True)


# 鉴别器的数据集
class DisDataset(Dataset):
    def __init__(self, tokens, mask, labels=[]):
        self.tokens = torch.tensor(tokens).long()
        self.mask = torch.tensor(mask).long()
        self.labels = torch.tensor(labels).long()

    def __getitem__(self, index):
        return self.tokens[index], self.mask[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


# 生成器的数据集
class GanDataset(Dataset):
    def __init__(self, tokens, tokenMask, summary, sumMask, rating):
        self.tokens = torch.tensor(tokens).long()
        self.tokenMask = torch.tensor(tokenMask).long()
        self.summary = torch.tensor(summary).long()
        self.sumMask = torch.tensor(sumMask).long()
        self.rating = torch.tensor(rating).long()

    def __getitem__(self, index):
        return self.tokens[index], self.mask[index], self.summary[index], self.sumMask,self.rating[index]

    def __len__(self):
        return self.rating.shape[0]
