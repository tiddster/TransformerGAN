from torch.utils.data import Dataset, DataLoader
import torch

import json
import pandas as pd
from transformers import BertTokenizer

bart_path = "P:\Dataset\Bert-uncased"

amazon_path = "P:\Dataset\\reviews\Amazon\\reviews_Wine\\reviews_Wine.json"
output_path = "P:\TransformerGAN\\generator_output\\output.txt"

max_seqLen =150
max_sumLen = 10

tokenizer = BertTokenizer.from_pretrained(bart_path)

def get_data(path):
    with open(path, 'r') as data_file:
        data = []
        for line in data_file:
            line_data = json.loads(line)
            data.append({'reviewText': line_data['reviewText'], 'overall': line_data['overall'],
                         'summary': line_data['summary'], 'allText': line_data['reviewText'] + line_data['summary']})
        df = pd.DataFrame(data)
    return df

amazon_df = get_data(amazon_path)


def getTokens(textList):
    tokenList = []
    for text in textList:
        tokens = tokenizer.tokenize(text)
        tokens = tokenizer.convert_tokens_to_ids(tokens)
        tokenList.append(tokens)
    return tokenList


def addPadding(tokenList, maxLen):
    for i in range(len(tokenList)):
        if len(tokenList[i]) < maxLen:
            tokenList[i] += [0] * (maxLen - len(tokenList[i]))
        else:
            tokenList[i] = tokenList[i][:maxLen]


def get_iter(is_discrimination=False):
    pos_text = amazon_df['reviewText']

    if is_discrimination:
        neg_df = get_data(output_path)
        neg_labels = [0 for _ in range(len(list(neg_df['overall'])))]
        pos_labels = [1 for _ in range(len(list(amazon_df['overall'])))]
        textList = pos_text + neg_df['reviewText']

        tokenList = getTokens(textList)
        labels = pos_labels + neg_labels

        dis_dataset = DisDataset(tokenList, labels)
        return DataLoader(dis_dataset, batch_size=64, shuffle=True)

    else:
        pos_summary = amazon_df['summary']
        pos_rating = amazon_df['overall']

        tokenList = getTokens(pos_text)
        addPadding(tokenList, max_seqLen)
        # tokenPaddingList = getPadding(tokenList, max_seqLen)
        # summaryTokenList = getTokens(pos_summary)
        # sumMaskList = getPaddingAndMasks(summaryTokenList, max_sumLen)

        gan_dataset = GanDataset(tokenList, pos_rating)
        return DataLoader(gan_dataset, batch_size=64, shuffle=True)


# 鉴别器的数据集
class DisDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = torch.tensor(tokens).long()
        self.labels = torch.tensor(labels).long()

    def __getitem__(self, index):
        return self.tokens[index], self.mask[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


# 生成器的数据集
class GanDataset(Dataset):
    def __init__(self, tokens, rating):
        self.tokens = torch.tensor(tokens).long()
        self.rating = torch.tensor(rating).long()

    def __getitem__(self, index):
        return self.tokens[index], self.rating[index]

    def __len__(self):
        return self.rating.shape[0]
