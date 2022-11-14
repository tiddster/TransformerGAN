import torch


class Config():
    def __init__(self):
        self.review_vocab_size = 1835
        self.summary_vocab_size = 1835
        self.max_seqLen = 80
        self.max_sumLen = 5

        self.valid_sum_token_num = 0
        self.valid_rev_token_num = 0

        self.emb_dim = 512

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.commandList = ['不推荐', '一般', '推荐']

config = Config()