import torch
import torch.nn as nn
import torch.nn.functional as F
class CEMLoss(torch.nn.Module):
    def __init__(self):
        super(CEMLoss, self).__init__()

    def forward(self, logits, target, mask_index=3):  # [bs,num_class]  CE=q*-log(p), q*log(1-p),p=softmax(logits)
        # 获取logit和loss mask矩阵
        logits_mask = CEMLoss.logits_mask(logits)
        loss_mask = CEMLoss.loss_mask(logits, target, mask_index)

        # 将logit经过softmax之后进行log_softmax
        logits_output = logits.masked_fill(logits_mask, -1e9)
        logits_output = F.softmax(logits_output, dim=1)
        logits_log = -1.0 * F.log_softmax(logits_output, dim=1)

        # 创建target onehot
        target = target.reshape(logits.shape[0], 1)
        one_hot = torch.zeros(logits.shape[0], logits.shape[1]).cuda()
        one_hot = one_hot.scatter_(1, target, 1).cuda()

        loss = torch.mul(logits_log, one_hot)
        loss = loss.masked_select(loss_mask).reshape(-1, logits.shape[1])
        loss = loss.sum(dim=1).mean()
        return loss

    @staticmethod
    def reScale(data):
        _range = torch.max(data) - torch.min(data) + 1e-5
        return 0.1 + ((data - torch.min(data)) / _range) * 0.9

    @staticmethod
    def logits_mask(logits):
        logitsMask = torch.zeros(logits.shape)
        logitsMask[:, 1] = 1
        return logitsMask.bool().to(logits.device)

    @staticmethod
    def loss_mask(logits, label, index=3):
        lossMask = torch.ones(logits.shape)
        lossMask[label==index, :] = 0
        return lossMask.bool().to(logits.device)

# loss_fun = CEMLoss()
#
# logits1 = torch.tensor([[-0.03,3,0.39, 0.5],[-0.01,4,0.3, 0.2],[0.16,10,0.1, 0.6], [0.03,9,0.39, 1], [0.03,8, 0.39, 1]],dtype=float,requires_grad=True) # bs,num_class  3,3
# label1 = torch.tensor([2, 2, 1, 3, 3])  # bs
#
# logitsMask = logits_mask(logits1)
# lossMask = loss_mask(logits1, label1)
# print(lossMask)
# print(logitsMask)
#
# loss1 = loss_fun(logits1, label1, logitsMask, lossMask)
#
#
# logits2 = torch.tensor([[-0.03,3,0.39, 0.5],[-0.01,4,0.3, 0.2],[0.16,10,0.1, 0.6], [0.03,9,0.39, 1], [0.03,8, 0.39, 1]],dtype=float,requires_grad=True)
# logits2 = logits2.masked_fill(logitsMask, -1e9)
# logits2 = F.softmax(logits2, dim=1)
# label2 = torch.tensor([2, 2, 1, 3, 3]) # bs
# ce = nn.CrossEntropyLoss(reduction='mean')
#
# loss2=ce(logits2, label2)
#
# loss1.backward()
# loss2.backward()
#
# logits1.grad
# logits2.grad
#
# print(loss1, loss2)