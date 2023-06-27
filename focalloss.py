# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
#
# class Focal_Loss(nn.Module):
#
#     def __init__(self, class_num, alpha=0.25, gamma=2, size_average=True):
#         super(Focal_Loss, self).__init__()
#         if alpha is None:
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = Variable(torch.tensor(alpha))
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average
#
#     def forward(self, inputs, targets):
#         N = inputs.size(0)
#         C = inputs.size(1)
#         P = F.softmax(inputs,dim=1)
#
#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)
#         #print(class_mask)
#
#
#         if inputs.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1)]
#
#         probs = (P*class_mask).sum(1).view(-1,1)
#
#         log_p = probs.log()
#         #print('probs size= {}'.format(probs.size()))
#         #print(probs)
#
#         batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
#         #print('-----bacth_loss------')
#         #print(batch_loss)
#
#
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss
# import torch
#
# class Focal_Loss():
#     def __init__(self, weight=0.25, gamma=2):
#         super(Focal_Loss, self).__init__()
#         self.gamma = gamma
#         self.weight = weight
#
#     def forward(self, preds, labels):
#         """
#         preds:softmax输出结果
#         labels:真实值
#         """
#         eps = 1e-7
#         y_pred = preds.view((preds.size()[0], preds.size()[1], -1))  # B*C*H*W->B*C*(H*W)
#
#         target = labels.view(y_pred.size())  # B*C*H*W->B*C*(H*W)
#
#         ce = -1 * torch.log(y_pred + eps) * target
#         floss = torch.pow((1 - y_pred), self.gamma) * ce
#         floss = torch.mul(floss, self.weight)
#         floss = torch.sum(floss, dim=1)
#         return torch.mean(floss)
import torch
from torch import nn
import torch.nn.functional as F


class Focal_Loss(nn.Module):
    def __init__(self, gamma=2, alpha=0.415, size_average=True):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
      """
        logits = logits[..., None]
        labels = labels[..., None]
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert (logits.size(0) == labels.size(0))
        assert (logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length], device=logits.device).scatter_(1, new_label, 1)

        # calculate log
        log_p = F.log_softmax(logits,dim=1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

